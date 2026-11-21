//! Online hippocampus: `m = W2 SiLU(W1 L2Norm(q))` with lock-free Fast-Path
//! reads and an asynchronous background Plasticity-Path worker.
//!
//! Causal contract per streaming token:
//! 1. **Read first**: Fast-Path reads `m_t` from [`SharedWeightArena`] with
//!    zero mutex contention.
//! 2. **Gate & fuse**: caller blends `m_t` into the residual stream.
//! 3. **Write async**: `(k_t, v_t)` is dispatched via a bounded non-blocking
//!    channel (`try_send`) to the background [`PlasticityWorker`] thread,
//!    which evaluates 3-tier surprise gating (`Drop` / `Local` / `High` with
//!    LR boost + binary WAL archiving) and flips the double-buffered arena.

mod arena;
mod mlp;
mod surprise;
mod wal;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub use arena::SharedWeightArena;
pub use mlp::{matvec, HippocampusWeights};
pub use surprise::{SurpriseBand, SurpriseThresholds};
pub use wal::{Episode, SnapshotStore, WalArchive};

/// Working-memory window. Cleared independently of θ.
#[derive(Clone, Debug, Default)]
pub struct ContextWindow {
    pub tokens: Vec<u32>,
    pub trace: Vec<f32>,
}

impl ContextWindow {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.trace.clear();
    }
}

/// Configuration for synchronous [`inject`].
#[derive(Clone, Copy, Debug)]
pub struct BindConfig {
    pub thresholds: SurpriseThresholds,
    pub lr_w1: f32,
    pub boost_multiplier: f32,
    pub momentum: f32,
    pub forget: f32,
}

impl Default for BindConfig {
    fn default() -> Self {
        Self {
            thresholds: SurpriseThresholds::default(),
            lr_w1: 1e-3,
            boost_multiplier: 2.5,
            momentum: 0.0,
            forget: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BindOutcome {
    pub band: SurpriseBand,
    pub surprise: f32,
    pub version_before: u64,
    pub wrote: bool,
}

/// Score `(k, v)` against the **active** snapshot.
/// * `Drop`: `< τ_low`, skip update.
/// * `Local`: `[τ_low, τ_high)`, write shadow slot at `lr_w1`.
/// * `High`: `>= τ_high`, write shadow slot with boosted LR (`lr_w1 * boost_multiplier`)
///   and append to `wal` when provided.
pub fn inject(
    arena: &SharedWeightArena,
    k: &[f32],
    v: &[f32],
    ctx: &ContextWindow,
    cfg: &BindConfig,
    wal: Option<&mut WalArchive>,
) -> BindOutcome {
    let (surprise, version_before) = arena.read(|w| (w.surprise(k, v), w.version));
    let band = cfg.thresholds.band(surprise);
    if band == SurpriseBand::Drop {
        return BindOutcome {
            band,
            surprise,
            version_before,
            wrote: false,
        };
    }
    let eff_lr = if band == SurpriseBand::High {
        cfg.lr_w1 * cfg.boost_multiplier
    } else {
        cfg.lr_w1
    };
    arena.write_shadow(|w| {
        w.bind_step(k, v, eff_lr, cfg.momentum, cfg.forget);
    });
    if band == SurpriseBand::High {
        if let Some(wal) = wal {
            let ep = Episode {
                surprise,
                key: k.to_vec(),
                value: v.to_vec(),
                context: ctx.tokens.clone(),
            };
            let _ = wal.append(&ep);
        }
    }
    BindOutcome {
        band,
        surprise,
        version_before,
        wrote: true,
    }
}

// ============================================================================
// Asynchronous Background Plasticity Worker & Lock-Free Handle
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct PlasticityConfig {
    pub surprise_low: f32,
    pub surprise_high: f32,
    pub base_lr: f32,
    pub boost_multiplier: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub commit_every_n: usize,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            surprise_low: 0.02,
            surprise_high: 0.25,
            base_lr: 0.08,
            boost_multiplier: 2.5,
            momentum: 0.85,
            weight_decay: 1e-4,
            commit_every_n: 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WriteReport {
    pub surprise: f32,
    pub updated: bool,
    pub archived_high_surprise: bool,
    pub committed_version: u64,
}

pub struct PlasticityWorker {
    pub cfg: PlasticityConfig,
    working: HippocampusWeights,
    k_norm: Vec<f32>,
    z1_buf: Vec<f32>,
    h1_buf: Vec<f32>,
    err_buf: Vec<f32>,
    grad_z1_buf: Vec<f32>,
    pending_updates: usize,
}

impl PlasticityWorker {
    pub fn new(initial: HippocampusWeights, cfg: PlasticityConfig) -> Self {
        let (d_in, hd, d_out) = (initial.d_in, initial.d_hidden, initial.d_out);
        Self {
            cfg,
            working: initial,
            k_norm: vec![0.0; d_in],
            z1_buf: vec![0.0; hd],
            h1_buf: vec![0.0; hd],
            err_buf: vec![0.0; d_out],
            grad_z1_buf: vec![0.0; hd],
            pending_updates: 0,
        }
    }

    /// Zero-allocation online outer-product step on `(k, v)`.
    pub fn learn_pair(
        &mut self,
        k: &[f32],
        v: &[f32],
        force_flush: bool,
        arena: &SharedWeightArena,
    ) -> WriteReport {
        let (surprise, updated, high) = self.working.learn_outer_product_with_scratch(
            k,
            v,
            self.cfg.surprise_low,
            self.cfg.surprise_high,
            self.cfg.base_lr,
            self.cfg.boost_multiplier,
            self.cfg.momentum,
            self.cfg.weight_decay,
            &mut self.k_norm,
            &mut self.z1_buf,
            &mut self.h1_buf,
            &mut self.err_buf,
            &mut self.grad_z1_buf,
        );
        if !updated {
            if force_flush && self.pending_updates > 0 {
                self.commit_to_arena(arena);
            }
            return WriteReport {
                surprise,
                updated: false,
                archived_high_surprise: false,
                committed_version: self.working.version,
            };
        }
        self.pending_updates += 1;
        if force_flush || self.pending_updates >= self.cfg.commit_every_n.max(1) {
            self.commit_to_arena(arena);
        }
        WriteReport {
            surprise,
            updated: true,
            archived_high_surprise: high,
            committed_version: self.working.version,
        }
    }

    fn commit_to_arena(&mut self, arena: &SharedWeightArena) {
        self.working.version = self.working.version.wrapping_add(1);
        self.pending_updates = 0;
        arena.publish_from_writer(&self.working);
    }
}

enum BusCommand {
    Learn {
        k: Vec<f32>,
        v: Vec<f32>,
        flush: bool,
        context: Vec<u32>,
    },
    Replace {
        weights: HippocampusWeights,
    },
    AttachWal {
        dir: PathBuf,
    },
    BarrierSync(SyncSender<u64>),
    Shutdown,
}

/// Lock-free Fast-Path reader + bounded non-blocking Plasticity-Path channel
/// backed by a dedicated worker thread.
pub struct AsyncHippocampusHandle {
    pub arena: Arc<SharedWeightArena>,
    tx: SyncSender<BusCommand>,
    worker_join: Option<JoinHandle<()>>,
}

impl AsyncHippocampusHandle {
    pub fn spawn(initial: HippocampusWeights, cfg: PlasticityConfig, queue_cap: usize) -> Self {
        Self::spawn_inner(initial, cfg, queue_cap, None)
    }

    pub fn spawn_with_wal(
        initial: HippocampusWeights,
        cfg: PlasticityConfig,
        queue_cap: usize,
        wal_dir: &Path,
    ) -> Self {
        Self::spawn_inner(initial, cfg, queue_cap, Some(wal_dir.to_path_buf()))
    }

    fn spawn_inner(
        initial: HippocampusWeights,
        cfg: PlasticityConfig,
        queue_cap: usize,
        wal_dir: Option<PathBuf>,
    ) -> Self {
        let arena = Arc::new(SharedWeightArena::new(initial.clone()));
        let arena_bg = Arc::clone(&arena);
        let (tx, rx): (SyncSender<BusCommand>, Receiver<BusCommand>) =
            mpsc::sync_channel(queue_cap.max(4));

        let handle = thread::spawn(move || {
            let mut worker = PlasticityWorker::new(initial, cfg);
            let mut wal = wal_dir.and_then(|d| WalArchive::create(&d, Duration::from_secs(5)).ok());
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    BusCommand::Learn {
                        k,
                        v,
                        flush,
                        context,
                    } => {
                        let report = worker.learn_pair(&k, &v, flush, &arena_bg);
                        if report.archived_high_surprise {
                            if let Some(w) = wal.as_mut() {
                                let _ = w.append(&Episode {
                                    surprise: report.surprise,
                                    key: k,
                                    value: v,
                                    context,
                                });
                            }
                        }
                    }
                    BusCommand::Replace { mut weights } => {
                        weights.reset_momentum();
                        worker.working = weights.clone();
                        worker.pending_updates = 0;
                        arena_bg.replace(weights);
                    }
                    BusCommand::AttachWal { dir } => {
                        if let Some(w) = wal.as_mut() {
                            let _ = w.maybe_fsync(true);
                        }
                        wal = WalArchive::create(&dir, Duration::from_secs(5)).ok();
                    }
                    BusCommand::BarrierSync(ack) => {
                        if worker.pending_updates > 0 {
                            worker.commit_to_arena(&arena_bg);
                        }
                        if let Some(w) = wal.as_mut() {
                            let _ = w.maybe_fsync(true);
                        }
                        let _ = ack.send(worker.working.version);
                    }
                    BusCommand::Shutdown => {
                        if let Some(w) = wal.as_mut() {
                            let _ = w.maybe_fsync(true);
                        }
                        break;
                    }
                }
            }
        });

        Self {
            arena,
            tx,
            worker_join: Some(handle),
        }
    }

    /// Lock-free Fast-Path forward read (`q` is L2-normalized inside `forward_into`).
    #[inline]
    pub fn read_forward(&self, q: &[f32], out: &mut [f32]) -> u64 {
        self.arena.read(|w| {
            w.forward_into(q, out);
            w.version
        })
    }

    /// Non-blocking Plasticity-Path submission; drops gracefully if queue is full
    /// so Fast-Path token latency is never stalled.
    #[inline]
    pub fn submit_learn(&self, k: Vec<f32>, v: Vec<f32>, flush: bool, context: Vec<u32>) {
        let _ = self.tx.try_send(BusCommand::Learn {
            k,
            v,
            flush,
            context,
        });
    }

    /// Attach or switch the directory where high-surprise WAL episodes are written.
    pub fn attach_wal_dir(&self, dir: &Path) {
        let _ = self.tx.send(BusCommand::AttachWal {
            dir: dir.to_path_buf(),
        });
    }

    /// Synchronously replace both slots and worker weights (used by sleep
    /// orthogonalization and snapshot rollback).
    pub fn replace_weights(&self, weights: HippocampusWeights) -> u64 {
        let _ = self.tx.send(BusCommand::Replace { weights });
        self.flush_barrier()
    }

    pub fn snapshot(&self) -> HippocampusWeights {
        self.arena.snapshot_active()
    }

    pub fn version(&self) -> u64 {
        self.arena.read(|w| w.version)
    }

    pub fn flush_barrier(&self) -> u64 {
        let (ack_tx, ack_rx) = mpsc::sync_channel(1);
        let _ = self.tx.send(BusCommand::BarrierSync(ack_tx));
        ack_rx.recv().unwrap_or(0)
    }
}

impl Drop for AsyncHippocampusHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(BusCommand::Shutdown);
        if let Some(h) = self.worker_join.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex_kernels::BitLinearWeights;

    fn key_value() -> (Vec<f32>, Vec<f32>) {
        let k: Vec<f32> = (0..32)
            .map(|i| if i % 2 == 0 { 0.35 } else { -0.2 })
            .collect();
        let v: Vec<f32> = (0..32)
            .map(|i| if i % 3 == 0 { 1.0 } else { -0.75 })
            .collect();
        (k, v)
    }

    #[test]
    fn test_oneshot_hippocampus_binding() {
        let weights = HippocampusWeights::new(32, 64, 32);
        let arena = SharedWeightArena::new(weights);
        let mut ctx = ContextWindow::new();
        let (k, v) = key_value();
        ctx.tokens.extend_from_slice(&[11, 22, 33]);
        ctx.trace.extend_from_slice(&k);
        let mse0 = arena.read(|w| w.mean_square_error(&k, &v));
        assert!(
            mse0 > 1e-3,
            "initial mse {mse0} should not already be bound"
        );

        let cfg = BindConfig {
            thresholds: SurpriseThresholds {
                tau_low: 1e-5,
                tau_high: 0.1,
            },
            lr_w1: 1e-3,
            boost_multiplier: 2.5,
            momentum: 0.5,
            forget: 0.0,
        };
        let dir = std::env::temp_dir().join(format!("cortex-hip-test-{}", std::process::id()));
        let mut wal = WalArchive::create(&dir, Duration::from_secs(5)).unwrap();
        let outcome = inject(&arena, &k, &v, &ctx, &cfg, Some(&mut wal));
        assert!(outcome.wrote);
        assert_eq!(outcome.band, SurpriseBand::High);
        arena.flush();
        ctx.clear();

        let mse1 = arena.read(|w| w.mean_square_error(&k, &v));
        let drop = (mse0 - mse1) / mse0;
        assert!(drop > 0.80, "retrieval MSE drop {drop} ({mse0} -> {mse1})");
        assert_eq!(wal.read_all().unwrap().len(), 1);

        let version = arena.read(|w| w.version);
        let again = inject(&arena, &k, &v, &ctx, &cfg, Some(&mut wal));
        assert_eq!(again.band, SurpriseBand::Drop);
        assert!(!again.wrote);
        assert_eq!(arena.read(|w| w.version), version);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn async_plasticity_worker_binds_and_writes_wal() {
        let dir = std::env::temp_dir().join(format!("cortex-async-hip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = PlasticityConfig {
            surprise_low: 0.005,
            surprise_high: 0.10,
            base_lr: 2.0,
            boost_multiplier: 2.5,
            momentum: 0.8,
            weight_decay: 0.0,
            commit_every_n: 2,
        };
        let handle = AsyncHippocampusHandle::spawn_with_wal(
            HippocampusWeights::new(32, 64, 32),
            cfg,
            64,
            &dir,
        );
        let (k, v) = key_value();
        let mse_before = handle.arena.read(|w| w.mean_square_error(&k, &v));
        for step in 0..12 {
            handle.submit_learn(k.clone(), v.clone(), step == 11, vec![1, 2, 3]);
        }
        let ver = handle.flush_barrier();
        assert!(ver >= 1);
        let mse_after = handle.arena.read(|w| w.mean_square_error(&k, &v));
        assert!(
            mse_after < mse_before * 0.2,
            "async worker failed to converge: {mse_before} -> {mse_after}"
        );
        let (eps, trunc) = WalArchive::read_file(&dir.join("episodes.wal")).unwrap();
        assert!(
            !trunc && !eps.is_empty(),
            "high-surprise episodes must land in WAL"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bitlinear_query_then_bind() {
        let proj = BitLinearWeights::from_seed(32, 32, 9);
        let x = vec![0.15f32; 32];
        let q = proj.forward(&x);
        let mut hip = HippocampusWeights::new(32, 48, 32);
        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let mse0 = hip.mean_square_error(&q, &v);
        hip.bind_step(&q, &v, 0.0, 0.0, 0.0);
        let mse1 = hip.mean_square_error(&q, &v);
        assert!((mse0 - mse1) / mse0 > 0.80);
        let q2 = proj.forward_scalar(&x);
        let err = q
            .iter()
            .zip(q2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-5);
    }
}
