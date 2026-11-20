//! Online hippocampus: `m = W2 SiLU(W1 q)`.
//!
//! Fast-path reads go through [`SharedWeightArena`] and never take a mutex.
//! Plasticity writes the shadow slot. [`SharedWeightArena::flush`] publishes
//! it. A repeat of content whose surprise has fallen under `τ_low` is dropped
//! and does not bump `version`.
//!
//! Frozen 1.58-bit projections are not stored inside θ (θ is FP32). Callers
//! that need a ternary query — the macro-cycle memory block — run
//! [`cortex_kernels::BitLinearWeights`] first and pass that activation here.
//! `bitlinear_query_then_bind` is the crate-level check that this link is real.

mod arena;
mod mlp;
mod surprise;
mod wal;

pub use arena::SharedWeightArena;
pub use mlp::HippocampusWeights;
pub use surprise::{SurpriseBand, SurpriseThresholds};
pub use wal::{Episode, WalArchive};

/// Working-memory window. Cleared independently of θ. Retrieval after
/// [`ContextWindow::clear`] must come from the weights alone.
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

#[derive(Clone, Copy, Debug)]
pub struct BindConfig {
    pub thresholds: SurpriseThresholds,
    pub lr_w1: f32,
    pub momentum: f32,
    pub forget: f32,
}

impl Default for BindConfig {
    fn default() -> Self {
        Self {
            thresholds: SurpriseThresholds::default(),
            lr_w1: 1e-3,
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

/// Score `(k, v)` against the **active** snapshot. On `Local` or `High`, write
/// the shadow slot (version increments there). The caller decides when to
/// [`SharedWeightArena::flush`]. High-band episodes are appended to `wal`
/// when one is provided.
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
    arena.write_shadow(|w| {
        w.bind_step(k, v, cfg.lr_w1, cfg.momentum, cfg.forget);
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
                tau_high: 1.0e6, // local write, WAL not required for this assertion
            },
            lr_w1: 1e-3,
            momentum: 0.0,
            forget: 0.0,
        };
        let outcome = inject(&arena, &k, &v, &ctx, &cfg, None);
        assert!(
            outcome.wrote,
            "first presentation must write, surprise={}",
            outcome.surprise
        );
        arena.flush();
        ctx.clear();
        assert!(ctx.tokens.is_empty());
        assert!(ctx.trace.is_empty());
        let mse1 = arena.read(|w| w.mean_square_error(&k, &v));
        let drop = (mse0 - mse1) / mse0;
        assert!(
            drop > 0.80,
            "retrieval MSE drop {drop} (mse {mse0} -> {mse1}) after flush+context clear"
        );
        let version = arena.read(|w| w.version);
        assert!(version >= 1);
        let again = inject(&arena, &k, &v, &ctx, &cfg, None);
        assert_eq!(again.band, SurpriseBand::Drop);
        assert!(!again.wrote);
        let version2 = arena.read(|w| w.version);
        assert_eq!(
            version, version2,
            "repeat of the same content must not increment version"
        );
    }

    #[test]
    fn bitlinear_query_then_bind() {
        let proj = BitLinearWeights::from_seed(32, 32, 9);
        let x = vec![0.15f32; 32];
        let q = proj.forward(&x);
        assert_eq!(q.len(), 32);
        let mut hip = HippocampusWeights::new(32, 48, 32);
        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let mse0 = hip.mean_square_error(&q, &v);
        hip.bind_step(&q, &v, 0.0, 0.0, 0.0);
        let mse1 = hip.mean_square_error(&q, &v);
        assert!((mse0 - mse1) / mse0 > 0.80);
        // The projection is a real kernel call: scalar and dispatched paths agree.
        let q2 = proj.forward_scalar(&x);
        let err = q
            .iter()
            .zip(q2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-5);
    }
}
