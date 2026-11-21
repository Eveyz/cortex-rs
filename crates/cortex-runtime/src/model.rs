//! Macro-cycle Cortex-Local model assembly.
//!
//! Each macro cycle stacks:
//! * Layers 1–3: [`BitSsmBlock`] (1.58-bit QKV, O(d²) Gated DeltaNet, 1.58-bit SwiGLU)
//! * Layer 4: [`MemoryBlock`] — INT8 sliding-window ring attention (`W = 512`,
//!   Branch A) + lock-free online neural hippocampus (`m = W2 SiLU(W1 L2Norm(q))`,
//!   Branch B, causal read-before-async-write), gated residual, then 1.58-bit SwiGLU.
//!
//! Input token embeddings and the output LM head share a single symmetric INT8
//! matrix `emb_i8` (`[-127, +127]`) plus per-row FP32 scales `emb_scale`
//! (weight tying). MeZO's continuous parameter vector `φ` comprises all
//! `BitLinearWeights::row_scales`, `emb_scale`, and `RmsNorm` weights, mutated
//! strictly in place via [`CortexModel::walk_phi_mut`] with zero parameter copies.

use std::path::Path;

use cortex_hippocampus::{
    AsyncHippocampusHandle, HippocampusWeights, PlasticityConfig, SnapshotStore,
};
use cortex_kernels::{l2_normalize_inplace, sigmoid, Aligned64, BitLinearWeights};
use cortex_sleep::{consolidate_hippocampus, rademacher, InPlacePhi, MezoScaleTuner};
use cortex_ssm::{BitSsmBlock, BitSwiGLU, DeltaState, RingSlidingAttention, RingState, RmsNorm};

use crate::sample::cross_entropy;

#[derive(Clone, Debug)]
pub struct CortexConfig {
    pub n_macros: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub window: usize,
    pub ffn_hidden: usize,
    pub hip_hidden: usize,
    pub rms_eps: f32,
}

impl CortexConfig {
    /// Production 1.6B-class shape: 6 macros = 24 layers, `d_model = 2048`, `vocab = 32768`, `W = 512`.
    pub fn prod() -> Self {
        Self {
            n_macros: 6,
            d_model: 2048,
            n_heads: 32,
            head_dim: 64,
            vocab: 32768,
            window: 512,
            ffn_hidden: 5504,
            hip_hidden: 4096,
            rms_eps: 1e-6,
        }
    }

    /// Micro config: 2 macros = 8 layers, `d_model = 256`, `vocab = 266` (256 bytes + 10 specials), `W = 512`.
    pub fn micro() -> Self {
        Self {
            n_macros: 2,
            d_model: 256,
            n_heads: 4,
            head_dim: 64,
            vocab: 266,
            window: 512,
            ffn_hidden: 512,
            hip_hidden: 512,
            rms_eps: 1e-6,
        }
    }

    /// 2 macros × 4 layers = 8, `d_model = 256`, `W = 512`.
    pub fn test_256() -> Self {
        Self {
            n_macros: 2,
            d_model: 256,
            n_heads: 8,
            head_dim: 32,
            vocab: 266,
            window: 512,
            ffn_hidden: 512,
            hip_hidden: 128,
            rms_eps: 1e-6,
        }
    }

    /// Compact 8-layer net with real ternary weights for fast MeZO & CLI tests.
    pub fn mezo_tiny() -> Self {
        Self {
            n_macros: 2,
            d_model: 32,
            n_heads: 4,
            head_dim: 8,
            vocab: 16,
            window: 8,
            ffn_hidden: 32,
            hip_hidden: 16,
            rms_eps: 1e-6,
        }
    }

    /// Step 1 sandbox: 2 macros = 8 layers, `d_model = 512`, `vocab = 4096`.
    pub fn step1_stories() -> Self {
        Self {
            n_macros: 2,
            d_model: 512,
            n_heads: 8,
            head_dim: 64,
            vocab: 4096,
            window: 64,
            ffn_hidden: 1120,
            hip_hidden: 128,
            rms_eps: 1e-6,
        }
    }

    pub fn mini() -> Self {
        Self {
            n_macros: 1,
            d_model: 16,
            n_heads: 2,
            head_dim: 8,
            vocab: 32,
            window: 8,
            ffn_hidden: 16,
            hip_hidden: 8,
            rms_eps: 1e-6,
        }
    }

    pub fn n_layers(&self) -> usize {
        self.n_macros * 4
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_heads == 0 || self.n_heads * self.head_dim != self.d_model {
            return Err(format!(
                "n_heads {} * head_dim {} != d_model {}",
                self.n_heads, self.head_dim, self.d_model
            ));
        }
        if self.vocab == 0 || self.ffn_hidden == 0 || self.hip_hidden == 0 || self.window == 0 {
            return Err("empty hyperparameter".into());
        }
        Ok(())
    }
}

pub struct MemoryBlock {
    pub norm1: RmsNorm,
    pub norm2: RmsNorm,
    pub attn: RingSlidingAttention,
    pub hip_proj: BitLinearWeights,
    pub hip_k: BitLinearWeights,
    pub hip_v: BitLinearWeights,
    pub hip: AsyncHippocampusHandle,
    pub gate_w: Aligned64<f32>,
    pub gate_b: Aligned64<f32>,
    pub ffn: BitSwiGLU,
    pub d_model: usize,
}

impl MemoryBlock {
    pub(crate) fn zeros(cfg: &CortexConfig) -> Self {
        Self {
            norm1: RmsNorm::ones(cfg.d_model, cfg.rms_eps),
            norm2: RmsNorm::ones(cfg.d_model, cfg.rms_eps),
            attn: RingSlidingAttention::zeros(cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.window),
            hip_proj: BitLinearWeights::zeros(cfg.d_model, cfg.d_model),
            hip_k: BitLinearWeights::zeros(cfg.d_model, cfg.d_model),
            hip_v: BitLinearWeights::zeros(cfg.d_model, cfg.d_model),
            hip: AsyncHippocampusHandle::spawn(
                HippocampusWeights::new(cfg.d_model, cfg.hip_hidden, cfg.d_model),
                PlasticityConfig::default(),
                64,
            ),
            gate_w: Aligned64::zeros(cfg.d_model),
            gate_b: Aligned64::zeros(cfg.d_model),
            ffn: BitSwiGLU::zeros(cfg.d_model, cfg.ffn_hidden),
            d_model: cfg.d_model,
        }
    }

    fn from_seed(cfg: &CortexConfig, seed: u32) -> Self {
        let mut gate_w = Aligned64::zeros(cfg.d_model);
        let mut gate_b = Aligned64::zeros(cfg.d_model);
        for (i, s) in gate_w.as_mut_slice().iter_mut().enumerate() {
            *s = det(i, seed) * 0.02;
        }
        for (i, s) in gate_b.as_mut_slice().iter_mut().enumerate() {
            *s = det(i, seed.wrapping_add(3)) * 0.02;
        }
        let inv_sqrt = (cfg.d_model as f32).sqrt().recip() * 0.5;
        let scale_proj = |mut w: BitLinearWeights| -> BitLinearWeights {
            w.row_scales.fill(inv_sqrt);
            w
        };
        Self {
            norm1: RmsNorm::ones(cfg.d_model, cfg.rms_eps),
            norm2: RmsNorm::ones(cfg.d_model, cfg.rms_eps),
            attn: RingSlidingAttention::from_seed(
                cfg.d_model,
                cfg.n_heads,
                cfg.head_dim,
                cfg.window,
                seed.wrapping_add(10),
            ),
            hip_proj: scale_proj(BitLinearWeights::from_seed(
                cfg.d_model,
                cfg.d_model,
                seed.wrapping_add(20),
            )),
            hip_k: scale_proj(BitLinearWeights::from_seed(
                cfg.d_model,
                cfg.d_model,
                seed.wrapping_add(21),
            )),
            hip_v: scale_proj(BitLinearWeights::from_seed(
                cfg.d_model,
                cfg.d_model,
                seed.wrapping_add(22),
            )),
            hip: AsyncHippocampusHandle::spawn(
                HippocampusWeights::new_seeded(cfg.d_model, cfg.hip_hidden, cfg.d_model, seed),
                PlasticityConfig::default(),
                64,
            ),
            gate_w,
            gate_b,
            ffn: BitSwiGLU::from_seed(cfg.d_model, cfg.ffn_hidden, seed.wrapping_add(30)),
            d_model: cfg.d_model,
        }
    }

    /// Pure read-only forward (used by deterministic `logits` self-check & MeZO).
    pub fn forward(&self, x: &[f32], ring: &mut RingState) -> Vec<f32> {
        self.forward_step(x, ring, false, false, &[])
    }

    /// Full causal MemoryBlock step:
    /// 1. Branch A: INT8 sliding-window ring attention
    /// 2. Branch B: L2-normalize `q_t` -> lock-free read `m_t` from old memory
    /// 3. Gated residual blend `x + g * a + (1 - g) * m_t`
    /// 4. Async non-blocking dispatch `(L2Norm(k_t), v_t)` to background plasticity worker
    /// 5. Pre-norm 1.58-bit SwiGLU FFN
    pub fn forward_step(
        &self,
        x: &[f32],
        ring: &mut RingState,
        learn: bool,
        is_eos: bool,
        context: &[u32],
    ) -> Vec<f32> {
        let xn = self.norm1.forward(x);
        let a = self.attn.forward(&xn, ring);
        let mut q = self.hip_proj.forward(&xn);
        l2_normalize_inplace(&mut q);

        // [1. Read old memory first — lock-free]
        let mut m = vec![0f32; self.d_model];
        self.hip.read_forward(&q, &mut m);

        // [2. Gate & fuse]
        let gw = self.gate_w.as_slice();
        let gb = self.gate_b.as_slice();
        let mut y = vec![0f32; self.d_model];
        for i in 0..self.d_model {
            let g = sigmoid(gw[i] * xn[i] + gb[i]);
            y[i] = x[i] + g * a[i] + (1.0 - g) * m[i];
        }

        // [3. Async write new memory — non-blocking background thread]
        if learn {
            let mut k = self.hip_k.forward(&xn);
            l2_normalize_inplace(&mut k);
            let v = self.hip_v.forward(&xn);
            self.hip.submit_learn(k, v, is_eos, context.to_vec());
        }

        let xn2 = self.norm2.forward(&y);
        let ffn = self.ffn.forward(&xn2);
        for i in 0..self.d_model {
            y[i] += ffn[i];
        }
        y
    }
}

pub struct ModelState {
    pub ssm: Vec<DeltaState>,
    pub rings: Vec<RingState>,
}

impl ModelState {
    pub fn state_bytes(&self) -> usize {
        self.ssm.iter().map(|s| s.state_bytes()).sum::<usize>()
            + self.rings.iter().map(|r| r.state_bytes()).sum::<usize>()
    }

    pub fn reset(&mut self) {
        for s in &mut self.ssm {
            s.reset();
        }
        for r in &mut self.rings {
            r.reset();
        }
    }
}

pub struct CortexModel {
    pub cfg: CortexConfig,
    /// Weight-tied symmetric INT8 embedding / LM head (`[vocab, d_model]`, `[-127, +127]`).
    pub emb_i8: Vec<i8>,
    pub emb_scale: Aligned64<f32>,
    pub ssm: Vec<BitSsmBlock>,
    pub mem: Vec<MemoryBlock>,
    pub final_norm: RmsNorm,
}

impl CortexModel {
    pub fn zeros(cfg: CortexConfig) -> Self {
        cfg.validate().expect("config");
        let mut emb_scale = Aligned64::zeros(cfg.vocab);
        emb_scale.fill(1.0);
        let ssm = (0..cfg.n_macros * 3)
            .map(|_| {
                BitSsmBlock::zeros(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.head_dim,
                    cfg.ffn_hidden,
                    cfg.rms_eps,
                )
            })
            .collect();
        let mem = (0..cfg.n_macros)
            .map(|_| MemoryBlock::zeros(&cfg))
            .collect();
        let final_norm = RmsNorm::ones(cfg.d_model, cfg.rms_eps);
        Self {
            emb_i8: vec![0i8; cfg.vocab * cfg.d_model],
            emb_scale,
            ssm,
            mem,
            final_norm,
            cfg,
        }
    }

    pub fn from_seed(cfg: CortexConfig, seed: u32) -> Self {
        cfg.validate().expect("config");
        let mut model = Self::zeros(cfg);
        for (i, slot) in model.emb_i8.iter_mut().enumerate() {
            let mut x = (i as u32).wrapping_mul(0x9E37_79B1) ^ seed;
            x ^= x >> 16;
            x = x.wrapping_mul(0x7FEB_352D);
            // Symmetric INT8 in [-127, +127] — never -128.
            *slot = ((x % 255) as i32 - 127) as i8;
        }
        let emb_unit = (model.cfg.d_model as f32).sqrt().recip() / 127.0;
        model.emb_scale.fill(emb_unit);
        let scale_ssm = (model.cfg.d_model as f32).sqrt().recip() * 0.05;
        model.ssm = (0..model.cfg.n_macros * 3)
            .map(|i| {
                let mut b = BitSsmBlock::from_seed(
                    model.cfg.d_model,
                    model.cfg.n_heads,
                    model.cfg.head_dim,
                    model.cfg.ffn_hidden,
                    model.cfg.rms_eps,
                    seed.wrapping_add(100 + i as u32 * 17),
                );
                b.q_proj.row_scales.fill(scale_ssm);
                b.k_proj.row_scales.fill(scale_ssm);
                b.v_proj.row_scales.fill(scale_ssm);
                b.o_proj.row_scales.fill(scale_ssm);
                b.ffn.w_gate.row_scales.fill(scale_ssm);
                b.ffn.w_up.row_scales.fill(scale_ssm);
                b.ffn.w_down.row_scales.fill(scale_ssm);
                b
            })
            .collect();
        model.mem = (0..model.cfg.n_macros)
            .map(|i| {
                let mut m =
                    MemoryBlock::from_seed(&model.cfg, seed.wrapping_add(500 + i as u32 * 19));
                m.attn.w_q.row_scales.fill(scale_ssm);
                m.attn.w_k.row_scales.fill(scale_ssm);
                m.attn.w_v.row_scales.fill(scale_ssm);
                m.attn.w_o.row_scales.fill(scale_ssm);
                m.ffn.w_gate.row_scales.fill(scale_ssm);
                m.ffn.w_up.row_scales.fill(scale_ssm);
                m.ffn.w_down.row_scales.fill(scale_ssm);
                m
            })
            .collect();
        model
    }

    pub fn n_layers(&self) -> usize {
        self.cfg.n_layers()
    }

    pub fn new_state(&self) -> ModelState {
        ModelState {
            ssm: self.ssm.iter().map(|b| b.new_state()).collect(),
            rings: self.mem.iter().map(|m| m.attn.new_state()).collect(),
        }
    }

    /// Attach a WAL root directory across all macro hippocampus workers
    /// (`{wal_root}/macro_{m}/episodes.wal`).
    pub fn attach_wal_dir(&self, wal_root: &Path) {
        for (m, block) in self.mem.iter().enumerate() {
            let sub = wal_root.join(format!("macro_{m}"));
            block.hip.attach_wal_dir(&sub);
        }
    }

    pub fn flush_hippocampus(&self) -> u64 {
        self.mem.iter().map(|m| m.hip.flush_barrier()).sum()
    }

    pub fn embed(&self, token: u32) -> Vec<f32> {
        let vocab = self.cfg.vocab.max(1);
        let t = (token as usize).min(vocab - 1);
        let d = self.cfg.d_model;
        let scale = self.emb_scale.as_slice()[t];
        let row = &self.emb_i8[t * d..(t + 1) * d];
        row.iter().map(|q| f32::from(*q) * scale).collect()
    }

    pub fn logits_from_hidden(&self, h: &[f32]) -> Vec<f32> {
        let hn = self.final_norm.forward(h);
        let d = self.cfg.d_model;
        let scales = self.emb_scale.as_slice();
        (0..self.cfg.vocab)
            .map(|v| {
                let row = &self.emb_i8[v * d..(v + 1) * d];
                let mut dot = 0f32;
                for i in 0..d {
                    dot += f32::from(row[i]) * hn[i];
                }
                scales[v] * dot
            })
            .collect()
    }

    /// Read-only deterministic token step (used by `logits` check and MeZO loss).
    pub fn step(&self, token: u32, state: &mut ModelState) -> Vec<f32> {
        self.step_with_plasticity(token, false, false, &[], state)
    }

    /// Continual-learning token step: reads old hippocampus on Fast-Path and
    /// dispatches `(k_t, v_t)` asynchronously to background plasticity workers.
    pub fn step_plastic(
        &self,
        token: u32,
        is_eos: bool,
        context: &[u32],
        state: &mut ModelState,
    ) -> Vec<f32> {
        self.step_with_plasticity(token, true, is_eos, context, state)
    }

    fn step_with_plasticity(
        &self,
        token: u32,
        learn: bool,
        is_eos: bool,
        context: &[u32],
        state: &mut ModelState,
    ) -> Vec<f32> {
        let mut x = self.embed(token);
        for m in 0..self.cfg.n_macros {
            for s in 0..3 {
                let idx = m * 3 + s;
                x = self.ssm[idx].forward(&x, &mut state.ssm[idx]);
            }
            x = self.mem[m].forward_step(&x, &mut state.rings[m], learn, is_eos, context);
        }
        self.logits_from_hidden(&x)
    }

    pub fn forward_hidden(&self, token: u32, state: &mut ModelState) -> Vec<f32> {
        let mut x = self.embed(token);
        for m in 0..self.cfg.n_macros {
            for s in 0..3 {
                let idx = m * 3 + s;
                x = self.ssm[idx].forward(&x, &mut state.ssm[idx]);
            }
            x = self.mem[m].forward(&x, &mut state.rings[m]);
        }
        x
    }

    pub fn mean_ce(&self, seqs: &[&[u32]]) -> f32 {
        let mut total = 0f32;
        let mut n = 0f32;
        for seq in seqs {
            if seq.len() < 2 {
                continue;
            }
            let mut state = self.new_state();
            for t in 0..seq.len() - 1 {
                let logits = self.step(seq[t], &mut state);
                total += cross_entropy(&logits, seq[t + 1] as usize);
                n += 1.0;
            }
        }
        if n == 0.0 {
            0.0
        } else {
            total / n
        }
    }

    pub fn phi_len(&self) -> usize {
        let mut n = self.emb_scale.len();
        for block in &self.ssm {
            n += block.q_proj.row_scales.len();
            n += block.k_proj.row_scales.len();
            n += block.v_proj.row_scales.len();
            n += block.o_proj.row_scales.len();
            n += block.ffn.w_gate.row_scales.len();
            n += block.ffn.w_up.row_scales.len();
            n += block.ffn.w_down.row_scales.len();
            n += block.norm1.weight.len();
            n += block.norm2.weight.len();
        }
        for block in &self.mem {
            n += block.attn.w_q.row_scales.len();
            n += block.attn.w_k.row_scales.len();
            n += block.attn.w_v.row_scales.len();
            n += block.attn.w_o.row_scales.len();
            n += block.hip_proj.row_scales.len();
            n += block.hip_k.row_scales.len();
            n += block.hip_v.row_scales.len();
            n += block.ffn.w_gate.row_scales.len();
            n += block.ffn.w_up.row_scales.len();
            n += block.ffn.w_down.row_scales.len();
            n += block.norm1.weight.len();
            n += block.norm2.weight.len();
        }
        n += self.final_norm.weight.len();
        n
    }

    /// Add `scale * z_i` in place to every continuous `φ` coordinate.
    /// `z` is regenerated on the fly from `seed` — zero buffer allocations.
    pub fn apply_z(&mut self, seed: u64, scale: f32) {
        let mut i = 0usize;
        self.walk_phi_mut(&mut |w| {
            *w += scale * rademacher(seed, i);
            i += 1;
        });
    }

    pub fn walk_phi_mut(&mut self, f: &mut dyn FnMut(&mut f32)) {
        for s in self.emb_scale.as_mut_slice() {
            f(s);
        }
        for block in &mut self.ssm {
            touch_scales(&mut block.q_proj, f);
            touch_scales(&mut block.k_proj, f);
            touch_scales(&mut block.v_proj, f);
            touch_scales(&mut block.o_proj, f);
            touch_scales(&mut block.ffn.w_gate, f);
            touch_scales(&mut block.ffn.w_up, f);
            touch_scales(&mut block.ffn.w_down, f);
            for s in block.norm1.weight.as_mut_slice() {
                f(s);
            }
            for s in block.norm2.weight.as_mut_slice() {
                f(s);
            }
        }
        for block in &mut self.mem {
            touch_scales(&mut block.attn.w_q, f);
            touch_scales(&mut block.attn.w_k, f);
            touch_scales(&mut block.attn.w_v, f);
            touch_scales(&mut block.attn.w_o, f);
            touch_scales(&mut block.hip_proj, f);
            touch_scales(&mut block.hip_k, f);
            touch_scales(&mut block.hip_v, f);
            touch_scales(&mut block.ffn.w_gate, f);
            touch_scales(&mut block.ffn.w_up, f);
            touch_scales(&mut block.ffn.w_down, f);
            for s in block.norm1.weight.as_mut_slice() {
                f(s);
            }
            for s in block.norm2.weight.as_mut_slice() {
                f(s);
            }
        }
        for s in self.final_norm.weight.as_mut_slice() {
            f(s);
        }
    }

    /// Zero-copy in-place MeZO tuning over `seqs` (strictly 2 forwards/step).
    pub fn mezo_steps(
        &mut self,
        seqs: &[Vec<u32>],
        steps: usize,
        eps: f32,
        lr: f32,
        seed0: u64,
    ) -> (f32, f32) {
        let refs: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let before = self.mean_ce(&refs);
        let mut tuner = MezoScaleTuner::new(eps, lr, seed0).with_max_step(eps.abs().max(1e-6));
        for _ in 0..steps {
            let mut driver = MezoDriver { model: self, seqs };
            tuner.step_in_place(&mut driver);
        }
        let refs: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let after = self.mean_ce(&refs);
        (before, after)
    }

    pub fn hip_version_sum(&self) -> u64 {
        self.mem.iter().map(|m| m.hip.version()).sum()
    }

    /// Consolidate ALL macro hippocampus modules (Gram–Schmidt row dedup +
    /// singular suppression) and optionally persist a versioned multi-macro
    /// `HPSP` snapshot. Returns `(deduped_w1_sum, deduped_w2_sum, snapshot_ver)`.
    pub fn consolidate_all_hippocampus(
        &self,
        store: Option<&SnapshotStore>,
    ) -> std::io::Result<(usize, usize, Option<u64>)> {
        let mut total_d1 = 0usize;
        let mut total_d2 = 0usize;
        let mut snapshots = Vec::with_capacity(self.mem.len());
        for m in &self.mem {
            m.hip.flush_barrier();
            let mut w = m.hip.snapshot();
            let (d1, d2) = consolidate_hippocampus(&mut w, 0.95);
            total_d1 += d1;
            total_d2 += d2;
            w.version = w.version.wrapping_add(1);
            m.hip.replace_weights(w.clone());
            snapshots.push(w);
        }
        let saved = match store {
            Some(s) => Some(s.save_macros(&snapshots)?),
            None => None,
        };
        Ok((total_d1, total_d2, saved))
    }

    /// Roll ALL macro hippocampus modules back to `version` from `store`.
    pub fn rollback_hippocampus(&self, store: &SnapshotStore, version: u64) -> std::io::Result<()> {
        let weights = store.rollback_macros(version)?;
        if weights.len() != self.mem.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "snapshot macro count {} != model macro count {}",
                    weights.len(),
                    self.mem.len()
                ),
            ));
        }
        for (m, w) in self.mem.iter().zip(weights.into_iter()) {
            m.hip.replace_weights(w);
        }
        Ok(())
    }

    pub fn reset_memory_to_init(&self) {
        for (i, m) in self.mem.iter().enumerate() {
            m.hip.replace_weights(HippocampusWeights::new_seeded(
                self.cfg.d_model,
                self.cfg.hip_hidden,
                self.cfg.d_model,
                (500 + i as u32 * 19).wrapping_add(1),
            ));
        }
    }
}

fn touch_scales(w: &mut BitLinearWeights, f: &mut dyn FnMut(&mut f32)) {
    for s in w.row_scales.as_mut_slice() {
        f(s);
    }
}

struct MezoDriver<'a> {
    model: &'a mut CortexModel,
    seqs: &'a [Vec<u32>],
}

impl InPlacePhi for MezoDriver<'_> {
    fn apply_z(&mut self, seed: u64, scale: f32) {
        self.model.apply_z(seed, scale);
    }
    fn loss(&mut self) -> f32 {
        let refs: Vec<&[u32]> = self.seqs.iter().map(|s| s.as_slice()).collect();
        self.model.mean_ce(&refs)
    }
}

fn det(i: usize, salt: u32) -> f32 {
    let mut x = (i as u32).wrapping_mul(0x9E37_79B1) ^ salt;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    (x as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step1_stories_and_prod_shapes_are_valid() {
        let c = CortexConfig::step1_stories();
        c.validate().unwrap();
        assert_eq!(c.n_layers(), 8);
        let p = CortexConfig::prod();
        p.validate().unwrap();
        assert_eq!(p.n_layers(), 24);
        assert_eq!(p.window, 512);
    }

    #[test]
    fn test_256_plastic_step_advances_hippocampus_in_background() {
        let model = CortexModel::from_seed(CortexConfig::test_256(), 1);
        assert_eq!(model.n_layers(), 8);
        let mut state = model.new_state();
        let bytes0 = state.state_bytes();
        for t in 0..8u32 {
            let logits = model.step_plastic(t + 1, t == 7, &[1, 2, 3], &mut state);
            assert_eq!(logits.len(), 266);
            assert!(logits.iter().all(|v| v.is_finite()));
        }
        let ver = model.flush_hippocampus();
        assert!(
            ver >= 2,
            "both macros' online hippocampus must advance: ver_sum={ver}"
        );
        assert_eq!(state.state_bytes(), bytes0);
    }

    #[test]
    fn test_mezo_loss_descent_on_nonzero_8_layer_model() {
        // Real 8-layer model with non-zero ternary weights across all blocks!
        let mut model = CortexModel::from_seed(CortexConfig::mezo_tiny(), 7);
        assert_eq!(model.n_layers(), 8);
        let seqs = vec![
            vec![1u32, 2, 3, 4, 1, 2],
            vec![2, 3, 4, 1, 2, 3],
            vec![3, 4, 1, 2, 3, 4],
            vec![4, 1, 2, 3, 4, 1],
        ];
        let (l0, l1) = model.mezo_steps(&seqs, 60, 1e-4, 3e-6, 0xC0FFEE);
        assert!(l0 > 1.0, "expected non-degenerate initial CE, got {l0}");
        assert!(
            l1 < l0,
            "zero-copy in-place MeZO did not descend on non-zero 8-layer model: {l0:.5} -> {l1:.5}"
        );
        assert!(model.phi_len() > 2000);
    }
}
