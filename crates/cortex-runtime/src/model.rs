//! Macro-cycle Cortex model.
//!
//! Production shape ([`CortexConfig::prod`]): `N_MACROS = 6`, 24 layers,
//! `d_model = 2048`, ring window 512. Tests may use
//! [`CortexConfig::test_256`] (2 macros, 8 layers, `d_model = 256`) or the
//! smaller [`CortexConfig::mezo_tiny`] probe.
//!
//! Each macro:
//! * layers 1–3 — [`BitSsmBlock`] (1.58-bit QKV, Gated DeltaNet, 1.58-bit SwiGLU)
//! * layer 4 — [`MemoryBlock`]: INT8 ring attention (branch A) + hippocampus
//!   `W2 SiLU(W1 q)` (branch B), gated residual, then 1.58-bit SwiGLU.
//!
//! The token embedding and the LM head are the same INT8 table plus per-row
//! FP32 scales (weight tying). MeZO's φ is exactly those row scales (embedding
//! and every BitLinear) and the RMSNorm weights.

use cortex_hippocampus::{HippocampusWeights, SharedWeightArena};
use cortex_kernels::{sigmoid, Aligned64, BitLinearWeights};
use cortex_sleep::{mezo_3pass, rademacher, InPlacePhi};
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
    pub fn prod() -> Self {
        Self {
            n_macros: 6,
            d_model: 2048,
            n_heads: 32,
            head_dim: 64,
            vocab: 256,
            window: 512,
            ffn_hidden: 8192,
            hip_hidden: 512,
            rms_eps: 1e-6,
        }
    }

    /// 2 macros × 4 layers = 8, d_model 256, window 512.
    pub fn test_256() -> Self {
        Self {
            n_macros: 2,
            d_model: 256,
            n_heads: 8,
            head_dim: 32,
            vocab: 64,
            window: 512,
            ffn_hidden: 512,
            hip_hidden: 64,
            rms_eps: 1e-6,
        }
    }

    /// Tiny 8-layer net used by MeZO descent and fast CLI fixtures.
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

    /// Step 1 sandbox: 2 macros, 8 layers, d_model 512, vocab 4096.
    /// Weight-matrix count is about 25.09M before the small row-scale vectors.
    /// This constructor does not allocate those weights.
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
    pub hip: SharedWeightArena,
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
            hip: SharedWeightArena::new(HippocampusWeights::new(
                cfg.d_model,
                cfg.hip_hidden,
                cfg.d_model,
            )),
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
            hip_proj: BitLinearWeights::from_seed(cfg.d_model, cfg.d_model, seed.wrapping_add(20)),
            hip: SharedWeightArena::new(HippocampusWeights::new(
                cfg.d_model,
                cfg.hip_hidden,
                cfg.d_model,
            )),
            gate_w,
            gate_b,
            ffn: BitSwiGLU::from_seed(cfg.d_model, cfg.ffn_hidden, seed.wrapping_add(30)),
            d_model: cfg.d_model,
        }
    }

    pub fn forward(&self, x: &[f32], ring: &mut RingState) -> Vec<f32> {
        let xn = self.norm1.forward(x);
        let a = self.attn.forward(&xn, ring);
        let q = self.hip_proj.forward(&xn);
        let m = self.hip.read(|w| w.forward(&q));
        let gw = self.gate_w.as_slice();
        let gb = self.gate_b.as_slice();
        let mut y = vec![0f32; self.d_model];
        for i in 0..self.d_model {
            let g = sigmoid(gw[i] * xn[i] + gb[i]);
            y[i] = x[i] + g * a[i] + (1.0 - g) * m[i];
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

pub struct CortexModel {
    pub cfg: CortexConfig,
    pub emb_i8: Vec<i8>,
    pub emb_scale: Aligned64<f32>,
    pub ssm: Vec<BitSsmBlock>,
    pub mem: Vec<MemoryBlock>,
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
        Self {
            emb_i8: vec![0i8; cfg.vocab * cfg.d_model],
            emb_scale,
            ssm,
            mem,
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
            // Map into [-127, 127]. Never -128.
            *slot = ((x % 255) as i32 - 127) as i8;
        }
        model.emb_scale.fill(0.02);
        model.ssm = (0..model.cfg.n_macros * 3)
            .map(|i| {
                BitSsmBlock::from_seed(
                    model.cfg.d_model,
                    model.cfg.n_heads,
                    model.cfg.head_dim,
                    model.cfg.ffn_hidden,
                    model.cfg.rms_eps,
                    seed.wrapping_add(100 + i as u32 * 17),
                )
            })
            .collect();
        model.mem = (0..model.cfg.n_macros)
            .map(|i| MemoryBlock::from_seed(&model.cfg, seed.wrapping_add(500 + i as u32 * 19)))
            .collect();
        model
    }

    /// Zero trunk (ternary weights are 0) and a one-hot INT8 embedding.
    /// Row scales start at 0.4 so repeated-token CE is a smooth function of φ
    /// and MeZO has a real descent direction. The 8 layer bodies still run.
    pub fn mezo_probe() -> Self {
        let mut model = Self::zeros(CortexConfig::mezo_tiny());
        let d = model.cfg.d_model;
        for t in 0..model.cfg.vocab {
            if t < d {
                model.emb_i8[t * d + t] = 1;
            }
        }
        model.emb_scale.fill(0.4);
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

    pub fn embed(&self, token: u32) -> Vec<f32> {
        let vocab = self.cfg.vocab.max(1);
        let t = (token as usize) % vocab;
        let d = self.cfg.d_model;
        let scale = self.emb_scale.as_slice()[t];
        let row = &self.emb_i8[t * d..(t + 1) * d];
        row.iter().map(|q| f32::from(*q) * scale).collect()
    }

    pub fn logits_from_hidden(&self, h: &[f32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let scales = self.emb_scale.as_slice();
        (0..self.cfg.vocab)
            .map(|v| {
                let row = &self.emb_i8[v * d..(v + 1) * d];
                let mut dot = 0f32;
                for i in 0..d {
                    dot += f32::from(row[i]) * h[i];
                }
                scales[v] * dot
            })
            .collect()
    }

    /// One token. Returns LM-head logits. State is per session, not global.
    pub fn step(&self, token: u32, state: &mut ModelState) -> Vec<f32> {
        let mut x = self.embed(token);
        for m in 0..self.cfg.n_macros {
            for s in 0..3 {
                let idx = m * 3 + s;
                x = self.ssm[idx].forward(&x, &mut state.ssm[idx]);
            }
            x = self.mem[m].forward(&x, &mut state.rings[m]);
        }
        self.logits_from_hidden(&x)
    }

    /// Hidden state after the trunk, before the LM head. Useful for tests.
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
            n += block.ffn.w_gate.row_scales.len();
            n += block.ffn.w_up.row_scales.len();
            n += block.ffn.w_down.row_scales.len();
            n += block.norm1.weight.len();
            n += block.norm2.weight.len();
        }
        n
    }

    /// Add `scale * z_i` to every φ coordinate. `z` is regenerated from `seed`.
    pub fn apply_z(&mut self, seed: u64, scale: f32) {
        let mut i = 0usize;
        self.walk_phi_mut(&mut |w| {
            *w += scale * rademacher(seed, i);
            i += 1;
        });
    }

    fn walk_phi_mut(&mut self, f: &mut dyn FnMut(&mut f32)) {
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
    }

    /// 30-step helper used by the descent test and `cortex-cli sleep`.
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
        let mut seed = seed0;
        for _ in 0..steps {
            let mut driver = MezoDriver { model: self, seqs };
            mezo_3pass(&mut driver, seed, eps, lr);
            seed = seed.wrapping_add(1);
        }
        let refs: Vec<&[u32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let after = self.mean_ce(&refs);
        (before, after)
    }

    pub fn hip_version_sum(&self) -> u64 {
        self.mem.iter().map(|m| m.hip.read(|w| w.version)).sum()
    }

    pub fn reset_memory_to_init(&self) {
        for m in &self.mem {
            m.hip.replace(HippocampusWeights::new(
                self.cfg.d_model,
                self.cfg.hip_hidden,
                self.cfg.d_model,
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
    fn step1_stories_shape_is_valid_and_unallocated() {
        let c = CortexConfig::step1_stories();
        c.validate().unwrap();
        assert_eq!(c.n_macros, 2);
        assert_eq!(c.n_layers(), 8);
        assert_eq!(c.d_model, 512);
        assert_eq!(c.vocab, 4096);
        assert_eq!(c.window, 64);
        assert_eq!(c.n_heads * c.head_dim, c.d_model);
    }

    #[test]
    fn prod_and_test_shapes() {
        let p = CortexConfig::prod();
        assert_eq!(p.n_macros, 6);
        assert_eq!(p.n_layers(), 24);
        assert_eq!(p.d_model, 2048);
        assert_eq!(p.window, 512);
        assert_eq!(p.n_heads * p.head_dim, p.d_model);
        let t = CortexConfig::test_256();
        assert_eq!(t.n_macros, 2);
        assert_eq!(t.n_layers(), 8);
        assert_eq!(t.d_model, 256);
        assert_eq!(t.window, 512);
    }

    #[test]
    fn test_256_one_token_is_finite() {
        let model = CortexModel::from_seed(CortexConfig::test_256(), 1);
        assert_eq!(model.n_layers(), 8);
        let mut state = model.new_state();
        let bytes = state.ssm.iter().map(|s| s.state_bytes()).sum::<usize>();
        let logits = model.step(3, &mut state);
        assert_eq!(logits.len(), 64);
        assert!(logits.iter().all(|v| v.is_finite()));
        let bytes2 = state.ssm.iter().map(|s| s.state_bytes()).sum::<usize>();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn test_mezo_loss_descent() {
        let mut model = CortexModel::mezo_probe();
        assert_eq!(model.n_layers(), 8);
        // Repeated tokens: with a zero ternary trunk the residual equals the
        // embedding, so CE is a smooth function of the tied row scales.
        // The eight blocks still execute on every step (no autograd).
        let seqs = vec![
            vec![1u32, 1, 1, 1],
            vec![2, 2, 2, 2],
            vec![3, 3, 3, 3],
            vec![4, 4, 4, 4],
        ];
        let (l0, l1) = model.mezo_steps(&seqs, 30, 0.05, 0.35, 0xC0FFEE);
        assert!(
            l1 < l0 - 0.05,
            "MeZO CE did not descend significantly: {l0} -> {l1}"
        );
        // φ was updated in place; there is no graph object to leak.
        assert!(model.phi_len() > 0);
    }
}
