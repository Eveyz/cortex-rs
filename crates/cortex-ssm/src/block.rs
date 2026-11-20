//! BitSSM block: RMSNorm → 1.58-bit QKV → Gated DeltaNet → out-proj → SwiGLU.
//!
//! This is one of layers 1–3 inside a macro cycle. Layer 4 is the memory
//! block and lives in `cortex-runtime`.

use cortex_kernels::{sigmoid, Aligned64, BitLinearWeights};

use crate::deltanet::{deltanet_step, DeltaState};
use crate::swiglu::BitSwiGLU;

#[derive(Clone, Debug)]
pub struct RmsNorm {
    pub weight: Aligned64<f32>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn ones(dim: usize, eps: f32) -> Self {
        let mut weight = Aligned64::zeros(dim);
        weight.fill(1.0);
        Self { weight, eps }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len().max(1) as f32;
        let mut ss = 0f32;
        for v in x {
            ss += v * v;
        }
        let inv = (ss / n + self.eps).sqrt().recip();
        let w = self.weight.as_slice();
        x.iter().zip(w.iter()).map(|(v, g)| v * inv * g).collect()
    }
}

pub struct BitSsmBlock {
    pub norm1: RmsNorm,
    pub q_proj: BitLinearWeights,
    pub k_proj: BitLinearWeights,
    pub v_proj: BitLinearWeights,
    pub o_proj: BitLinearWeights,
    /// `[n_heads, d_model]` row-major.
    pub alpha_w: Aligned64<f32>,
    pub alpha_b: Aligned64<f32>,
    pub beta_w: Aligned64<f32>,
    pub beta_b: Aligned64<f32>,
    pub norm2: RmsNorm,
    pub ffn: BitSwiGLU,
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
}

impl BitSsmBlock {
    pub fn zeros(d_model: usize, n_heads: usize, head_dim: usize, ffn: usize, eps: f32) -> Self {
        assert_eq!(n_heads * head_dim, d_model);
        Self {
            norm1: RmsNorm::ones(d_model, eps),
            q_proj: BitLinearWeights::zeros(d_model, d_model),
            k_proj: BitLinearWeights::zeros(d_model, d_model),
            v_proj: BitLinearWeights::zeros(d_model, d_model),
            o_proj: BitLinearWeights::zeros(d_model, d_model),
            alpha_w: Aligned64::zeros(n_heads * d_model),
            alpha_b: Aligned64::zeros(n_heads),
            beta_w: Aligned64::zeros(n_heads * d_model),
            beta_b: Aligned64::zeros(n_heads),
            norm2: RmsNorm::ones(d_model, eps),
            ffn: BitSwiGLU::zeros(d_model, ffn),
            n_heads,
            head_dim,
            d_model,
        }
    }

    pub fn from_seed(
        d_model: usize,
        n_heads: usize,
        head_dim: usize,
        ffn: usize,
        eps: f32,
        seed: u32,
    ) -> Self {
        assert_eq!(n_heads * head_dim, d_model);
        let mut alpha_w = Aligned64::zeros(n_heads * d_model);
        let mut beta_w = Aligned64::zeros(n_heads * d_model);
        let mut alpha_b = Aligned64::zeros(n_heads);
        let mut beta_b = Aligned64::zeros(n_heads);
        for (i, slot) in alpha_w.as_mut_slice().iter_mut().enumerate() {
            *slot = det(i, seed) * 0.02;
        }
        for (i, slot) in beta_w.as_mut_slice().iter_mut().enumerate() {
            *slot = det(i, seed.wrapping_add(9)) * 0.02;
        }
        // Biases keep α near 0.5 and β near 0.25 at zero input.
        alpha_b.fill(0.0);
        beta_b.fill(-1.1);
        Self {
            norm1: RmsNorm::ones(d_model, eps),
            q_proj: BitLinearWeights::from_seed(d_model, d_model, seed),
            k_proj: BitLinearWeights::from_seed(d_model, d_model, seed.wrapping_add(11)),
            v_proj: BitLinearWeights::from_seed(d_model, d_model, seed.wrapping_add(22)),
            o_proj: BitLinearWeights::from_seed(d_model, d_model, seed.wrapping_add(33)),
            alpha_w,
            alpha_b,
            beta_w,
            beta_b,
            norm2: RmsNorm::ones(d_model, eps),
            ffn: BitSwiGLU::from_seed(d_model, ffn, seed.wrapping_add(40)),
            n_heads,
            head_dim,
            d_model,
        }
    }

    pub fn new_state(&self) -> DeltaState {
        DeltaState::zeros(self.n_heads, self.head_dim)
    }

    pub fn forward(&self, x: &[f32], state: &mut DeltaState) -> Vec<f32> {
        assert_eq!(x.len(), self.d_model);
        let xn = self.norm1.forward(x);
        let q = self.q_proj.forward(&xn);
        let k = self.k_proj.forward(&xn);
        let v = self.v_proj.forward(&xn);
        let (alpha, beta) = self.gates(&xn);
        let o = deltanet_step(state, &q, &k, &v, &alpha, &beta);
        let o = self.o_proj.forward(&o);
        let mut x1 = vec![0f32; self.d_model];
        for i in 0..self.d_model {
            x1[i] = x[i] + o[i];
        }
        let xn2 = self.norm2.forward(&x1);
        let ffn = self.ffn.forward(&xn2);
        for i in 0..self.d_model {
            x1[i] += ffn[i];
        }
        x1
    }

    fn gates(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let d = self.d_model;
        let aw = self.alpha_w.as_slice();
        let bw = self.beta_w.as_slice();
        let ab = self.alpha_b.as_slice();
        let bb = self.beta_b.as_slice();
        let mut alpha = vec![0f32; self.n_heads];
        let mut beta = vec![0f32; self.n_heads];
        for h in 0..self.n_heads {
            let mut da = ab[h];
            let mut db = bb[h];
            let row = h * d;
            for i in 0..d {
                da += aw[row + i] * x[i];
                db += bw[row + i] * x[i];
            }
            alpha[h] = sigmoid(da);
            beta[h] = sigmoid(db);
        }
        (alpha, beta)
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
    fn three_layers_are_deterministic_and_finite() {
        let d = 32usize;
        let heads = 4usize;
        let hd = 8usize;
        let blocks: Vec<_> = (0..3)
            .map(|i| BitSsmBlock::from_seed(d, heads, hd, 32, 1e-6, 100 + i))
            .collect();
        let mut states: Vec<_> = blocks.iter().map(|b| b.new_state()).collect();
        let bytes: Vec<_> = states.iter().map(|s| s.state_bytes()).collect();
        let mut x = vec![0.05f32; d];
        x[0] = 0.2;
        let mut first = None;
        for _ in 0..4 {
            let mut h = x.clone();
            for (b, st) in blocks.iter().zip(states.iter_mut()) {
                h = b.forward(&h, st);
            }
            assert!(h.iter().all(|v| v.is_finite()));
            if first.is_none() {
                first = Some(h);
            }
        }
        // Replay from a fresh state: same first-token result.
        let mut states2: Vec<_> = blocks.iter().map(|b| b.new_state()).collect();
        let mut h = x.clone();
        for (b, st) in blocks.iter().zip(states2.iter_mut()) {
            h = b.forward(&h, st);
        }
        for (a, b) in first.unwrap().iter().zip(h.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        for (s, b0) in states.iter().zip(bytes.iter()) {
            assert_eq!(s.state_bytes(), *b0);
        }
    }
}
