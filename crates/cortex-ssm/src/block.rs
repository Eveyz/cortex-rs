//! BitSSM block: RMSNorm → 1.58-bit QKV → Gated DeltaNet → out-proj → SwiGLU.
//!
//! Layers 1–3 inside each macro cycle. Supports both single-token O(1)-memory
//! streaming (`forward`) and WY chunkwise prefill (`forward_chunk`).

use cortex_kernels::{sigmoid, Aligned64, BitLinearWeights};

use crate::deltanet::{deltanet_chunk, deltanet_step, DeltaState};
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
        let mut out = vec![0f32; x.len()];
        self.forward_into(x, &mut out);
        out
    }

    pub fn forward_into(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.weight.len());
        assert_eq!(out.len(), x.len());
        let n = x.len().max(1) as f32;
        let mut ss = 0f32;
        for v in x {
            ss += v * v;
        }
        let inv = (ss / n + self.eps).sqrt().recip();
        let w = self.weight.as_slice();
        for i in 0..x.len() {
            out[i] = x[i] * inv * w[i];
        }
    }
}

pub struct BitSsmBlock {
    pub norm1: RmsNorm,
    pub q_proj: BitLinearWeights,
    pub k_proj: BitLinearWeights,
    pub v_proj: BitLinearWeights,
    pub o_proj: BitLinearWeights,
    /// `[n_heads, d_model]` row-major gate projections.
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
        let mut alpha_b = Aligned64::zeros(n_heads);
        alpha_b.fill(4.6); // sigmoid(4.6) ≈ 0.99 long-horizon default
        let mut beta_b = Aligned64::zeros(n_heads);
        beta_b.fill(-1.1);
        Self {
            norm1: RmsNorm::ones(d_model, eps),
            q_proj: BitLinearWeights::zeros(d_model, d_model),
            k_proj: BitLinearWeights::zeros(d_model, d_model),
            v_proj: BitLinearWeights::zeros(d_model, d_model),
            o_proj: BitLinearWeights::zeros(d_model, d_model),
            alpha_w: Aligned64::zeros(n_heads * d_model),
            alpha_b,
            beta_w: Aligned64::zeros(n_heads * d_model),
            beta_b,
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
        // Keep α ≈ 0.99 (long-horizon retention) and β ≈ 0.25 at init.
        alpha_b.fill(4.6);
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

    /// Chunkwise prefill over `xs: [L * d_model]`, chunked by `chunk_size`.
    pub fn forward_chunk(&self, xs: &[f32], state: &mut DeltaState, chunk_size: usize) -> Vec<f32> {
        let d = self.d_model;
        let l = xs.len() / d;
        assert_eq!(xs.len(), l * d);
        let chunk = chunk_size.max(1);
        let mut outs = vec![0f32; l * d];
        let mut start = 0;
        while start < l {
            let end = (start + chunk).min(l);
            let clen = end - start;
            let mut qs = vec![0f32; clen * d];
            let mut ks = vec![0f32; clen * d];
            let mut vs = vec![0f32; clen * d];
            let mut alphas = vec![0f32; clen * self.n_heads];
            let mut betas = vec![0f32; clen * self.n_heads];
            for t in 0..clen {
                let x = &xs[(start + t) * d..(start + t + 1) * d];
                let xn = self.norm1.forward(x);
                self.q_proj.forward_into(&xn, &mut qs[t * d..(t + 1) * d]);
                self.k_proj.forward_into(&xn, &mut ks[t * d..(t + 1) * d]);
                self.v_proj.forward_into(&xn, &mut vs[t * d..(t + 1) * d]);
                let (a, b) = self.gates(&xn);
                alphas[t * self.n_heads..(t + 1) * self.n_heads].copy_from_slice(&a);
                betas[t * self.n_heads..(t + 1) * self.n_heads].copy_from_slice(&b);
            }
            let mixed = deltanet_chunk(state, &qs, &ks, &vs, &alphas, &betas, clen);
            for t in 0..clen {
                let x = &xs[(start + t) * d..(start + t + 1) * d];
                let o = self.o_proj.forward(&mixed[t * d..(t + 1) * d]);
                let mut x1 = vec![0f32; d];
                for i in 0..d {
                    x1[i] = x[i] + o[i];
                }
                let xn2 = self.norm2.forward(&x1);
                let ffn = self.ffn.forward(&xn2);
                for i in 0..d {
                    outs[(start + t) * d + i] = x1[i] + ffn[i];
                }
            }
            start = end;
        }
        outs
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
    fn three_layers_are_deterministic_and_chunk_matches_step() {
        let d = 32usize;
        let heads = 4usize;
        let hd = 8usize;
        let block = BitSsmBlock::from_seed(d, heads, hd, 32, 1e-6, 100);
        let l = 11usize;
        let xs: Vec<f32> = (0..l * d)
            .map(|i| ((i as f32) * 0.17).sin() * 0.3)
            .collect();

        let mut st_step = block.new_state();
        let bytes0 = st_step.state_bytes();
        let mut out_step = Vec::with_capacity(l * d);
        for t in 0..l {
            out_step.extend_from_slice(&block.forward(&xs[t * d..(t + 1) * d], &mut st_step));
        }

        let mut st_chunk = block.new_state();
        let out_chunk = block.forward_chunk(&xs, &mut st_chunk, 4);

        let err = out_step
            .iter()
            .zip(out_chunk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-3, "BitSsmBlock chunk vs step err {err}");
        assert_eq!(st_step.state_bytes(), bytes0);
    }
}
