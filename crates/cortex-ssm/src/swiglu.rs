//! 1.58-bit SwiGLU: `down(SiLU(gate(x)) * up(x))`.
//! All three projections are [`BitLinearWeights`].

use cortex_kernels::{silu, BitLinearWeights};

#[derive(Clone, Debug)]
pub struct BitSwiGLU {
    pub w_gate: BitLinearWeights,
    pub w_up: BitLinearWeights,
    pub w_down: BitLinearWeights,
}

impl BitSwiGLU {
    pub fn zeros(d_model: usize, hidden: usize) -> Self {
        Self {
            w_gate: BitLinearWeights::zeros(hidden, d_model),
            w_up: BitLinearWeights::zeros(hidden, d_model),
            w_down: BitLinearWeights::zeros(d_model, hidden),
        }
    }

    pub fn from_seed(d_model: usize, hidden: usize, seed: u32) -> Self {
        Self {
            w_gate: BitLinearWeights::from_seed(hidden, d_model, seed),
            w_up: BitLinearWeights::from_seed(hidden, d_model, seed.wrapping_add(1)),
            w_down: BitLinearWeights::from_seed(d_model, hidden, seed.wrapping_add(2)),
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let gate = self.w_gate.forward(x);
        let up = self.w_up.forward(x);
        let mut h = vec![0f32; gate.len()];
        for i in 0..gate.len() {
            h[i] = silu(gate[i]) * up[i];
        }
        self.w_down.forward(&h)
    }
}
