//! INT8 ring sliding-window attention.
//!
//! Keys and values are quantized to symmetric INT8 when they enter the ring.
//! The ring is allocated once at `window` slots (production default 512) and
//! never grows with the session length. Scores are the dequantized dot
//! product, scaled by `1/sqrt(head_dim)`, with a causal mask over the slots
//! that have actually been written.

use cortex_kernels::{quantize_activations_sym_i8, BitLinearWeights};

pub struct RingState {
    k: Vec<i8>,
    v: Vec<i8>,
    k_scale: Vec<f32>,
    v_scale: Vec<f32>,
    /// Next write index in `0..window`.
    pos: usize,
    filled: usize,
    window: usize,
    n_heads: usize,
    head_dim: usize,
}

impl RingState {
    pub fn new(window: usize, n_heads: usize, head_dim: usize) -> Self {
        let slots = window.max(1);
        let width = n_heads * head_dim;
        Self {
            k: vec![0i8; slots * width],
            v: vec![0i8; slots * width],
            k_scale: vec![1.0; slots * n_heads],
            v_scale: vec![1.0; slots * n_heads],
            pos: 0,
            filled: 0,
            window: slots,
            n_heads,
            head_dim,
        }
    }

    pub fn state_bytes(&self) -> usize {
        self.k.len() + self.v.len() + (self.k_scale.len() + self.v_scale.len()) * 4
    }

    pub fn reset(&mut self) {
        self.k.fill(0);
        self.v.fill(0);
        self.k_scale.fill(1.0);
        self.v_scale.fill(1.0);
        self.pos = 0;
        self.filled = 0;
    }

    pub fn filled(&self) -> usize {
        self.filled
    }
}

pub struct RingSlidingAttention {
    pub w_q: BitLinearWeights,
    pub w_k: BitLinearWeights,
    pub w_v: BitLinearWeights,
    pub w_o: BitLinearWeights,
    pub n_heads: usize,
    pub head_dim: usize,
    pub window: usize,
    pub d_model: usize,
}

impl RingSlidingAttention {
    pub fn zeros(d_model: usize, n_heads: usize, head_dim: usize, window: usize) -> Self {
        assert_eq!(n_heads * head_dim, d_model);
        Self {
            w_q: BitLinearWeights::zeros(d_model, d_model),
            w_k: BitLinearWeights::zeros(d_model, d_model),
            w_v: BitLinearWeights::zeros(d_model, d_model),
            w_o: BitLinearWeights::zeros(d_model, d_model),
            n_heads,
            head_dim,
            window: window.max(1),
            d_model,
        }
    }

    pub fn from_seed(
        d_model: usize,
        n_heads: usize,
        head_dim: usize,
        window: usize,
        seed: u32,
    ) -> Self {
        assert_eq!(n_heads * head_dim, d_model);
        Self {
            w_q: BitLinearWeights::from_seed(d_model, d_model, seed),
            w_k: BitLinearWeights::from_seed(d_model, d_model, seed.wrapping_add(3)),
            w_v: BitLinearWeights::from_seed(d_model, d_model, seed.wrapping_add(5)),
            w_o: BitLinearWeights::from_seed(d_model, d_model, seed.wrapping_add(7)),
            n_heads,
            head_dim,
            window: window.max(1),
            d_model,
        }
    }

    pub fn new_state(&self) -> RingState {
        RingState::new(self.window, self.n_heads, self.head_dim)
    }

    /// Push the current token and attend over the retained window.
    pub fn forward(&self, x: &[f32], state: &mut RingState) -> Vec<f32> {
        assert_eq!(x.len(), self.d_model);
        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);
        self.push(state, &k, &v);
        let mixed = attend(state, &q);
        self.w_o.forward(&mixed)
    }

    fn push(&self, state: &mut RingState, k: &[f32], v: &[f32]) {
        let width = self.n_heads * self.head_dim;
        let slot = state.pos % state.window;
        for h in 0..self.n_heads {
            let off = h * self.head_dim;
            let (qk, sk) = quantize_activations_sym_i8(&k[off..off + self.head_dim]);
            let (qv, sv) = quantize_activations_sym_i8(&v[off..off + self.head_dim]);
            let dst = slot * width + off;
            state.k[dst..dst + self.head_dim].copy_from_slice(&qk);
            state.v[dst..dst + self.head_dim].copy_from_slice(&qv);
            state.k_scale[slot * self.n_heads + h] = sk;
            state.v_scale[slot * self.n_heads + h] = sv;
        }
        state.pos += 1;
        if state.filled < state.window {
            state.filled += 1;
        }
    }
}

fn attend(state: &RingState, q: &[f32]) -> Vec<f32> {
    let n = state.n_heads;
    let d = state.head_dim;
    let width = n * d;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0f32; width];
    if state.filled == 0 {
        return out;
    }
    // Chronological order, oldest first.
    let start = if state.filled == state.window {
        state.pos % state.window
    } else {
        0
    };
    for h in 0..n {
        let mut scores = vec![0f32; state.filled];
        let qh = &q[h * d..h * d + d];
        for t in 0..state.filled {
            let slot = (start + t) % state.window;
            let ks = state.k_scale[slot * n + h];
            let src = slot * width + h * d;
            let mut dot = 0f32;
            for i in 0..d {
                dot += qh[i] * (state.k[src + i] as f32) * ks;
            }
            scores[t] = dot * scale;
        }
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for t in 0..state.filled {
            let w = scores[t] * inv;
            let slot = (start + t) % state.window;
            let vs = state.v_scale[slot * n + h];
            let src = slot * width + h * d;
            for i in 0..d {
                out[h * d + i] += w * (state.v[src + i] as f32) * vs;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_capacity_does_not_grow() {
        let attn = RingSlidingAttention::from_seed(32, 4, 8, 8, 3);
        let mut st = attn.new_state();
        let b0 = st.state_bytes();
        let x = vec![0.1f32; 32];
        for _ in 0..40 {
            let y = attn.forward(&x, &mut st);
            assert!(y.iter().all(|v| v.is_finite()));
            assert_eq!(st.state_bytes(), b0);
            assert!(st.filled() <= 8);
        }
        assert_eq!(st.filled(), 8);
    }
}
