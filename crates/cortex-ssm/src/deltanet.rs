//! Gated DeltaNet, rank-1 form.
//!
//! After per-head L2-normalization of `q` and `k`:
//!
//! ```text
//! S_t = α_t S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T
//! o_t = S_t^T q_t
//! ```
//!
//! Implemented as the algebraically identical O(d_head²) update
//!
//! ```text
//! u = S_{t-1} k_t
//! S_t = α_t S_{t-1} + β_t (v_t - α_t u) k_t^T
//! ```
//!
//! State shape is `[num_heads, head_dim, head_dim]`, fixed for the life of
//! the session. Memory is O(1) in sequence length: there is no KV cache here
//! (the ring window lives in [`crate::ring_attn`]).

use cortex_kernels::Aligned64;

/// Caller-owned recurrent state. Scratch for `u` is included and fixed-size.
pub struct DeltaState {
    /// Row-major `S[head, i, j]`.
    s: Aligned64<f32>,
    n_heads: usize,
    head_dim: usize,
}

impl DeltaState {
    pub fn zeros(n_heads: usize, head_dim: usize) -> Self {
        Self {
            s: Aligned64::zeros(n_heads * head_dim * head_dim),
            n_heads,
            head_dim,
        }
    }

    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Byte size of `S`. Independent of how many tokens have been applied.
    pub fn state_bytes(&self) -> usize {
        self.s.len() * std::mem::size_of::<f32>()
    }

    pub fn as_slice(&self) -> &[f32] {
        self.s.as_slice()
    }

    pub fn reset(&mut self) {
        self.s.fill(0.0);
    }
}

fn l2_normalize(v: &mut [f32]) {
    let mut ss = 0f32;
    for x in v.iter() {
        ss += x * x;
    }
    let n = ss.sqrt();
    if n > 1e-8 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// In-place rank-1 step for every head.
///
/// `q`, `k`, `v` are concatenated heads, each of length `head_dim`.
/// `alpha` and `beta` are one scalar per head, already in `(0, 1)`.
/// Returns the concatenated `o` (`n_heads * head_dim`).
pub fn deltanet_step(
    state: &mut DeltaState,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let n = state.n_heads;
    let d = state.head_dim;
    assert_eq!(q.len(), n * d);
    assert_eq!(k.len(), n * d);
    assert_eq!(v.len(), n * d);
    assert_eq!(alpha.len(), n);
    assert_eq!(beta.len(), n);
    let mut o = vec![0f32; n * d];
    let s = state.s.as_mut_slice();
    let mut qh = vec![0f32; d];
    let mut kh = vec![0f32; d];
    let mut u = vec![0f32; d];
    for h in 0..n {
        let base_q = h * d;
        qh.copy_from_slice(&q[base_q..base_q + d]);
        kh.copy_from_slice(&k[base_q..base_q + d]);
        l2_normalize(&mut qh);
        l2_normalize(&mut kh);
        let vh = &v[base_q..base_q + d];
        let a = alpha[h];
        let b = beta[h];
        let sb = h * d * d;
        // u = S k
        for i in 0..d {
            let mut acc = 0f32;
            let row = sb + i * d;
            for j in 0..d {
                acc += s[row + j] * kh[j];
            }
            u[i] = acc;
        }
        // S = α S + β (v - α u) k^T
        for i in 0..d {
            let coeff = b * (vh[i] - a * u[i]);
            let row = sb + i * d;
            for j in 0..d {
                s[row + j] = a * s[row + j] + coeff * kh[j];
            }
        }
        // o = S^T q
        for j in 0..d {
            let mut acc = 0f32;
            for i in 0..d {
                acc += s[sb + i * d + j] * qh[i];
            }
            o[base_q + j] = acc;
        }
    }
    o
}

/// Expanded matrix form, used to lock the rank-1 identity on one head.
/// `S` is row-major `[d, d]`.
pub fn deltanet_step_matrix_form(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let d = q.len();
    assert_eq!(s.len(), d * d);
    let mut q = q.to_vec();
    let mut k = k.to_vec();
    l2_normalize(&mut q);
    l2_normalize(&mut k);
    // M = I - β k k^T ; S' = α S M + β v k^T
    let mut sm = vec![0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0f32;
            for t in 0..d {
                let m_tj = if t == j { 1.0 } else { 0.0 } - beta * k[t] * k[j];
                acc += s[i * d + t] * m_tj;
            }
            sm[i * d + j] = alpha * acc + beta * v[i] * k[j];
        }
    }
    s.copy_from_slice(&sm);
    let mut o = vec![0f32; d];
    for j in 0..d {
        let mut acc = 0f32;
        for i in 0..d {
            acc += s[i * d + j] * q[i];
        }
        o[j] = acc;
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_unit(seed: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u32).wrapping_mul(0x45D9_F3B) ^ seed;
                x ^= x >> 16;
                (x as f32) / (u32::MAX as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn rank1_matches_matrix_form() {
        let d = 8usize;
        let mut s_a = DeltaState::zeros(1, d);
        let mut s_b = vec![0f32; d * d];
        // Non-zero initial state.
        for (i, slot) in s_a.s.as_mut_slice().iter_mut().enumerate() {
            *slot = ((i % 5) as f32) * 0.01;
        }
        s_b.copy_from_slice(s_a.as_slice());
        for t in 0..6 {
            let q = fill_unit(10 + t, d);
            let k = fill_unit(30 + t, d);
            let v = fill_unit(50 + t, d);
            let alpha = 0.4 + (t as f32) * 0.05;
            let beta = 0.2 + (t as f32) * 0.03;
            let o1 = deltanet_step(&mut s_a, &q, &k, &v, &[alpha], &[beta]);
            let o2 = deltanet_step_matrix_form(&mut s_b, &q, &k, &v, alpha, beta);
            for (a, b) in o1.iter().zip(o2.iter()) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
            for (a, b) in s_a.as_slice().iter().zip(s_b.iter()) {
                assert!((a - b).abs() < 1e-5, "S {a} vs {b}");
            }
        }
    }

    #[test]
    fn test_deltanet_constant_memory() {
        // Stream 2048 steps. State storage does not grow with T.
        let n_heads = 4usize;
        let head_dim = 16usize;
        let mut state = DeltaState::zeros(n_heads, head_dim);
        let bytes0 = state.state_bytes();
        let width = n_heads * head_dim;
        assert!(bytes0 > 0);
        assert_eq!(bytes0, n_heads * head_dim * head_dim * 4);
        for t in 0..2048u32 {
            let q = fill_unit(t.wrapping_mul(3), width);
            let k = fill_unit(t.wrapping_mul(5).wrapping_add(1), width);
            let v = fill_unit(t.wrapping_mul(7).wrapping_add(2), width);
            let alpha: Vec<f32> = (0..n_heads).map(|h| 0.3 + 0.05 * h as f32).collect();
            let beta: Vec<f32> = (0..n_heads).map(|h| 0.15 + 0.02 * h as f32).collect();
            let o = deltanet_step(&mut state, &q, &k, &v, &alpha, &beta);
            assert_eq!(o.len(), width);
            assert!(o.iter().all(|v| v.is_finite()), "NaN/Inf at step {t}");
            assert!(state.as_slice().iter().all(|v| v.is_finite()));
            assert_eq!(state.state_bytes(), bytes0);
            assert_eq!(state.as_slice().len(), n_heads * head_dim * head_dim);
        }
    }
}
