//! Gated DeltaNet: O(d_head²) rank-1 recurrence + WY chunkwise parallel prefill.
//!
//! # Mathematical & Storage Convention
//!
//! State `S` per head has shape `[head_dim, head_dim]`, stored row-major with
//! row = value axis and col = key axis (`S[i, j]`). After per-head L2
//! normalization of `q_t` and `k_t`:
//!
//! ```text
//! u_t   = S_{t-1} k_t                                     # O(d²)
//! S_t   = α_t S_{t-1} + β_t (v_t - α_t u_t) k_t^T         # O(d²) rank-1 update
//!       = α_t S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T
//! o_t   = S_t q_t                                         # O(d²) associative read
//! ```
//!
//! Under this storage orientation, writing `(k, v)` with `α = 1, β = 1` sets
//! `S = v k^T`, and querying with `q = k` (`‖k‖₂ = 1`) returns
//! `o = (v k^T) k = v (k^T k) = v` — exact one-shot associative recall and
//! exact erasure of stale values bound to the same key.

use cortex_kernels::{l2_normalize_inplace, Aligned64};

/// Caller-owned recurrent state (`[n_heads, head_dim, head_dim]`, 64B-aligned).
pub struct DeltaState {
    /// Row-major `S[head, val_i, key_j]`.
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

    /// Byte size of `S`. Constant in sequence length `T`.
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

/// In-place O(d_head²) rank-1 step for every head.
///
/// `q`, `k`, `v` are concatenated heads, each of length `head_dim`.
/// `alpha` and `beta` are one scalar per head in `(0, 1]`.
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
        l2_normalize_inplace(&mut qh);
        l2_normalize_inplace(&mut kh);
        let vh = &v[base_q..base_q + d];
        let a = alpha[h];
        let b = beta[h];
        let sb = h * d * d;
        // u = S @ k
        for (i, ui) in u.iter_mut().enumerate().take(d) {
            let mut acc = 0f32;
            let row = sb + i * d;
            for j in 0..d {
                acc += s[row + j] * kh[j];
            }
            *ui = acc;
        }
        // S = α S + β (v - α u) k^T
        for i in 0..d {
            let coeff = b * (vh[i] - a * u[i]);
            let row = sb + i * d;
            for j in 0..d {
                s[row + j] = a * s[row + j] + coeff * kh[j];
            }
        }
        // o = S @ q (updated state)
        for i in 0..d {
            let mut acc = 0f32;
            let row = sb + i * d;
            for j in 0..d {
                acc += s[row + j] * qh[j];
            }
            o[base_q + i] = acc;
        }
    }
    o
}

/// Chunkwise WY / lower-triangular forward over `chunk_len` tokens (`[L, n_heads * d]`).
///
/// Uses cumulative decay `Γ_t = Π_{s<=t} α_s` to reduce gated delta updates to
/// a lower-triangular system `(I + diag(β) A) w = diag(β) e`. Falls back to
/// sequential [`deltanet_step`] if `Γ_{L-1} < 1e-3` to guard against underflow.
pub fn deltanet_chunk(
    state: &mut DeltaState,
    qs: &[f32],
    ks: &[f32],
    vs: &[f32],
    alphas: &[f32],
    betas: &[f32],
    chunk_len: usize,
) -> Vec<f32> {
    let n = state.n_heads;
    let d = state.head_dim;
    let width = n * d;
    let l = chunk_len;
    assert_eq!(qs.len(), l * width);
    assert_eq!(ks.len(), l * width);
    assert_eq!(vs.len(), l * width);
    assert_eq!(alphas.len(), l * n);
    assert_eq!(betas.len(), l * n);
    if l == 0 {
        return Vec::new();
    }
    if l == 1 {
        return deltanet_step(state, qs, ks, vs, alphas, betas);
    }

    let mut outs = vec![0f32; l * width];
    for h in 0..n {
        let mut gamma = vec![0f32; l];
        let mut g = 1.0f32;
        for t in 0..l {
            g *= alphas[t * n + h];
            gamma[t] = g;
        }

        let sb = h * d * d;
        // Gather normalized per-token q, k for this head.
        let mut q_h = vec![0f32; l * d];
        let mut k_h = vec![0f32; l * d];
        for t in 0..l {
            let src = t * width + h * d;
            q_h[t * d..(t + 1) * d].copy_from_slice(&qs[src..src + d]);
            k_h[t * d..(t + 1) * d].copy_from_slice(&ks[src..src + d]);
            l2_normalize_inplace(&mut q_h[t * d..(t + 1) * d]);
            l2_normalize_inplace(&mut k_h[t * d..(t + 1) * d]);
        }

        if gamma[l - 1] < 1e-3 {
            // Sequential fallback for this head when cumulative decay is tiny.
            let s = &mut state.s.as_mut_slice()[sb..sb + d * d];
            let mut u = vec![0f32; d];
            for t in 0..l {
                let qt = &q_h[t * d..(t + 1) * d];
                let kt = &k_h[t * d..(t + 1) * d];
                let vt = &vs[t * width + h * d..t * width + (h + 1) * d];
                let a = alphas[t * n + h];
                let b = betas[t * n + h];
                for i in 0..d {
                    let mut acc = 0f32;
                    for j in 0..d {
                        acc += s[i * d + j] * kt[j];
                    }
                    u[i] = acc;
                }
                for i in 0..d {
                    let coeff = b * (vt[i] - a * u[i]);
                    for j in 0..d {
                        s[i * d + j] = a * s[i * d + j] + coeff * kt[j];
                    }
                }
                for i in 0..d {
                    let mut acc = 0f32;
                    for j in 0..d {
                        acc += s[i * d + j] * qt[j];
                    }
                    outs[t * width + h * d + i] = acc;
                }
            }
            continue;
        }

        let s0 = state.s.as_slice()[sb..sb + d * d].to_vec();
        let mut e = vec![0f32; l * d];
        for t in 0..l {
            let kt = &k_h[t * d..(t + 1) * d];
            let vt = &vs[t * width + h * d..t * width + (h + 1) * d];
            let inv_g = 1.0 / gamma[t];
            for i in 0..d {
                let mut pred = 0f32;
                for j in 0..d {
                    pred += s0[i * d + j] * kt[j];
                }
                e[t * d + i] = vt[i] * inv_g - pred;
            }
        }

        // Forward substitution: w_t = β_t e_t - β_t Σ_{j < t} (k_j · k_t) w_j
        let mut w = vec![0f32; l * d];
        for t in 0..l {
            let kt = &k_h[t * d..(t + 1) * d];
            let bt = betas[t * n + h];
            for j in 0..t {
                let kj = &k_h[j * d..(j + 1) * d];
                let dot_jk: f32 = kj.iter().zip(kt.iter()).map(|(x, y)| x * y).sum();
                if dot_jk != 0.0 {
                    let scale = bt * dot_jk;
                    for r in 0..d {
                        w[t * d + r] -= scale * w[j * d + r];
                    }
                }
            }
            for r in 0..d {
                w[t * d + r] += bt * e[t * d + r];
            }
        }

        // Outputs: o_t = Γ_t * (S_0 q_t + Σ_{j <= t} w_j (k_j · q_t))
        for t in 0..l {
            let qt = &q_h[t * d..(t + 1) * d];
            let gt = gamma[t];
            for r in 0..d {
                let mut acc = 0f32;
                for c in 0..d {
                    acc += s0[r * d + c] * qt[c];
                }
                outs[t * width + h * d + r] = acc;
            }
            for j in 0..=t {
                let kj = &k_h[j * d..(j + 1) * d];
                let dot_jq: f32 = kj.iter().zip(qt.iter()).map(|(x, y)| x * y).sum();
                if dot_jq != 0.0 {
                    for r in 0..d {
                        outs[t * width + h * d + r] += w[j * d + r] * dot_jq;
                    }
                }
            }
            for r in 0..d {
                outs[t * width + h * d + r] *= gt;
            }
        }

        // End-of-chunk state: S_L = Γ_{L-1} * (S_0 + Σ_j w_j ⊗ k_j)
        let s_mut = &mut state.s.as_mut_slice()[sb..sb + d * d];
        let g_last = gamma[l - 1];
        for r in 0..d {
            for c in 0..d {
                let mut acc = s0[r * d + c];
                for j in 0..l {
                    acc += w[j * d + r] * k_h[j * d + c];
                }
                s_mut[r * d + c] = g_last * acc;
            }
        }
    }
    outs
}

/// Expanded O(d³) matrix oracle used ONLY in unit tests to verify the O(d²)
/// rank-1 identity `S' = α S (I - β k k^T) + β v k^T`, `o = S' q`.
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
    l2_normalize_inplace(&mut q);
    l2_normalize_inplace(&mut k);
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
    for i in 0..d {
        let mut acc = 0f32;
        for j in 0..d {
            acc += s[i * d + j] * q[j];
        }
        o[i] = acc;
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_unit(seed: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u32).wrapping_mul(0x045D_9F3B) ^ seed;
                x ^= x >> 16;
                (x as f32) / (u32::MAX as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn delta_step_binds_and_overwrites_associative_fact() {
        let d = 8usize;
        let mut state = DeltaState::zeros(1, d);
        let k = fill_unit(11, d);
        let v1 = fill_unit(22, d);
        let v2 = fill_unit(33, d);

        // One-shot write (α=1, β=1) then query with q = k: must recall v1 exactly!
        let o1 = deltanet_step(&mut state, &k, &k, &v1, &[1.0], &[1.0]);
        for i in 0..d {
            assert!(
                (o1[i] - v1[i]).abs() < 1e-5,
                "recall v1 mismatch at {i}: {} vs {}",
                o1[i],
                v1[i]
            );
        }

        // Overwrite same key k with v2: delta rule erases v1 and returns v2!
        let o2 = deltanet_step(&mut state, &k, &k, &v2, &[1.0], &[1.0]);
        for i in 0..d {
            assert!(
                (o2[i] - v2[i]).abs() < 1e-5,
                "overwrite v2 mismatch at {i}: {} vs {}",
                o2[i],
                v2[i]
            );
        }
    }

    #[test]
    fn rank1_matches_matrix_form() {
        let d = 8usize;
        let mut s_a = DeltaState::zeros(1, d);
        let mut s_b = vec![0f32; d * d];
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
    fn chunkwise_matches_sequential_normal_and_fallback() {
        for &alpha_base in &[0.985f32, 0.5f32] {
            let (n_heads, d, l) = (2usize, 8usize, 16usize);
            let width = n_heads * d;
            let qs = fill_unit(101, l * width);
            let ks = fill_unit(202, l * width);
            let vs = fill_unit(303, l * width);
            let alphas: Vec<f32> = (0..l * n_heads)
                .map(|i| alpha_base - 0.002 * ((i % 3) as f32))
                .collect();
            let betas: Vec<f32> = (0..l * n_heads)
                .map(|i| 0.25 + 0.05 * ((i % 4) as f32))
                .collect();

            let mut st_seq = DeltaState::zeros(n_heads, d);
            let mut out_seq = Vec::with_capacity(l * width);
            for t in 0..l {
                let o = deltanet_step(
                    &mut st_seq,
                    &qs[t * width..(t + 1) * width],
                    &ks[t * width..(t + 1) * width],
                    &vs[t * width..(t + 1) * width],
                    &alphas[t * n_heads..(t + 1) * n_heads],
                    &betas[t * n_heads..(t + 1) * n_heads],
                );
                out_seq.extend_from_slice(&o);
            }

            let mut st_chunk = DeltaState::zeros(n_heads, d);
            let out_chunk = deltanet_chunk(&mut st_chunk, &qs, &ks, &vs, &alphas, &betas, l);

            let max_out_err = out_seq
                .iter()
                .zip(out_chunk.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let max_st_err = st_seq
                .as_slice()
                .iter()
                .zip(st_chunk.as_slice().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max_out_err < 1e-4, "chunk out err {max_out_err}");
            assert!(max_st_err < 1e-4, "chunk state err {max_st_err}");
        }
    }

    #[test]
    fn test_deltanet_constant_memory() {
        let n_heads = 4usize;
        let head_dim = 16usize;
        let mut state = DeltaState::zeros(n_heads, head_dim);
        let bytes0 = state.state_bytes();
        let width = n_heads * head_dim;
        assert_eq!(bytes0, n_heads * head_dim * head_dim * 4);
        for t in 0..2048u32 {
            let q = fill_unit(t.wrapping_mul(3), width);
            let k = fill_unit(t.wrapping_mul(5).wrapping_add(1), width);
            let v = fill_unit(t.wrapping_mul(7).wrapping_add(2), width);
            let alpha: Vec<f32> = (0..n_heads).map(|h| 0.95 + 0.01 * h as f32).collect();
            let beta: Vec<f32> = (0..n_heads).map(|h| 0.15 + 0.02 * h as f32).collect();
            let o = deltanet_step(&mut state, &q, &k, &v, &alpha, &beta);
            assert_eq!(o.len(), width);
            assert!(o.iter().all(|v| v.is_finite()), "NaN/Inf at step {t}");
            assert!(state.as_slice().iter().all(|v| v.is_finite()));
            assert_eq!(state.state_bytes(), bytes0);
        }
    }
}
