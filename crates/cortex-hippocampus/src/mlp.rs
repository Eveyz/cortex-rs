//! Two-layer online neural hippocampus MLP (`m_t = W2 SiLU(W1 q̂_t)`).
//!
//! Stability invariants:
//! * Queries `q` and keys `k` are **always L2-normalized** onto the unit sphere
//!   before touching `W1`, preventing unbounded activation/gradient explosion
//!   during multi-thousand-step online updates.
//! * `W2` is initialized at `0.1×` fan-in scale so cold-start memory injections
//!   do not perturb the frozen 1.58-bit residual stream.
//! * Online plasticity uses hand-derived analytic outer-product gradients with
//!   first-order momentum (`mom1`, `mom2`) and controlled weight decay — zero
//!   autograd graph allocation.

use cortex_kernels::{crc32_ieee, l2_normalize_inplace, silu, silu_prime, Aligned64};

pub struct HippocampusWeights {
    pub d_in: usize,
    pub d_hidden: usize,
    pub d_out: usize,
    /// `[d_hidden, d_in]` row-major.
    pub w1: Aligned64<f32>,
    /// `[d_out, d_hidden]` row-major. Init scale is `0.1×` fan-in.
    pub w2: Aligned64<f32>,
    pub mom1: Aligned64<f32>,
    pub mom2: Aligned64<f32>,
    pub version: u64,
}

impl Clone for HippocampusWeights {
    fn clone(&self) -> Self {
        Self {
            d_in: self.d_in,
            d_hidden: self.d_hidden,
            d_out: self.d_out,
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            mom1: self.mom1.clone(),
            mom2: self.mom2.clone(),
            version: self.version,
        }
    }
}

impl HippocampusWeights {
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize) -> Self {
        Self::new_seeded(d_in, d_hidden, d_out, 0xA11)
    }

    pub fn new_seeded(d_in: usize, d_hidden: usize, d_out: usize, seed: u32) -> Self {
        Self {
            d_in,
            d_hidden,
            d_out,
            w1: init_matrix(d_hidden, d_in, seed, 1.0),
            // Cold-start residual protection: W2 scaled by 0.1×.
            w2: init_matrix(d_out, d_hidden, seed.wrapping_add(0x111), 0.1),
            mom1: Aligned64::zeros(d_hidden * d_in),
            mom2: Aligned64::zeros(d_out * d_hidden),
            version: 0,
        }
    }

    /// Fast-path forward read `m = W2 SiLU(W1 L2Norm(q))`.
    pub fn forward(&self, q: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.d_out];
        self.forward_into(q, &mut out);
        out
    }

    /// Allocation-light forward read into `out` (`q` is L2-normalized internally).
    pub fn forward_into(&self, q: &[f32], out: &mut [f32]) {
        assert_eq!(q.len(), self.d_in);
        assert_eq!(out.len(), self.d_out);
        let mut q_norm = q.to_vec();
        l2_normalize_inplace(&mut q_norm);
        let mut h = vec![0f32; self.d_hidden];
        let w1 = self.w1.as_slice();
        for i in 0..self.d_hidden {
            let row = &w1[i * self.d_in..(i + 1) * self.d_in];
            let z: f32 = row.iter().zip(q_norm.iter()).map(|(&w, &x)| w * x).sum();
            h[i] = silu(z);
        }
        matvec(self.w2.as_slice(), self.d_out, self.d_hidden, &h, out);
    }

    /// `(1/(2d)) ||M(L2Norm(k)) - v||^2` with `d = d_out`.
    pub fn surprise(&self, k: &[f32], v: &[f32]) -> f32 {
        0.5 * self.mean_square_error(k, v)
    }

    pub fn mean_square_error(&self, k: &[f32], v: &[f32]) -> f32 {
        assert_eq!(v.len(), self.d_out);
        let y = self.forward(k);
        let mut sse = 0f32;
        for i in 0..self.d_out {
            let e = y[i] - v[i];
            sse += e * e;
        }
        sse / self.d_out as f32
    }

    /// Pure analytic outer-product momentum SGD step on `(L2Norm(k), v)` with
    /// zero autograd. Updates both `(W1, mom1)` and `(W2, mom2)` smoothly and
    /// returns the pre-update `surprise`. Does NOT bump `version` (caller /
    /// worker bumps `version` upon arena commit).
    #[allow(clippy::too_many_arguments)]
    pub fn learn_outer_product_with_scratch(
        &mut self,
        k: &[f32],
        v: &[f32],
        tau_low: f32,
        tau_high: f32,
        base_lr: f32,
        boost_multiplier: f32,
        momentum: f32,
        weight_decay: f32,
        k_norm: &mut [f32],
        z1: &mut [f32],
        h1: &mut [f32],
        err: &mut [f32],
        grad_z1: &mut [f32],
    ) -> (f32, bool, bool) {
        let (d_in, hd, d_out) = (self.d_in, self.d_hidden, self.d_out);
        debug_assert_eq!(k.len(), d_in);
        debug_assert_eq!(v.len(), d_out);

        k_norm.copy_from_slice(k);
        l2_normalize_inplace(k_norm);

        // 1. Forward reconstruction: z1 = W1 @ k_norm, h1 = SiLU(z1), pred = W2 @ h1
        let w1_slice = self.w1.as_slice();
        for i in 0..hd {
            let row = &w1_slice[i * d_in..(i + 1) * d_in];
            let z: f32 = row.iter().zip(k_norm.iter()).map(|(&w, &x)| w * x).sum();
            z1[i] = z;
            h1[i] = silu(z);
        }

        let w2_slice = self.w2.as_slice();
        let mut sq_err = 0.0f32;
        let inv_d = 1.0 / (d_out as f32);
        for r in 0..d_out {
            let row = &w2_slice[r * hd..(r + 1) * hd];
            let pred: f32 = row.iter().zip(h1.iter()).map(|(&w, &hv)| w * hv).sum();
            let diff = pred - v[r];
            sq_err += diff * diff;
            err[r] = diff * inv_d;
        }

        let surprise = 0.5 * sq_err * inv_d;
        if surprise < tau_low {
            return (surprise, false, false);
        }

        let high = surprise >= tau_high;
        let eff_lr = if high {
            base_lr * boost_multiplier
        } else {
            base_lr
        };

        // 2. Analytic hidden gradient using pre-update W2: grad_z1 = (W2^T @ err) * silu'(z1)
        grad_z1.fill(0.0);
        for r in 0..d_out {
            let e_r = err[r];
            let row = &w2_slice[r * hd..(r + 1) * hd];
            for i in 0..hd {
                grad_z1[i] += row[i] * e_r;
            }
        }
        for i in 0..hd {
            grad_z1[i] *= silu_prime(z1[i]);
        }

        // 3. Momentum outer-product update on W2 & mom2: grad_W2 = outer(err, h1)
        let keep = 1.0 - weight_decay;
        let w2_mut = self.w2.as_mut_slice();
        let m2_mut = self.mom2.as_mut_slice();
        for r in 0..d_out {
            let e_r = err[r];
            let w2_row = &mut w2_mut[r * hd..(r + 1) * hd];
            let m2_row = &mut m2_mut[r * hd..(r + 1) * hd];
            for i in 0..hd {
                let g = e_r * h1[i];
                let m_new = momentum * m2_row[i] - eff_lr * g;
                m2_row[i] = m_new;
                w2_row[i] = w2_row[i] * keep + m_new;
            }
        }

        // 4. Momentum outer-product update on W1 & mom1: grad_W1 = outer(grad_z1, k_norm)
        let w1_mut = self.w1.as_mut_slice();
        let m1_mut = self.mom1.as_mut_slice();
        for i in 0..hd {
            let gz_i = grad_z1[i];
            let w1_row = &mut w1_mut[i * d_in..(i + 1) * d_in];
            let m1_row = &mut m1_mut[i * d_in..(i + 1) * d_in];
            for c in 0..d_in {
                let g = gz_i * k_norm[c];
                let m_new = momentum * m1_row[c] - eff_lr * g;
                m1_row[c] = m_new;
                w1_row[c] = w1_row[c] * keep + m_new;
            }
        }

        (surprise, true, high)
    }

    /// One-shot synchronous bind step: L2-normalizes `k`, applies analytic
    /// momentum outer-product updates to `W1` (`mom1`) and `W2` (`mom2`), plus
    /// a damped rank-1 residual projection on `W2` so single-exposure fact
    /// binding converges immediately. Increments [`Self::version`].
    pub fn bind_step(&mut self, k: &[f32], v: &[f32], lr_w1: f32, momentum: f32, forget: f32) {
        assert_eq!(k.len(), self.d_in);
        assert_eq!(v.len(), self.d_out);
        let mut k_norm = k.to_vec();
        l2_normalize_inplace(&mut k_norm);

        let d = self.d_out as f32;
        let mut pre = vec![0f32; self.d_hidden];
        matvec(
            self.w1.as_slice(),
            self.d_hidden,
            self.d_in,
            &k_norm,
            &mut pre,
        );
        let h: Vec<f32> = pre.iter().copied().map(silu).collect();
        let mut y = vec![0f32; self.d_out];
        matvec(self.w2.as_slice(), self.d_out, self.d_hidden, &h, &mut y);

        let mut dy = vec![0f32; self.d_out];
        for i in 0..self.d_out {
            dy[i] = (y[i] - v[i]) / d;
        }
        let mut dh = vec![0f32; self.d_hidden];
        let w2 = self.w2.as_slice();
        for j in 0..self.d_hidden {
            let mut acc = 0f32;
            for i in 0..self.d_out {
                acc += w2[i * self.d_hidden + j] * dy[i];
            }
            dh[j] = acc;
        }
        let mut dpre = vec![0f32; self.d_hidden];
        for j in 0..self.d_hidden {
            dpre[j] = dh[j] * silu_prime(pre[j]);
        }

        if forget > 0.0 {
            let keep = 1.0 - forget;
            for w in self.w1.as_mut_slice() {
                *w *= keep;
            }
            for w in self.w2.as_mut_slice() {
                *w *= keep;
            }
        }

        let w1 = self.w1.as_mut_slice();
        let m1 = self.mom1.as_mut_slice();
        for (i, &dp) in dpre.iter().enumerate().take(self.d_hidden) {
            for (j, &kj) in k_norm.iter().enumerate().take(self.d_in) {
                let idx = i * self.d_in + j;
                let g = dp * kj;
                m1[idx] = momentum * m1[idx] + g;
                w1[idx] -= lr_w1 * m1[idx];
            }
        }

        let w2_mut = self.w2.as_mut_slice();
        let m2_mut = self.mom2.as_mut_slice();
        for (i, &dyi) in dy.iter().enumerate().take(self.d_out) {
            for (j, &hj) in h.iter().enumerate().take(self.d_hidden) {
                let idx = i * self.d_hidden + j;
                let g = dyi * hj;
                m2_mut[idx] = momentum * m2_mut[idx] + g;
                w2_mut[idx] -= lr_w1 * m2_mut[idx];
            }
        }

        // Recompute hidden state and apply residual rank-1 projection on W2.
        matvec(
            self.w1.as_slice(),
            self.d_hidden,
            self.d_in,
            &k_norm,
            &mut pre,
        );
        let h: Vec<f32> = pre.iter().copied().map(silu).collect();
        matvec(self.w2.as_slice(), self.d_out, self.d_hidden, &h, &mut y);
        let hnorm: f32 = h.iter().map(|x| x * x).sum();
        if hnorm > 1e-12 {
            let inv = 1.0 / hnorm;
            let w2 = self.w2.as_mut_slice();
            for i in 0..self.d_out {
                let resid = v[i] - y[i];
                let row = i * self.d_hidden;
                for j in 0..self.d_hidden {
                    w2[row + j] += resid * h[j] * inv;
                }
            }
        }
        self.version = self.version.wrapping_add(1);
    }

    pub fn reset_momentum(&mut self) {
        self.mom1.fill(0.0);
        self.mom2.fill(0.0);
    }

    /// Serialize snapshot with `HPSP` magic (version 2 adds trailing IEEE CRC32).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"HPSP");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(self.d_in as u32).to_le_bytes());
        out.extend_from_slice(&(self.d_hidden as u32).to_le_bytes());
        out.extend_from_slice(&(self.d_out as u32).to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        push_f32(&mut out, self.w1.as_slice());
        push_f32(&mut out, self.w2.as_slice());
        let crc = crc32_ieee(&out[4..]);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 4 + 4 + 12 + 8 || &bytes[0..4] != b"HPSP" {
            return Err("bad hippocampus snapshot magic".into());
        }
        let mut o = 4;
        let ver_fmt = read_u32(bytes, &mut o)?;
        if ver_fmt != 1 && ver_fmt != 2 {
            return Err(format!("unsupported snapshot version {ver_fmt}"));
        }
        if ver_fmt == 2 {
            if bytes.len() < 32 {
                return Err("truncated snapshot crc".into());
            }
            let body = &bytes[4..bytes.len() - 4];
            let stored = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
            if crc32_ieee(body) != stored {
                return Err("hippocampus snapshot CRC32 mismatch".into());
            }
        }
        let d_in = read_u32(bytes, &mut o)? as usize;
        let d_hidden = read_u32(bytes, &mut o)? as usize;
        let d_out = read_u32(bytes, &mut o)? as usize;
        let version = read_u64(bytes, &mut o)?;
        let n1 = d_hidden * d_in;
        let n2 = d_out * d_hidden;
        let w1s = read_f32s(bytes, &mut o, n1)?;
        let w2s = read_f32s(bytes, &mut o, n2)?;
        let mut w = Self::new(d_in, d_hidden, d_out);
        w.w1.as_mut_slice().copy_from_slice(&w1s);
        w.w2.as_mut_slice().copy_from_slice(&w2s);
        w.mom1.fill(0.0);
        w.mom2.fill(0.0);
        w.version = version;
        Ok(w)
    }

    pub fn copy_params_from(&mut self, other: &Self) {
        assert_eq!(self.d_in, other.d_in);
        assert_eq!(self.d_hidden, other.d_hidden);
        assert_eq!(self.d_out, other.d_out);
        self.w1.as_mut_slice().copy_from_slice(other.w1.as_slice());
        self.w2.as_mut_slice().copy_from_slice(other.w2.as_slice());
        self.mom1
            .as_mut_slice()
            .copy_from_slice(other.mom1.as_slice());
        self.mom2
            .as_mut_slice()
            .copy_from_slice(other.mom2.as_slice());
        self.version = other.version;
    }
}

fn init_matrix(rows: usize, cols: usize, salt: u32, scale: f32) -> Aligned64<f32> {
    let mut m = Aligned64::zeros(rows * cols);
    let fan = (cols as f32).sqrt().max(1.0);
    let s = scale / fan;
    for (i, slot) in m.as_mut_slice().iter_mut().enumerate() {
        *slot = det(i, salt) * s;
    }
    m
}

fn det(i: usize, salt: u32) -> f32 {
    let mut x = (i as u32).wrapping_mul(0x9E37_79B1) ^ salt;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    (x as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

pub fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);
    for (r, yr) in y.iter_mut().enumerate().take(rows) {
        let mut acc = 0f32;
        let row = r * cols;
        for c in 0..cols {
            acc += w[row + c] * x[c];
        }
        *yr = acc;
    }
}

fn push_f32(out: &mut Vec<u8>, xs: &[f32]) {
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn read_u32(bytes: &[u8], o: &mut usize) -> Result<u32, String> {
    if *o + 4 > bytes.len() {
        return Err("truncated u32".into());
    }
    let v = u32::from_le_bytes(bytes[*o..*o + 4].try_into().unwrap());
    *o += 4;
    Ok(v)
}

fn read_u64(bytes: &[u8], o: &mut usize) -> Result<u64, String> {
    if *o + 8 > bytes.len() {
        return Err("truncated u64".into());
    }
    let v = u64::from_le_bytes(bytes[*o..*o + 8].try_into().unwrap());
    *o += 8;
    Ok(v)
}

fn read_f32s(bytes: &[u8], o: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if *o + 4 > bytes.len() {
            return Err("truncated f32".into());
        }
        let v = f32::from_le_bytes(bytes[*o..*o + 4].try_into().unwrap());
        *o += 4;
        out.push(v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w2_init_is_scaled_by_0_1() {
        let h = HippocampusWeights::new(16, 32, 16);
        let fan = (32f32).sqrt();
        let max_w2 = h.w2.as_slice().iter().map(|v| v.abs()).fold(0f32, f32::max);
        let max_w1 = h.w1.as_slice().iter().map(|v| v.abs()).fold(0f32, f32::max);
        assert!(max_w2 > 0.0);
        assert!(max_w2 <= 0.1 / fan + 1e-6);
        assert!(max_w1 > max_w2 * 5.0);
    }

    #[test]
    fn q_and_k_are_l2_normalized_invariant_to_input_magnitude() {
        let h = HippocampusWeights::new(16, 32, 16);
        let q_small: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let q_huge: Vec<f32> = q_small.iter().map(|x| x * 10_000.0).collect();
        let y1 = h.forward(&q_small);
        let y2 = h.forward(&q_huge);
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "L2 norm invariance broken: {a} vs {b}"
            );
        }
    }
}
