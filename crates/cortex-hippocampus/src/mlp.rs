//! Two-layer memory MLP.
//!
//! ```text
//! m = W2 SiLU(W1 q)
//! Surprise_t = (1 / (2d)) || m - v_t ||^2
//! ```
//!
//! `W2` is initialized at `0.1` times a fan-in scale. Updates are analytic
//! outer products (no autograd): momentum on `W1`, then a one-step Newton
//! correction on `W2` so a single accepted write can bind `(k, v)`.

use cortex_kernels::{silu, silu_prime, Aligned64};

pub struct HippocampusWeights {
    pub d_in: usize,
    pub d_hidden: usize,
    pub d_out: usize,
    /// `[d_hidden, d_in]` row-major.
    pub w1: Aligned64<f32>,
    /// `[d_out, d_hidden]` row-major. Init scale is 0.1× fan-in.
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
        Self {
            d_in,
            d_hidden,
            d_out,
            w1: init_matrix(d_hidden, d_in, 0xA11, 1.0),
            // Brief: init W2 scaled by 0.1.
            w2: init_matrix(d_out, d_hidden, 0xB22, 0.1),
            mom1: Aligned64::zeros(d_hidden * d_in),
            mom2: Aligned64::zeros(d_out * d_hidden),
            version: 0,
        }
    }

    pub fn forward(&self, q: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), self.d_in);
        let mut pre = vec![0f32; self.d_hidden];
        matvec(self.w1.as_slice(), self.d_hidden, self.d_in, q, &mut pre);
        let h: Vec<f32> = pre.iter().copied().map(silu).collect();
        let mut y = vec![0f32; self.d_out];
        matvec(self.w2.as_slice(), self.d_out, self.d_hidden, &h, &mut y);
        y
    }

    /// `(1/(2d)) ||M(k) - v||^2` with `d = d_out`.
    pub fn surprise(&self, k: &[f32], v: &[f32]) -> f32 {
        assert_eq!(v.len(), self.d_out);
        let y = self.forward(k);
        let mut sse = 0f32;
        for i in 0..self.d_out {
            let e = y[i] - v[i];
            sse += e * e;
        }
        sse / (2.0 * self.d_out as f32)
    }

    pub fn mean_square_error(&self, k: &[f32], v: &[f32]) -> f32 {
        let y = self.forward(k);
        let mut sse = 0f32;
        for i in 0..self.d_out {
            let e = y[i] - v[i];
            sse += e * e;
        }
        sse / self.d_out as f32
    }

    /// Analytic write. `lr_w1` scales the momentum step on `W1`. `W2` then
    /// takes the rank-1 Newton step that realises `W2 h = v` for the hidden
    /// state produced by the updated `W1`. Increments [`Self::version`].
    pub fn bind_step(&mut self, k: &[f32], v: &[f32], lr_w1: f32, momentum: f32, forget: f32) {
        assert_eq!(k.len(), self.d_in);
        assert_eq!(v.len(), self.d_out);
        let d = self.d_out as f32;
        let mut pre = vec![0f32; self.d_hidden];
        matvec(self.w1.as_slice(), self.d_hidden, self.d_in, k, &mut pre);
        let h: Vec<f32> = pre.iter().copied().map(silu).collect();
        let mut y = vec![0f32; self.d_out];
        matvec(self.w2.as_slice(), self.d_out, self.d_hidden, &h, &mut y);

        let mut dy = vec![0f32; self.d_out];
        for i in 0..self.d_out {
            dy[i] = (y[i] - v[i]) / d;
        }
        let mut dh = vec![0f32; self.d_hidden];
        // W2 is [d_out, d_hidden], so W2^T dy is over hidden.
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
        }
        let w1 = self.w1.as_mut_slice();
        let m1 = self.mom1.as_mut_slice();
        for i in 0..self.d_hidden {
            for j in 0..self.d_in {
                let idx = i * self.d_in + j;
                let g = dpre[i] * k[j];
                m1[idx] = momentum * m1[idx] + g;
                w1[idx] -= lr_w1 * m1[idx];
            }
        }

        // Recompute hidden state so the Newton step matches the next forward.
        matvec(self.w1.as_slice(), self.d_hidden, self.d_in, k, &mut pre);
        let h: Vec<f32> = pre.iter().copied().map(silu).collect();
        matvec(self.w2.as_slice(), self.d_out, self.d_hidden, &h, &mut y);
        let mut hnorm = 0f32;
        for x in &h {
            hnorm += x * x;
        }
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

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"HPSP");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(self.d_in as u32).to_le_bytes());
        out.extend_from_slice(&(self.d_hidden as u32).to_le_bytes());
        out.extend_from_slice(&(self.d_out as u32).to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        push_f32(&mut out, self.w1.as_slice());
        push_f32(&mut out, self.w2.as_slice());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 4 + 4 + 12 + 8 || &bytes[0..4] != b"HPSP" {
            return Err("bad hippocampus snapshot magic".into());
        }
        let mut o = 4;
        let ver_fmt = read_u32(bytes, &mut o)?;
        if ver_fmt != 1 {
            return Err(format!("unsupported snapshot version {ver_fmt}"));
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
    for r in 0..rows {
        let mut acc = 0f32;
        let row = r * cols;
        for c in 0..cols {
            acc += w[row + c] * x[c];
        }
        y[r] = acc;
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
        // W1 uses scale 1, so it should sit about 10× higher than W2.
        assert!(max_w1 > max_w2 * 5.0);
    }
}
