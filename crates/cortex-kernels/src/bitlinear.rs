//! BitLinear: symmetric INT8 activations × packed ternary weights.
//!
//! `y[r] = acc_i32[r] as f32 * x_scale * row_scales[r]`
//! with `x_scale = absmax(x) / 127`. The integer GEMV is exact; the scale is
//! applied once per row at the end.

use crate::align::Aligned64;
use crate::gemv::{gemv_bit158, gemv_bit158_scalar};
use crate::tile::{pack_ternary, PackedMatrix, TritError};

/// Half-away-from-zero rounding. Matched by `tools/train_seed_qat.py`.
#[inline]
pub fn round_half_away(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    x.signum() * (x.abs() + 0.5).floor()
}

/// Quantize a vector to symmetric INT8.
///
/// The scale is `absmax / 127`, so the peak maps to ±127 and **-128 is never
/// produced**. Zeros map to zeros with scale `1` (a harmless convention: the
/// integer activations are all zero).
pub fn quantize_activations_sym_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let mut absmax = 0f32;
    for v in x {
        if v.is_finite() {
            absmax = absmax.max(v.abs());
        }
    }
    if absmax == 0.0 {
        return (vec![0i8; x.len()], 1.0);
    }
    let scale = absmax / 127.0;
    let q = x
        .iter()
        .map(|v| {
            if !v.is_finite() {
                return 0i8;
            }
            let r = round_half_away(v / scale);
            let c = if r > 127.0 {
                127.0
            } else if r < -127.0 {
                -127.0
            } else {
                r
            };
            c as i8
        })
        .collect();
    (q, scale)
}

/// Explicit-scale quantizer used to prove the clamp. Still never emits -128.
pub fn quantize_with_scale(x: &[f32], scale: f32) -> Vec<i8> {
    let scale = if scale.abs() < 1e-12 { 1.0 } else { scale };
    x.iter()
        .map(|v| {
            let r = round_half_away(v / scale);
            let c = if r > 127.0 {
                127.0
            } else if r < -127.0 {
                -127.0
            } else {
                r
            };
            c as i8
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct BitLinearWeights {
    pub rows: usize,
    pub cols: usize,
    pub rows_pad: usize,
    pub cols_pad: usize,
    pub blocks: Vec<crate::tile::CacheLine16x16>,
    /// Per-output-row FP32 scale. Length `rows` (logical, not padded).
    pub row_scales: Aligned64<f32>,
}

impl BitLinearWeights {
    pub fn from_ternary(
        weights: &[i8],
        rows: usize,
        cols: usize,
        row_scales: &[f32],
    ) -> Result<Self, TritError> {
        if row_scales.len() != rows {
            return Err(TritError::NotTernary(0));
        }
        let packed: PackedMatrix = pack_ternary(weights, rows, cols)?;
        let mut scales = Aligned64::zeros(rows);
        scales.as_mut_slice().copy_from_slice(row_scales);
        Ok(Self {
            rows: packed.rows,
            cols: packed.cols,
            rows_pad: packed.rows_pad,
            cols_pad: packed.cols_pad,
            blocks: packed.blocks,
            row_scales: scales,
        })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        let scales = vec![1.0f32; rows];
        let w = vec![0i8; rows * cols];
        Self::from_ternary(&w, rows, cols, &scales).expect("zero ternary")
    }

    /// Deterministic non-zero ternary, scales default to 1.
    pub fn from_packed(
        rows: usize,
        cols: usize,
        rows_pad: usize,
        cols_pad: usize,
        blocks: Vec<crate::tile::CacheLine16x16>,
        row_scales: &[f32],
    ) -> Result<Self, TritError> {
        if row_scales.len() != rows {
            return Err(TritError::NotTernary(0));
        }
        if rows_pad % 16 != 0 || cols_pad % 16 != 0 {
            return Err(TritError::NotTernary(0));
        }
        let expect = if rows_pad == 0 || cols_pad == 0 {
            0
        } else {
            (rows_pad / 16) * (cols_pad / 16)
        };
        if blocks.len() != expect {
            return Err(TritError::NotTernary(0));
        }
        let mut scales = Aligned64::zeros(rows);
        if rows > 0 {
            scales.as_mut_slice().copy_from_slice(row_scales);
        }
        Ok(Self {
            rows,
            cols,
            rows_pad,
            cols_pad,
            blocks,
            row_scales: scales,
        })
    }

    pub fn from_seed(rows: usize, cols: usize, seed: u32) -> Self {
        let mut w = vec![0i8; rows * cols];
        for (i, slot) in w.iter_mut().enumerate() {
            let mut x = (i as u32).wrapping_mul(0x9E37_79B1) ^ seed;
            x ^= x >> 16;
            x = x.wrapping_mul(0x7FEB_352D);
            x ^= x >> 15;
            *slot = match x % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            };
        }
        let scales = vec![1.0f32; rows];
        Self::from_ternary(&w, rows, cols, &scales).expect("seed ternary")
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.forward_with(x, false)
    }

    pub fn forward_scalar(&self, x: &[f32]) -> Vec<f32> {
        self.forward_with(x, true)
    }

    fn forward_with(&self, x: &[f32], force_scalar: bool) -> Vec<f32> {
        assert_eq!(x.len(), self.cols, "bitlinear k-dim");
        if self.rows == 0 || self.cols == 0 {
            return vec![0.0; self.rows];
        }
        let (q, x_scale) = quantize_activations_sym_i8(x);
        let mut qpad = vec![0i8; self.cols_pad];
        qpad[..q.len()].copy_from_slice(&q);
        let mut acc = vec![0i32; self.rows_pad];
        if force_scalar {
            gemv_bit158_scalar(&self.blocks, &qpad, &mut acc, self.rows_pad, self.cols_pad);
        } else {
            gemv_bit158(&self.blocks, &qpad, &mut acc, self.rows_pad, self.cols_pad);
        }
        let scales = self.row_scales.as_slice();
        (0..self.rows)
            .map(|r| acc[r] as f32 * x_scale * scales[r])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_never_emits_minus_128() {
        let x = [-1.0e20, -3.5, -0.0, 0.4, 127.0, 1.0e20, f32::NAN];
        let (q, scale) = quantize_activations_sym_i8(&x);
        assert!(scale.is_finite() && scale > 0.0);
        assert!(q
            .iter()
            .all(|v| i16::from(*v) >= -127 && i16::from(*v) <= 127));
        assert!(!q.contains(&i8::MIN));
        // A hostile scale that would round past -128 is clamped.
        let q2 = quantize_with_scale(&[-1000.0, 1000.0, -128.4], 1.0);
        assert_eq!(q2, vec![-127, 127, -127]);
    }

    #[test]
    fn hand_gemv_matches_formula() {
        // 1×4 logical, padded. W = [1, -1, 0, 1], x = [1, 1, 1, 1], scale row = 2.
        let mut w = vec![0i8; 16 * 16];
        w[0] = 1;
        w[1] = -1;
        w[2] = 0;
        w[3] = 1;
        let mut scales = vec![0.0f32; 16];
        scales[0] = 2.0;
        // Build via the logical 1×4 API so padding is zeros.
        let weights = vec![1i8, -1, 0, 1];
        let layer = BitLinearWeights::from_ternary(&weights, 1, 4, &[2.0]).unwrap();
        let y = layer.forward_scalar(&[1.0, 1.0, 1.0, 1.0]);
        // absmax=1, q=127, acc = 127 - 127 + 0 + 127 = 127
        // y = 127 * (1/127) * 2 = 2
        assert!(
            (y[0] - 2.0).abs() < 1e-5,
            "y={y:?} scales={scales:?} w0={}",
            w[0]
        );
    }

    /// DoD: unaligned logical shape 37×523, SIMD vs scalar, max abs err < 1e-5.
    #[test]
    fn test_simd_bitexact() {
        let rows = 37usize;
        let cols = 523usize;
        let mut w = vec![0i8; rows * cols];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = match (i * 3 + 1) % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            };
        }
        let scales: Vec<f32> = (0..rows).map(|r| 0.5 + (r % 7) as f32 * 0.125).collect();
        let layer = BitLinearWeights::from_ternary(&w, rows, cols, &scales).unwrap();
        // Misaligned activation view (offset by one f32) plus an awkward K.
        let mut buf = vec![0f32; cols + 1];
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = ((i * 17) % 50) as f32 - 25.0;
        }
        let x = &buf[1..1 + cols];
        assert_ne!(x.as_ptr() as usize % 64, 0, "x pointer should be unaligned");
        let y_simd = layer.forward(x);
        let y_scalar = layer.forward_scalar(x);
        assert_eq!(y_simd.len(), 37);
        let mut max_err = 0f32;
        for (a, b) in y_simd.iter().zip(y_scalar.iter()) {
            max_err = max_err.max((a - b).abs());
            assert!(a.is_finite() && b.is_finite());
        }
        assert!(
            max_err < 1e-5,
            "simd vs scalar max abs err {max_err} on 37x523"
        );
    }
}
