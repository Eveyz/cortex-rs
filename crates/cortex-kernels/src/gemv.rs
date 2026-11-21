//! Multiply-free 1.58-bit ternary GEMV microkernels.
//!
//! Activations are symmetric INT8 in `[-127, +127]` (never `-128`, so negation
//! `0i8 - x` cannot overflow `i8`). Each [`CacheLine16x16`] holds 16 output rows
//! × 16 input columns (four [`Tile16x4`]s). Inside the K loop, 16-row `i16`
//! accumulators collect exact sums and promote losslessly into `i32` every
//! 8 cache-line blocks (128 columns; worst-case magnitude `128 * 127 = 16_256`
//! has 2× headroom under `i16::MAX = 32_767`).
//!
//! Hardware paths:
//! * **AVX2 (`x86_64` / `x86`)**: packs column pairs into 256-bit lanes, decodes
//!   2-bit trits via `_mm256_shuffle_epi8` + `_mm256_sign_epi8` (zero float or
//!   integer multiplications), accumulates in `i16x16`, widens to `i32` every
//!   128 columns.
//! * **NEON (`aarch64`)**: decodes 2-bit trits via `vqtbl1q_u8` LUT masks and
//!   uses dual-column `vaddl_s8` / `vaddl_high_s8` wide-adds into `i16x8` halves,
//!   widening to `i32x4` registers every 128 columns.
//! * **Scalar**: portable bit-exact reference.
//!
//! Dequantization is **not** done here; callers apply `x_scale * row_scale[r]`
//! once per output row at the end.

use crate::tile::{CacheLine16x16, Tile16x4};

#[inline]
fn accum_tile_scalar(tile: &Tile16x4, x: &[i8], base: usize, acc: &mut [i16; 16]) {
    for r in 0..16 {
        let byte = tile.data[r];
        let mut a = acc[r];
        for c in 0..4 {
            let code = (byte >> (2 * c)) & 0b11;
            let xv = i16::from(x[base + c]);
            match code {
                0b01 => a += xv,
                0b11 => a -= xv,
                _ => {}
            }
        }
        acc[r] = a;
    }
}

/// Portable reference GEMV. `y` is overwritten for every padded row.
///
/// `rows_pad` and `cols_pad` are multiples of 16. `x.len() == cols_pad`.
/// `blocks` is row-group major: `blocks[rg * ncb + cb]`.
pub fn gemv_bit158_scalar(
    blocks: &[CacheLine16x16],
    x: &[i8],
    y: &mut [i32],
    rows_pad: usize,
    cols_pad: usize,
) {
    assert!(rows_pad % 16 == 0 && cols_pad % 16 == 0, "padded shape");
    assert_eq!(x.len(), cols_pad);
    assert!(y.len() >= rows_pad);
    let nrg = rows_pad / 16;
    let ncb = cols_pad / 16;
    assert_eq!(blocks.len(), nrg * ncb);
    for rg in 0..nrg {
        let mut acc32 = [0i32; 16];
        let mut acc16 = [0i16; 16];
        for cb in 0..ncb {
            let block = &blocks[rg * ncb + cb];
            for t in 0..4 {
                let col = cb * 16 + t * 4;
                accum_tile_scalar(&block.tiles[t], x, col, &mut acc16);
            }
            // Promote every 8 cache lines (128 K) and on the tail.
            if (cb + 1) % 8 == 0 || cb + 1 == ncb {
                for r in 0..16 {
                    acc32[r] += i32::from(acc16[r]);
                    acc16[r] = 0;
                }
            }
        }
        let base = rg * 16;
        y[base..base + 16].copy_from_slice(&acc32);
    }
}

/// Dispatch: AVX2, then NEON, otherwise the scalar reference.
pub fn gemv_bit158(
    blocks: &[CacheLine16x16],
    x: &[i8],
    y: &mut [i32],
    rows_pad: usize,
    cols_pad: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { gemv_bit158_avx2(blocks, x, y, rows_pad, cols_pad) }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { gemv_bit158_neon(blocks, x, y, rows_pad, cols_pad) }
            return;
        }
    }
    gemv_bit158_scalar(blocks, x, y, rows_pad, cols_pad);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_bit158_avx2(
    blocks: &[CacheLine16x16],
    x: &[i8],
    y: &mut [i32],
    rows_pad: usize,
    cols_pad: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    assert!(rows_pad % 16 == 0 && cols_pad % 16 == 0);
    assert_eq!(x.len(), cols_pad);
    assert!(y.len() >= rows_pad);
    let nrg = rows_pad / 16;
    let ncb = cols_pad / 16;
    assert_eq!(blocks.len(), nrg * ncb);

    // 256-bit sign table for `_mm256_shuffle_epi8`: 00->0, 01->+1, 10->0, 11->-1.
    let sign_lut = _mm256_setr_epi8(
        0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1,
        0, 1, 0, -1,
    );
    let code_mask = _mm_set1_epi8(0x03);

    for rg in 0..nrg {
        let mut acc32_lo = _mm256_setzero_si256();
        let mut acc32_hi = _mm256_setzero_si256();
        let mut acc16 = _mm256_setzero_si256();

        for cb in 0..ncb {
            let block = &blocks[rg * ncb + cb];
            for t in 0..4 {
                let col = cb * 16 + t * 4;
                let tile = &block.tiles[t];
                let w = _mm_load_si128(tile.data.as_ptr() as *const __m128i);

                macro_rules! pair_cols {
                    ($s0:expr, $s1:expr, $c0:expr, $c1:expr) => {{
                        let codes0 = _mm_and_si128($s0, code_mask);
                        let codes1 = _mm_and_si128($s1, code_mask);
                        let codes01 = _mm256_set_m128i(codes1, codes0);
                        let signs01 = _mm256_shuffle_epi8(sign_lut, codes01);
                        let xv01 = _mm256_set_m128i(
                            _mm_set1_epi8(x[col + $c1]),
                            _mm_set1_epi8(x[col + $c0]),
                        );
                        let prod01 = _mm256_sign_epi8(xv01, signs01);
                        let p0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod01));
                        let p1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod01, 1));
                        acc16 = _mm256_add_epi16(acc16, _mm256_add_epi16(p0, p1));
                    }};
                }

                pair_cols!(w, _mm_srli_epi16(w, 2), 0, 1);
                pair_cols!(_mm_srli_epi16(w, 4), _mm_srli_epi16(w, 6), 2, 3);
            }

            if (cb + 1) % 8 == 0 || cb + 1 == ncb {
                let lo16 = _mm256_castsi256_si128(acc16);
                let hi16 = _mm256_extracti128_si256(acc16, 1);
                acc32_lo = _mm256_add_epi32(acc32_lo, _mm256_cvtepi16_epi32(lo16));
                acc32_hi = _mm256_add_epi32(acc32_hi, _mm256_cvtepi16_epi32(hi16));
                acc16 = _mm256_setzero_si256();
            }
        }

        let base = rg * 16;
        _mm256_storeu_si256(y[base..].as_mut_ptr() as *mut __m256i, acc32_lo);
        _mm256_storeu_si256(y[base + 8..].as_mut_ptr() as *mut __m256i, acc32_hi);
    }
}

/// NEON microkernel (`aarch64`): `vqtbl1q_u8` LUT mask lookup + dual-column
/// `vaddl_s8` / `vaddl_high_s8` wide-adds into `i16`, promoted to `i32` every
/// 8 cache-line blocks (128 columns). Zero multiplications in the inner loop.
///
/// # Safety
/// Caller must run on an `aarch64` target with `neon` enabled and pass padded
/// `rows_pad` / `cols_pad` multiples of 16 matching `blocks`, `x`, and `y`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_bit158_neon(
    blocks: &[CacheLine16x16],
    x: &[i8],
    y: &mut [i32],
    rows_pad: usize,
    cols_pad: usize,
) {
    use std::arch::aarch64::*;

    assert!(rows_pad % 16 == 0 && cols_pad % 16 == 0);
    assert_eq!(x.len(), cols_pad);
    assert!(y.len() >= rows_pad);
    let nrg = rows_pad / 16;
    let ncb = cols_pad / 16;
    assert_eq!(blocks.len(), nrg * ncb);

    // 16-byte LUTs for `vqtbl1q_u8`: code 01 -> 0xFF (add), code 11 -> 0xFF (sub).
    let mut add_tbl = vdupq_n_u8(0);
    add_tbl = vsetq_lane_u8::<1>(0xFF, add_tbl);
    let mut sub_tbl = vdupq_n_u8(0);
    sub_tbl = vsetq_lane_u8::<3>(0xFF, sub_tbl);
    let mask03 = vdupq_n_u8(0x03);

    for rg in 0..nrg {
        let mut acc32_0 = vdupq_n_s32(0);
        let mut acc32_1 = vdupq_n_s32(0);
        let mut acc32_2 = vdupq_n_s32(0);
        let mut acc32_3 = vdupq_n_s32(0);
        let mut acc_lo = vdupq_n_s16(0);
        let mut acc_hi = vdupq_n_s16(0);

        for cb in 0..ncb {
            let block = &blocks[rg * ncb + cb];
            for t in 0..4 {
                let col = cb * 16 + t * 4;
                let bytes = vld1q_u8(block.tiles[t].data.as_ptr());

                macro_rules! decode_col {
                    ($shifted:expr, $c:expr) => {{
                        let codes = vandq_u8($shifted, mask03);
                        let add_m = vreinterpretq_s8_u8(vqtbl1q_u8(add_tbl, codes));
                        let sub_m = vreinterpretq_s8_u8(vqtbl1q_u8(sub_tbl, codes));
                        let xv = vdupq_n_s8(x[col + $c]);
                        vsubq_s8(vandq_s8(xv, add_m), vandq_s8(xv, sub_m))
                    }};
                }

                let c0 = decode_col!(bytes, 0);
                let c1 = decode_col!(vshrq_n_u8::<2>(bytes), 1);
                acc_lo = vaddq_s16(acc_lo, vaddl_s8(vget_low_s8(c0), vget_low_s8(c1)));
                acc_hi = vaddq_s16(acc_hi, vaddl_high_s8(c0, c1));

                let c2 = decode_col!(vshrq_n_u8::<4>(bytes), 2);
                let c3 = decode_col!(vshrq_n_u8::<6>(bytes), 3);
                acc_lo = vaddq_s16(acc_lo, vaddl_s8(vget_low_s8(c2), vget_low_s8(c3)));
                acc_hi = vaddq_s16(acc_hi, vaddl_high_s8(c2, c3));
            }

            if (cb + 1) % 8 == 0 || cb + 1 == ncb {
                acc32_0 = vaddw_s16(acc32_0, vget_low_s16(acc_lo));
                acc32_1 = vaddw_high_s16(acc32_1, acc_lo);
                acc32_2 = vaddw_s16(acc32_2, vget_low_s16(acc_hi));
                acc32_3 = vaddw_high_s16(acc32_3, acc_hi);
                acc_lo = vdupq_n_s16(0);
                acc_hi = vdupq_n_s16(0);
            }
        }

        let base = rg * 16;
        vst1q_s32(y[base..].as_mut_ptr(), acc32_0);
        vst1q_s32(y[base + 4..].as_mut_ptr(), acc32_1);
        vst1q_s32(y[base + 8..].as_mut_ptr(), acc32_2);
        vst1q_s32(y[base + 12..].as_mut_ptr(), acc32_3);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn gemv_bit158_neon(
    blocks: &[CacheLine16x16],
    x: &[i8],
    y: &mut [i32],
    rows_pad: usize,
    cols_pad: usize,
) {
    gemv_bit158_scalar(blocks, x, y, rows_pad, cols_pad);
}

/// Feature-gated row-group split. Each group writes a disjoint `y` slice,
/// so there is no atomic reduction. Off unless `--features parallel`.
#[cfg(feature = "parallel")]
pub fn gemv_bit158_parallel(
    blocks: &[CacheLine16x16],
    x: &[i8],
    y: &mut [i32],
    rows_pad: usize,
    cols_pad: usize,
) {
    let nrg = rows_pad / 16;
    let ncb = cols_pad / 16;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(nrg.max(1));
    if threads <= 1 || nrg == 0 {
        gemv_bit158(blocks, x, y, rows_pad, cols_pad);
        return;
    }
    let chunk = nrg.div_ceil(threads);
    std::thread::scope(|scope| {
        for (tid, y_chunk) in y.chunks_mut(chunk * 16).enumerate() {
            let rg0 = tid * chunk;
            if rg0 >= nrg {
                break;
            }
            let rg1 = (rg0 + chunk).min(nrg);
            let block_slice = &blocks[rg0 * ncb..rg1 * ncb];
            let rows = (rg1 - rg0) * 16;
            scope.spawn(move || {
                gemv_bit158(block_slice, x, y_chunk, rows, cols_pad);
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::pack_ternary;

    fn pattern_matrix(rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
        let mut w = vec![0i8; rows * cols];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = match (i * 3 + 1) % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            };
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i * 17) % 50) as f32 - 25.0).collect();
        (w, x)
    }

    #[test]
    fn promotion_survives_past_i16() {
        // 17 cache-line blocks × 16 cols of +1, activations all 127.
        // 272 * 127 = 34544, which does not fit in i16. Promoting every
        // 8 blocks must yield the full i32 sum.
        let rows = 16usize;
        let cols = 272usize;
        let w = vec![1i8; rows * cols];
        let packed = pack_ternary(&w, rows, cols).unwrap();
        let x = vec![127i8; packed.cols_pad];
        let mut y = vec![0i32; packed.rows_pad];
        gemv_bit158_scalar(&packed.blocks, &x, &mut y, packed.rows_pad, packed.cols_pad);
        assert_eq!(y[0], 272 * 127);
        assert_eq!(y[15], 272 * 127);
        let mut y2 = vec![0i32; packed.rows_pad];
        gemv_bit158(
            &packed.blocks,
            &x,
            &mut y2,
            packed.rows_pad,
            packed.cols_pad,
        );
        assert_eq!(y, y2);
    }

    #[test]
    fn simd_matches_scalar_on_aligned_and_negative_extremes() {
        let (w, x) = pattern_matrix(48, 304);
        let packed = pack_ternary(&w, 48, 304).unwrap();
        let (q, _) = crate::bitlinear::quantize_activations_sym_i8(&x);
        let mut qpad = vec![0i8; packed.cols_pad];
        qpad[..q.len()].copy_from_slice(&q);
        let mut ys = vec![0i32; packed.rows_pad];
        let mut yd = vec![0i32; packed.rows_pad];
        gemv_bit158_scalar(
            &packed.blocks,
            &qpad,
            &mut ys,
            packed.rows_pad,
            packed.cols_pad,
        );
        gemv_bit158(
            &packed.blocks,
            &qpad,
            &mut yd,
            packed.rows_pad,
            packed.cols_pad,
        );
        assert_eq!(ys, yd);

        // Extreme -127 activations against -1 trits across 272 cols: (+34544).
        let w_neg = vec![-1i8; 16 * 272];
        let p_neg = pack_ternary(&w_neg, 16, 272).unwrap();
        let x_neg = vec![-127i8; p_neg.cols_pad];
        let mut y_neg = vec![0i32; p_neg.rows_pad];
        gemv_bit158(
            &p_neg.blocks,
            &x_neg,
            &mut y_neg,
            p_neg.rows_pad,
            p_neg.cols_pad,
        );
        assert!(y_neg.iter().all(|&v| v == 272 * 127));
    }
}
