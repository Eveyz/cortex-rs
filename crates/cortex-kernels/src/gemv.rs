//! Multiply-free ternary GEMV.
//!
//! Activations are symmetric INT8 in `[-127, +127]` (never `-128`, so
//! `0i8 - x` cannot overflow `i8`). The accumulator is `i16` inside a
//! cache-line tile and is promoted into `i32` after every 8 cache-line
//! blocks (128 columns). Worst case inside one bucket is `128 * 127 = 16256`,
//! about 2× headroom under `i16::MAX`.
//!
//! Dequantization is **not** done here. Callers scale the `i32` row sum once.

use crate::tile::CacheLine16x16;

/// Compile-time sign LUTs for `vpshufb`. Index is the 2-bit code.
/// `01 -> 0xFF` (add), `11 -> 0xFF` (subtract on the other table). `10` stays 0.
#[repr(C, align(16))]
struct AlignedLut([u8; 16]);

static ADD_LUT: AlignedLut =
    AlignedLut([0x00, 0xFF, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
static SUB_LUT: AlignedLut =
    AlignedLut([0x00, 0x00, 0x00, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

#[inline]
fn sat_add(a: i16, b: i16) -> i16 {
    a.saturating_add(b)
}

#[inline]
fn sat_sub(a: i16, b: i16) -> i16 {
    a.saturating_sub(b)
}

fn accum_tile_scalar(tile: &crate::tile::Tile16x4, x: &[i8], base: usize, acc: &mut [i16; 16]) {
    for r in 0..16 {
        let byte = tile.data[r];
        let mut a = acc[r];
        for c in 0..4 {
            let code = (byte >> (2 * c)) & 0b11;
            let xv = x[base + c] as i16;
            match code {
                0b01 => a = sat_add(a, xv),
                0b11 => a = sat_sub(a, xv),
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
    gemv_with(blocks, x, y, rows_pad, cols_pad, accum_tile_scalar);
}

fn gemv_with(
    blocks: &[CacheLine16x16],
    x: &[i8],
    y: &mut [i32],
    rows_pad: usize,
    cols_pad: usize,
    mut tile_fn: impl FnMut(&crate::tile::Tile16x4, &[i8], usize, &mut [i16; 16]),
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
                tile_fn(&block.tiles[t], x, col, &mut acc16);
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
        for r in 0..16 {
            y[base + r] = acc32[r];
        }
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
#[target_feature(enable = "avx2,ssse3")]
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

    let add_lut = _mm_load_si128(ADD_LUT.0.as_ptr() as *const __m128i);
    let sub_lut = _mm_load_si128(SUB_LUT.0.as_ptr() as *const __m128i);
    let code_mask = _mm_set1_epi8(0x03);

    for rg in 0..nrg {
        let mut acc32 = [0i32; 16];
        let mut acc16 = [0i16; 16];
        for cb in 0..ncb {
            let block = &blocks[rg * ncb + cb];
            for t in 0..4 {
                let col = cb * 16 + t * 4;
                let tile = &block.tiles[t];
                // vmovdqa — tile is align(16).
                let w = _mm_load_si128(tile.data.as_ptr() as *const __m128i);
                let mut accv = _mm256_loadu_si256(acc16.as_ptr() as *const __m256i);
                // Unrolled: srli immediate must be a constant. Shifts 0/2/4/6
                // extract codes for columns 0..3. Masking with 0x03 after an
                // epi16 shift drops the cross-byte contamination.
                macro_rules! col {
                    ($shifted:expr, $idx:expr) => {{
                        let codes = _mm_and_si128($shifted, code_mask);
                        let add_m = _mm_shuffle_epi8(add_lut, codes);
                        let sub_m = _mm_shuffle_epi8(sub_lut, codes);
                        let xv = i16::from(x[col + $idx]);
                        let xvec = _mm256_set1_epi16(xv);
                        let add16 = _mm256_cvtepi8_epi16(add_m);
                        let sub16 = _mm256_cvtepi8_epi16(sub_m);
                        accv = _mm256_adds_epi16(accv, _mm256_and_si256(xvec, add16));
                        accv = _mm256_subs_epi16(accv, _mm256_and_si256(xvec, sub16));
                    }};
                }
                col!(w, 0);
                col!(_mm_srli_epi16(w, 2), 1);
                col!(_mm_srli_epi16(w, 4), 2);
                col!(_mm_srli_epi16(w, 6), 3);
                _mm256_storeu_si256(acc16.as_mut_ptr() as *mut __m256i, accv);
            }
            if (cb + 1) % 8 == 0 || cb + 1 == ncb {
                for r in 0..16 {
                    acc32[r] += i32::from(acc16[r]);
                    acc16[r] = 0;
                }
            }
        }
        let base = rg * 16;
        for r in 0..16 {
            y[base + r] = acc32[r];
        }
    }
}

/// NEON microkernel. Same numerical contract as the scalar path:
/// per-column `vsubq_s8(pos, neg)` contribution, widened with
/// `vaddl_s8` / `vaddl_high_s8` into the `i16` bucket.
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

    for rg in 0..nrg {
        let mut acc32 = [0i32; 16];
        // Two i16x8 halves stand in for the 16-lane bucket.
        let mut acc_lo = vdupq_n_s16(0);
        let mut acc_hi = vdupq_n_s16(0);
        for cb in 0..ncb {
            let block = &blocks[rg * ncb + cb];
            for t in 0..4 {
                let col = cb * 16 + t * 4;
                let bytes = vld1q_u8(block.tiles[t].data.as_ptr());
                for c in 0..4u32 {
                    let shifted = if c == 0 {
                        bytes
                    } else {
                        vshrq_n_u8(bytes, 2 * c)
                    };
                    let codes = vandq_u8(shifted, vdupq_n_u8(0x03));
                    let add_m = vceqq_u8(codes, vdupq_n_u8(0x01));
                    let sub_m = vceqq_u8(codes, vdupq_n_u8(0x03));
                    let xv = vdupq_n_s8(x[col + c as usize]);
                    let pos = vandq_s8(xv, vreinterpretq_s8_u8(add_m));
                    let neg = vandq_s8(xv, vreinterpretq_s8_u8(sub_m));
                    let contrib = vsubq_s8(pos, neg);
                    let zero = vdupq_n_s8(0);
                    let wide_lo = vaddl_s8(vget_low_s8(contrib), vget_low_s8(zero));
                    let wide_hi = vaddl_high_s8(contrib, zero);
                    acc_lo = vaddq_s16(acc_lo, wide_lo);
                    acc_hi = vaddq_s16(acc_hi, wide_hi);
                }
            }
            if (cb + 1) % 8 == 0 || cb + 1 == ncb {
                let mut tmp_lo = [0i16; 8];
                let mut tmp_hi = [0i16; 8];
                vst1q_s16(tmp_lo.as_mut_ptr(), acc_lo);
                vst1q_s16(tmp_hi.as_mut_ptr(), acc_hi);
                for r in 0..8 {
                    acc32[r] += i32::from(tmp_lo[r]);
                    acc32[r + 8] += i32::from(tmp_hi[r]);
                }
                acc_lo = vdupq_n_s16(0);
                acc_hi = vdupq_n_s16(0);
            }
        }
        let base = rg * 16;
        for r in 0..16 {
            y[base + r] = acc32[r];
        }
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
    // Host has no NEON. Keep the symbol so callers can name it; it is the
    // scalar contract, not a silent different kernel.
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
    fn simd_matches_scalar_on_aligned_block() {
        let (w, x) = pattern_matrix(32, 64);
        let packed = pack_ternary(&w, 32, 64).unwrap();
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
    }
}
