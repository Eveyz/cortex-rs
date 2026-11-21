//! Pure-Rust 1.58-bit microkernels and shared low-level primitives.
//!
//! No C compiler, no `cc` build script. ISA paths are `core::arch` intrinsics
//! gated by runtime feature detection (`AVX2`) or `cfg(target_arch)` (`NEON`).
//! The scalar GEMV is the numerical reference and is always available.
//!
//! # Layout
//!
//! * [`Tile16x4`] — 16 bytes, 16 rows × 4 cols, bits packed low-to-high.
//! * [`CacheLine16x16`] — 4 tiles, 64 bytes, `align(64)`.
//! * Activations: symmetric INT8 in `[-127, 127]`.
//! * `i16` partials promote into `i32` every 8 cache-line blocks (128 columns).

#![allow(clippy::needless_range_loop)]

mod align;
mod bitlinear;
mod gemv;
mod tile;

pub use align::Aligned64;
pub use bitlinear::{
    quantize_activations_sym_i8, quantize_with_scale, round_half_away, BitLinearWeights,
};
pub use gemv::{gemv_bit158, gemv_bit158_neon, gemv_bit158_scalar};
pub use tile::{
    decode_code, encode_trit, logical_density, pack_ternary, try_decode_code, unpack_padded,
    CacheLine16x16, PackedMatrix, Tile16x4, TritError,
};

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use gemv::gemv_bit158_avx2;

/// SiLU / swish: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Analytic SiLU derivative: `σ(x) * (1 + x * (1 - σ(x)))`.
#[inline]
pub fn silu_prime(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// Numerically stable logistic. Range is `(0, 1)`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Numerically stable softplus: `ln(1 + exp(x))`.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// In-place L2 normalization onto the unit sphere (`eps = 1e-8`).
/// Zero / near-zero vectors stay untouched rather than producing NaN.
#[inline]
pub fn l2_normalize_inplace(v: &mut [f32]) {
    let ss: f32 = v.iter().map(|&x| x * x).sum();
    let n = ss.sqrt();
    if n > 1e-8 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Compile-time IEEE CRC32 lookup table (poly `0xEDB88320`).
const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = build_crc32_table();

/// Fast table-driven IEEE CRC32 (`crc32(b"123456789") == 0xCBF4_3926`).
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
        crc = CRC32_TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

/// Which GEMV implementation `gemv_bit158` will call on this process.
pub fn active_isa() -> &'static str {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return "neon";
        }
    }
    "scalar"
}
