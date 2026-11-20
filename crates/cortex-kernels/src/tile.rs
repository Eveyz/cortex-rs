//! 2-bit interleaved 1.58-bit tiles.
//!
//! # Tile16x4
//!
//! 16 output rows × 4 input columns, 16 bytes, `align(16)`.
//! `data[r]` packs the four trits of output row `r`:
//!
//! ```text
//! bits 7..6 = w[r, 3]
//! bits 5..4 = w[r, 2]
//! bits 3..2 = w[r, 1]
//! bits 1..0 = w[r, 0]
//! ```
//!
//! Encoding: `00 -> 0` (skip), `01 -> +1` (add), `11 -> -1` (subtract).
//! `10` is illegal and is never emitted by [`encode_trit`].
//!
//! # CacheLine16x16
//!
//! Four consecutive [`Tile16x4`] along the input axis (columns
//! `0..4`, `4..8`, `8..12`, `12..16`). The struct is `align(64)` and 64 bytes
//! wide, so one block occupies exactly one cache line and never splits.

/// One 16×4 packed ternary tile.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile16x4 {
    pub data: [u8; 16],
}

impl Tile16x4 {
    pub const fn zero() -> Self {
        Self { data: [0; 16] }
    }

    /// Trit at `(row, col)` with `col` in `0..4` and `row` in `0..16`.
    pub fn trit(&self, row: usize, col: usize) -> i8 {
        debug_assert!(row < 16 && col < 4);
        decode_code((self.data[row] >> (2 * col)) & 0b11)
    }
}

/// 16×16 packed block. Four tiles along the K (input) dimension.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheLine16x16 {
    pub tiles: [Tile16x4; 4],
}

impl CacheLine16x16 {
    pub const fn zero() -> Self {
        Self {
            tiles: [Tile16x4::zero(); 4],
        }
    }
}

const _: () = assert!(std::mem::size_of::<Tile16x4>() == 16);
const _: () = assert!(std::mem::align_of::<Tile16x4>() == 16);
const _: () = assert!(std::mem::size_of::<CacheLine16x16>() == 64);
const _: () = assert!(std::mem::align_of::<CacheLine16x16>() == 64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TritError {
    /// Value was not in `{-1, 0, +1}`.
    NotTernary(i8),
    /// Packed stream contained the reserved code `0b10`.
    ReservedCode,
}

/// `00 -> 0`, `01 -> +1`, `11 -> -1`. Reserved `10` decodes as 0 and is
/// reported by [`try_decode_code`].
pub fn decode_code(code: u8) -> i8 {
    match code & 0b11 {
        0b00 => 0,
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

pub fn try_decode_code(code: u8) -> Result<i8, TritError> {
    match code & 0b11 {
        0b00 => Ok(0),
        0b01 => Ok(1),
        0b11 => Ok(-1),
        _ => Err(TritError::ReservedCode),
    }
}

pub fn encode_trit(w: i8) -> Result<u8, TritError> {
    match w {
        0 => Ok(0b00),
        1 => Ok(0b01),
        -1 => Ok(0b11),
        other => Err(TritError::NotTernary(other)),
    }
}

/// Pack a row-major ternary matrix into cache-line blocks.
///
/// Logical shape may be any positive size. The physical shape is rounded up
/// to a multiple of 16. Padded rows and columns are zeros, so a GEMV of the
/// padded matrix matches the unpadded product when the extra activations are 0.
pub fn pack_ternary(weights: &[i8], rows: usize, cols: usize) -> Result<PackedMatrix, TritError> {
    if rows == 0 || cols == 0 {
        return Ok(PackedMatrix {
            rows,
            cols,
            rows_pad: 0,
            cols_pad: 0,
            blocks: Vec::new(),
        });
    }
    if weights.len() < rows * cols {
        return Err(TritError::NotTernary(0));
    }
    let rows_pad = rows.div_ceil(16) * 16;
    let cols_pad = cols.div_ceil(16) * 16;
    let nrg = rows_pad / 16;
    let ncb = cols_pad / 16;
    let mut blocks = vec![CacheLine16x16::zero(); nrg * ncb];
    for r in 0..rows {
        for c in 0..cols {
            let w = weights[r * cols + c];
            let code = encode_trit(w)?;
            let rg = r / 16;
            let cb = c / 16;
            let tile = (c % 16) / 4;
            let col_in = c % 4;
            let row_in = r % 16;
            let block = &mut blocks[rg * ncb + cb];
            block.tiles[tile].data[row_in] |= code << (2 * col_in);
        }
    }
    debug_assert_eq!(blocks.as_ptr() as usize % 64, 0);
    Ok(PackedMatrix {
        rows,
        cols,
        rows_pad,
        cols_pad,
        blocks,
    })
}

/// Unpack physical padded matrix back to a dense row-major `i8` buffer of
/// shape `rows_pad × cols_pad`. Logical cropping is the caller's job.
pub fn unpack_padded(blocks: &[CacheLine16x16], rows_pad: usize, cols_pad: usize) -> Vec<i8> {
    assert_eq!(rows_pad % 16, 0);
    assert_eq!(cols_pad % 16, 0);
    let ncb = cols_pad / 16;
    assert_eq!(blocks.len(), (rows_pad / 16) * ncb);
    let mut out = vec![0i8; rows_pad * cols_pad];
    for rg in 0..(rows_pad / 16) {
        for cb in 0..ncb {
            let block = &blocks[rg * ncb + cb];
            for tile in 0..4 {
                for row in 0..16 {
                    for col in 0..4 {
                        let r = rg * 16 + row;
                        let c = cb * 16 + tile * 4 + col;
                        out[r * cols_pad + c] = block.tiles[tile].trit(row, col);
                    }
                }
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct PackedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub rows_pad: usize,
    pub cols_pad: usize,
    pub blocks: Vec<CacheLine16x16>,
}

/// Non-zero trit count over the logical (unpadded) matrix, for density checks.
pub fn logical_density(weights: &[i8], rows: usize, cols: usize) -> (usize, usize) {
    let mut nz = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            if weights[r * cols + c] != 0 {
                nz += 1;
            }
        }
    }
    (nz, rows * cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip_and_density() {
        let rows = 37;
        let cols = 20;
        let mut w = vec![0i8; rows * cols];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = match i % 5 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => 1,
                _ => 0,
            };
        }
        let packed = pack_ternary(&w, rows, cols).unwrap();
        assert_eq!(packed.blocks.as_ptr() as usize % 64, 0);
        assert_eq!(packed.rows_pad, 48);
        assert_eq!(packed.cols_pad, 32);
        let dense = unpack_padded(&packed.blocks, packed.rows_pad, packed.cols_pad);
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(dense[r * packed.cols_pad + c], w[r * cols + c]);
            }
        }
        for r in 0..packed.rows_pad {
            for c in 0..packed.cols_pad {
                if r >= rows || c >= cols {
                    assert_eq!(dense[r * packed.cols_pad + c], 0);
                }
            }
        }
        let (nz, total) = logical_density(&w, rows, cols);
        assert_eq!(total, rows * cols);
        assert!(nz > 0 && nz < total);
        // Reserved code 0b10 never appears.
        for block in &packed.blocks {
            for tile in &block.tiles {
                for byte in tile.data {
                    for shift in [0, 2, 4, 6] {
                        let code = (byte >> shift) & 0b11;
                        assert_ne!(code, 0b10);
                        assert!(try_decode_code(code).is_ok());
                    }
                }
            }
        }
    }

    #[test]
    fn encode_rejects_non_ternary() {
        assert_eq!(encode_trit(0).unwrap(), 0b00);
        assert_eq!(encode_trit(1).unwrap(), 0b01);
        assert_eq!(encode_trit(-1).unwrap(), 0b11);
        assert_eq!(encode_trit(2), Err(TritError::NotTernary(2)));
        assert_eq!(try_decode_code(0b10), Err(TritError::ReservedCode));
    }

    #[test]
    fn bit_positions_match_contract() {
        // Single +1 at (row=3, col=2) inside the first tile => bits 5..4 of data[3].
        let mut w = vec![0i8; 16 * 16];
        w[3 * 16 + 2] = 1;
        w[3 * 16 + 0] = -1;
        let packed = pack_ternary(&w, 16, 16).unwrap();
        let byte = packed.blocks[0].tiles[0].data[3];
        assert_eq!((byte >> 4) & 0b11, 0b01, "w[3,2] lives in bits 5..4");
        assert_eq!(byte & 0b11, 0b11, "w[3,0] lives in bits 1..0");
    }
}
