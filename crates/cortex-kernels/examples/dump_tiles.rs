//! Dev tool: read a ternary matrix (`rows*cols` ints followed by `rows cols`)
//! from stdin and emit raw `CacheLine16x16` bytes as hex. Used by
//! `tools/test_ctxb_packing.py` to verify byte-level parity with Python.

use cortex_kernels::pack_ternary;
use std::io::Read;

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let mut nums: Vec<i64> = text
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let cols = nums.pop().unwrap() as usize;
    let rows = nums.pop().unwrap() as usize;
    let w: Vec<i8> = nums.into_iter().map(|v| v as i8).collect();
    assert_eq!(w.len(), rows * cols);
    let packed = pack_ternary(&w, rows, cols).expect("valid ternary matrix");
    for block in &packed.blocks {
        for tile in &block.tiles {
            for b in tile.data {
                print!("{b:02x}");
            }
        }
    }
    println!();
}
