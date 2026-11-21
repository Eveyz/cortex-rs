//! `.ctxb` compact 64-byte-aligned binary model format (v1).
//!
//! ```text
//! Header (64 bytes, little-endian):
//!   0..4   magic b"CTXB"
//!   4..8   u32 version = 1
//!   8..40  u32 n_macros, d_model, n_heads, head_dim, vocab, window, ffn, hip
//!   40..44 f32 rms_eps
//!   44..48 u32 flags (bit 0 = 1: hip_k, hip_v, and final_norm present; 0 = legacy)
//!   48..64 reserved zero bytes
//! ```
//!
//! Payload:
//! * `emb_i8`: `i8[vocab * d_model]` (`[-127, +127]`, weight-tied with LM head)
//! * `emb_scale`: `f32[vocab]`
//! * Per macro (`0..n_macros`): 3 × [`BitSsmBlock`] + 1 × [`MemoryBlock`]
//! * If `flags & 1 != 0`: `final_norm`: `f32[d_model]`
//!
//! Every [`BitLinearWeights`] block aligns its packed [`CacheLine16x16`] array
//! to a 64-byte file boundary. Last 4 bytes of the file are IEEE CRC32.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use cortex_hippocampus::{AsyncHippocampusHandle, HippocampusWeights, PlasticityConfig};
use cortex_kernels::{crc32_ieee, BitLinearWeights, CacheLine16x16};
use cortex_ssm::{BitSsmBlock, BitSwiGLU, RmsNorm};

use crate::model::{CortexConfig, CortexModel, MemoryBlock};

pub const CTXB_FLAG_FULL_MEM_FINAL_NORM: u32 = 1;

pub fn crc32(data: &[u8]) -> u32 {
    crc32_ieee(data)
}

pub fn save_ctxb(model: &CortexModel, path: &Path) -> Result<(), String> {
    let bytes = to_ctxb(model);
    let mut f = File::create(path).map_err(|e| e.to_string())?;
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn load_ctxb(path: &Path) -> Result<CortexModel, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    from_ctxb(&buf)
}

pub fn to_ctxb(model: &CortexModel) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"CTXB");
    put_u32(&mut buf, 1);
    let c = &model.cfg;
    put_u32(&mut buf, c.n_macros as u32);
    put_u32(&mut buf, c.d_model as u32);
    put_u32(&mut buf, c.n_heads as u32);
    put_u32(&mut buf, c.head_dim as u32);
    put_u32(&mut buf, c.vocab as u32);
    put_u32(&mut buf, c.window as u32);
    put_u32(&mut buf, c.ffn_hidden as u32);
    put_u32(&mut buf, c.hip_hidden as u32);
    put_f32(&mut buf, c.rms_eps);
    put_u32(&mut buf, CTXB_FLAG_FULL_MEM_FINAL_NORM);
    while buf.len() < 64 {
        buf.push(0);
    }
    let emb_bytes: Vec<u8> = model.emb_i8.iter().map(|v| *v as u8).collect();
    buf.extend_from_slice(&emb_bytes);
    put_f32s(&mut buf, model.emb_scale.as_slice());
    for m in 0..c.n_macros {
        for s in 0..3 {
            write_ssm(&mut buf, &model.ssm[m * 3 + s]);
        }
        write_mem(&mut buf, &model.mem[m]);
    }
    put_f32s(&mut buf, model.final_norm.weight.as_slice());
    let sum = crc32(&buf);
    put_u32(&mut buf, sum);
    buf
}

pub fn from_ctxb(bytes: &[u8]) -> Result<CortexModel, String> {
    if bytes.len() < 68 || &bytes[0..4] != b"CTXB" {
        return Err("bad CTXB magic".into());
    }
    let crc_stored = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    let body = &bytes[..bytes.len() - 4];
    let sum = crc32(body);
    if sum != crc_stored {
        return Err(format!("CTXB crc mismatch {sum:#x} != {crc_stored:#x}"));
    }
    let mut cur = Cursor { buf: body, o: 4 };
    let version = cur.u32()?;
    if version != 1 {
        return Err(format!("unsupported CTXB version {version}"));
    }
    let cfg = CortexConfig {
        n_macros: cur.u32()? as usize,
        d_model: cur.u32()? as usize,
        n_heads: cur.u32()? as usize,
        head_dim: cur.u32()? as usize,
        vocab: cur.u32()? as usize,
        window: cur.u32()? as usize,
        ffn_hidden: cur.u32()? as usize,
        hip_hidden: cur.u32()? as usize,
        rms_eps: cur.f32()?,
    };
    let flags = cur.u32()?;
    let full_ext = (flags & CTXB_FLAG_FULL_MEM_FINAL_NORM) != 0;
    cur.align(64)?;
    cfg.validate()?;
    let emb_i8 = cur.i8s(cfg.vocab * cfg.d_model)?;
    let scales = cur.f32s(cfg.vocab)?;
    let mut emb_scale = cortex_kernels::Aligned64::zeros(cfg.vocab);
    emb_scale.as_mut_slice().copy_from_slice(&scales);
    let mut ssm = Vec::with_capacity(cfg.n_macros * 3);
    let mut mem = Vec::with_capacity(cfg.n_macros);
    for _ in 0..cfg.n_macros {
        for _ in 0..3 {
            ssm.push(read_ssm(&mut cur, &cfg)?);
        }
        mem.push(read_mem(&mut cur, &cfg, full_ext)?);
    }
    let mut final_norm = RmsNorm::ones(cfg.d_model, cfg.rms_eps);
    if full_ext {
        fill_f32(&mut final_norm, &cur.f32s(cfg.d_model)?)?;
    }
    if cur.o != body.len() {
        return Err(format!(
            "CTXB trailing bytes: cursor {} body {}",
            cur.o,
            body.len()
        ));
    }
    Ok(CortexModel {
        cfg,
        emb_i8,
        emb_scale,
        ssm,
        mem,
        final_norm,
    })
}

fn write_ssm(buf: &mut Vec<u8>, b: &BitSsmBlock) {
    put_f32s(buf, b.norm1.weight.as_slice());
    put_f32s(buf, b.norm2.weight.as_slice());
    put_f32s(buf, b.alpha_w.as_slice());
    put_f32s(buf, b.alpha_b.as_slice());
    put_f32s(buf, b.beta_w.as_slice());
    put_f32s(buf, b.beta_b.as_slice());
    write_bit(buf, &b.q_proj);
    write_bit(buf, &b.k_proj);
    write_bit(buf, &b.v_proj);
    write_bit(buf, &b.o_proj);
    write_swiglu(buf, &b.ffn);
}

fn read_ssm(cur: &mut Cursor, cfg: &CortexConfig) -> Result<BitSsmBlock, String> {
    let mut block = BitSsmBlock::zeros(
        cfg.d_model,
        cfg.n_heads,
        cfg.head_dim,
        cfg.ffn_hidden,
        cfg.rms_eps,
    );
    fill_f32(&mut block.norm1, &cur.f32s(cfg.d_model)?)?;
    fill_f32(&mut block.norm2, &cur.f32s(cfg.d_model)?)?;
    copy_aligned(&mut block.alpha_w, &cur.f32s(cfg.n_heads * cfg.d_model)?)?;
    copy_aligned(&mut block.alpha_b, &cur.f32s(cfg.n_heads)?)?;
    copy_aligned(&mut block.beta_w, &cur.f32s(cfg.n_heads * cfg.d_model)?)?;
    copy_aligned(&mut block.beta_b, &cur.f32s(cfg.n_heads)?)?;
    block.q_proj = read_bit(cur)?;
    block.k_proj = read_bit(cur)?;
    block.v_proj = read_bit(cur)?;
    block.o_proj = read_bit(cur)?;
    block.ffn = read_swiglu(cur)?;
    Ok(block)
}

fn write_mem(buf: &mut Vec<u8>, b: &MemoryBlock) {
    put_f32s(buf, b.norm1.weight.as_slice());
    put_f32s(buf, b.norm2.weight.as_slice());
    put_f32s(buf, b.gate_w.as_slice());
    put_f32s(buf, b.gate_b.as_slice());
    write_bit(buf, &b.attn.w_q);
    write_bit(buf, &b.attn.w_k);
    write_bit(buf, &b.attn.w_v);
    write_bit(buf, &b.attn.w_o);
    write_bit(buf, &b.hip_proj);
    write_bit(buf, &b.hip_k);
    write_bit(buf, &b.hip_v);
    let snap = b.hip.snapshot();
    put_u32(buf, snap.d_in as u32);
    put_u32(buf, snap.d_hidden as u32);
    put_u32(buf, snap.d_out as u32);
    put_u64(buf, snap.version);
    put_f32s(buf, snap.w1.as_slice());
    put_f32s(buf, snap.w2.as_slice());
    write_swiglu(buf, &b.ffn);
}

fn read_mem(cur: &mut Cursor, cfg: &CortexConfig, full_ext: bool) -> Result<MemoryBlock, String> {
    let mut block = MemoryBlock::zeros(cfg);
    fill_f32(&mut block.norm1, &cur.f32s(cfg.d_model)?)?;
    fill_f32(&mut block.norm2, &cur.f32s(cfg.d_model)?)?;
    copy_aligned(&mut block.gate_w, &cur.f32s(cfg.d_model)?)?;
    copy_aligned(&mut block.gate_b, &cur.f32s(cfg.d_model)?)?;
    block.attn.w_q = read_bit(cur)?;
    block.attn.w_k = read_bit(cur)?;
    block.attn.w_v = read_bit(cur)?;
    block.attn.w_o = read_bit(cur)?;
    block.hip_proj = read_bit(cur)?;
    if full_ext {
        block.hip_k = read_bit(cur)?;
        block.hip_v = read_bit(cur)?;
    }
    let d_in = cur.u32()? as usize;
    let d_h = cur.u32()? as usize;
    let d_out = cur.u32()? as usize;
    let ver = cur.u64()?;
    let w1 = cur.f32s(d_in * d_h)?;
    let w2 = cur.f32s(d_out * d_h)?;
    let mut hip = HippocampusWeights::new(d_in, d_h, d_out);
    hip.w1.as_mut_slice().copy_from_slice(&w1);
    hip.w2.as_mut_slice().copy_from_slice(&w2);
    hip.version = ver;
    block.hip = AsyncHippocampusHandle::spawn(hip, PlasticityConfig::default(), 64);
    block.ffn = read_swiglu(cur)?;
    Ok(block)
}

fn write_swiglu(buf: &mut Vec<u8>, ffn: &BitSwiGLU) {
    write_bit(buf, &ffn.w_gate);
    write_bit(buf, &ffn.w_up);
    write_bit(buf, &ffn.w_down);
}

fn read_swiglu(cur: &mut Cursor) -> Result<BitSwiGLU, String> {
    Ok(BitSwiGLU {
        w_gate: read_bit(cur)?,
        w_up: read_bit(cur)?,
        w_down: read_bit(cur)?,
    })
}

fn write_bit(buf: &mut Vec<u8>, w: &BitLinearWeights) {
    put_u32(buf, w.rows as u32);
    put_u32(buf, w.cols as u32);
    put_u32(buf, w.rows_pad as u32);
    put_u32(buf, w.cols_pad as u32);
    pad64(buf);
    let nbytes = w.blocks.len() * 64;
    let start = buf.len();
    buf.resize(start + nbytes, 0);
    if nbytes > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                w.blocks.as_ptr() as *const u8,
                buf[start..].as_mut_ptr(),
                nbytes,
            );
        }
    }
    put_f32s(buf, w.row_scales.as_slice());
}

fn read_bit(cur: &mut Cursor) -> Result<BitLinearWeights, String> {
    let rows = cur.u32()? as usize;
    let cols = cur.u32()? as usize;
    let rows_pad = cur.u32()? as usize;
    let cols_pad = cur.u32()? as usize;
    cur.align(64)?;
    let nblocks = if rows_pad == 0 || cols_pad == 0 {
        0
    } else {
        (rows_pad / 16) * (cols_pad / 16)
    };
    let raw = cur.bytes(nblocks * 64)?.to_vec();
    let mut blocks = vec![CacheLine16x16::zero(); nblocks];
    if nblocks > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(raw.as_ptr(), blocks.as_mut_ptr() as *mut u8, raw.len());
        }
    }
    debug_assert!(blocks.as_ptr() as usize % 64 == 0 || nblocks == 0);
    let scales = cur.f32s(rows)?;
    BitLinearWeights::from_packed(rows, cols, rows_pad, cols_pad, blocks, &scales)
        .map_err(|e| format!("packed bitlinear: {e:?}"))
}

fn fill_f32(norm: &mut RmsNorm, src: &[f32]) -> Result<(), String> {
    if norm.weight.len() != src.len() {
        return Err("rmsnorm width".into());
    }
    norm.weight.as_mut_slice().copy_from_slice(src);
    Ok(())
}

fn copy_aligned(dst: &mut cortex_kernels::Aligned64<f32>, src: &[f32]) -> Result<(), String> {
    if dst.len() != src.len() {
        return Err("aligned width".into());
    }
    dst.as_mut_slice().copy_from_slice(src);
    Ok(())
}

struct Cursor<'a> {
    buf: &'a [u8],
    o: usize,
}

impl<'a> Cursor<'a> {
    fn need(&self, n: usize) -> Result<(), String> {
        if self.o + n > self.buf.len() {
            Err(format!("truncated ctxb at {} need {n}", self.o))
        } else {
            Ok(())
        }
    }
    fn u32(&mut self) -> Result<u32, String> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.buf[self.o..self.o + 4].try_into().unwrap());
        self.o += 4;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64, String> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.buf[self.o..self.o + 8].try_into().unwrap());
        self.o += 8;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.f32()?);
        }
        Ok(out)
    }
    fn i8s(&mut self, n: usize) -> Result<Vec<i8>, String> {
        self.need(n)?;
        let v = self.buf[self.o..self.o + n]
            .iter()
            .map(|b| *b as i8)
            .collect();
        self.o += n;
        Ok(v)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        self.need(n)?;
        let s = &self.buf[self.o..self.o + n];
        self.o += n;
        Ok(s)
    }
    fn align(&mut self, align: usize) -> Result<(), String> {
        let mis = self.o % align;
        if mis != 0 {
            let skip = align - mis;
            self.need(skip)?;
            self.o += skip;
        }
        Ok(())
    }
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(buf: &mut Vec<u8>, v: f32) {
    put_u32(buf, v.to_bits());
}
fn put_f32s(buf: &mut Vec<u8>, xs: &[f32]) {
    for x in xs {
        put_f32(buf, *x);
    }
}
fn pad64(buf: &mut Vec<u8>) {
    while buf.len() % 64 != 0 {
        buf.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_ieee_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn ctxb_roundtrip_logits_match_and_corrupt_crc_fails() {
        let model = CortexModel::from_seed(CortexConfig::mini(), 5);
        let bytes = to_ctxb(&model);
        assert_eq!(&bytes[0..4], b"CTXB");
        let loaded = from_ctxb(&bytes).unwrap();
        let mut s1 = model.new_state();
        let mut s2 = loaded.new_state();
        let a = model.step(4, &mut s1);
        let b = loaded.step(4, &mut s2);
        let err = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-5, "roundtrip logit err {err}");
        assert_eq!(loaded.ssm[0].q_proj.blocks.as_ptr() as usize % 64, 0);

        let mut bad = bytes.clone();
        bad[64] ^= 0xFF;
        assert!(from_ctxb(&bad).is_err());
    }
}
