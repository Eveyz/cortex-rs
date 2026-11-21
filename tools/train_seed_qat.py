#!/usr/bin/env python3
"""Offline QAT seed trainer & byte-exact `.ctxb` exporter for Cortex-Local.

Implements the complete PyTorch training counterpart of `crates/cortex-runtime`:

1. `BitLinearQAT` — per-row abs-mean ternary weight quantization (`{-1, 0, +1}`)
   and per-token symmetric INT8 activation quantization (`[-127, +127]`, never
   `-128`) using `round_half_away` (matching Rust `f32::round`) and straight-
   through estimators (`w + (w_hat - w).detach()`).
2. `SsmBlock` — O(d_head^2) Gated DeltaNet (`u = S k`,
   `S = a S + b (v - a u) k^T`, `o = S q`) with `alpha_b = 4.6` (`a ≈ 0.99`),
   plus 1.58-bit SwiGLU.
3. `MemBlock` — Branch A per-head INT8 sliding-window causal attention
   (`0 <= t - s < window`, `q` in FP32, `k/v` quantized per head) + Branch B
   neural hippocampus (`m = W2 SiLU(W1 L2Norm(q))`). Under `--joint-hippocampus`
   (`hippo_train=True`), the first half of each sequence writes `(L2Norm(k), v)`
   into fast weights via the exact analytic outer-product rule (differentiable
   w.r.t. `hip_k` and `hip_v` projections!) and the second half reads via
   `hip_proj`, training all three read/write interface projections jointly.
4. `CortexSeedModel` (`CortexModelTorch`) — weight-tied symmetric INT8
   embedding / LM head + `final_norm` + `CTXB` (`flags = 1`) exporter packing
   `CacheLine16x16` / `Tile16x4` blocks at 64-byte file offsets with IEEE CRC32.
5. `--self-check` — trains `CortexSeedModel` with QAT, verifies non-zero grads
   across `hip_proj` / `hip_k` / `hip_v`, exports `.ctxb`, runs `cortex-cli`
   (`logits` and `--check-ctxb`), and requires `max |PyTorch - Rust| < 1e-3`.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    print("PyTorch is required for tools/train_seed_qat.py:", exc, file=sys.stderr)
    sys.exit(1)

CTXB_FLAG_FULL_MEM_FINAL_NORM = 1


# ---------------------------------------------------------------------------
# Rounding & tile packing (AST-extractable without torch for test_ctxb_packing)
# ---------------------------------------------------------------------------


def round_half_away_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def pack_ternary(trit: np.ndarray) -> tuple[int, int, int, int, bytes]:
    """Pack a `[rows, cols]` ternary matrix (`{-1, 0, +1}`) into `CacheLine16x16` bytes.

    Matches `cortex_kernels::pack_ternary` byte-for-byte:
    - Physical shape padded up to multiples of 16 (`rows_pad`, `cols_pad`).
    - Block `(rg, cb)` is 64 bytes (`4 × Tile16x4`).
    - `tile = (c % 16) // 4`, `row_in = r % 16`, `col_in = c % 4`.
    - `0 -> 0b00`, `+1 -> 0b01`, `-1 -> 0b11` shifted by `2 * col_in`.
    """
    rows, cols = trit.shape
    if rows == 0 or cols == 0:
        return rows, cols, 0, 0, b""
    rows_pad = (rows + 15) // 16 * 16
    cols_pad = (cols + 15) // 16 * 16
    nrg = rows_pad // 16
    ncb = cols_pad // 16
    blocks = bytearray(nrg * ncb * 64)
    for r in range(rows):
        for c in range(cols):
            w = int(trit[r, c])
            if w == 0:
                code = 0
            elif w == 1:
                code = 0b01
            elif w == -1:
                code = 0b11
            else:
                raise ValueError(f"non-ternary weight {w} at ({r}, {c})")
            rg = r // 16
            cb = c // 16
            tile = (c % 16) // 4
            col_in = c % 4
            row_in = r % 16
            off = (rg * ncb + cb) * 64 + tile * 16 + row_in
            blocks[off] |= code << (2 * col_in)
    return rows, cols, rows_pad, cols_pad, bytes(blocks)


def pack_tiles(trit: np.ndarray) -> bytes:
    """Return raw `CacheLine16x16` bytes for `trit` (used by `test_ctxb_packing.py`)."""
    return pack_ternary(trit)[4]


# ---------------------------------------------------------------------------
# PyTorch STE quantization helpers (matching Rust f32::round & [-127, +127])
# ---------------------------------------------------------------------------


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def quantize_act_ste(x: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 activation quantization along `dim=-1` in `[-127, +127]`.

    Returns `(q_int_float, scale)` in eval mode, or `(x_ste_dequant, scale)` helper.
    """
    amax = x.abs().amax(dim=-1, keepdim=True)
    zero_mask = amax <= 0.0
    scale = torch.where(zero_mask, torch.ones_like(amax), amax / 127.0)
    q = torch.where(
        zero_mask,
        torch.zeros_like(x),
        round_half_away(x / scale).clamp(-127.0, 127.0),
    )
    if training:
        x_hat = q * scale
        return x + (x_hat - x).detach(), scale
    return q, scale


def l2_norm_last(x: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return torch.where(n > 1e-8, x / n.clamp_min(1e-8), x)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * weight


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    n_macros: int = 2
    d_model: int = 256
    n_heads: int = 4
    head_dim: int = 64
    vocab_size: int = 266
    window_size: int = 512
    ffn_hidden_dim: int = 512
    hippo_hidden_dim: int = 512
    rms_eps: float = 1e-6

    @classmethod
    def mini(cls) -> "ModelConfig":
        return cls(
            n_macros=1,
            d_model=16,
            n_heads=2,
            head_dim=8,
            vocab_size=32,
            window_size=8,
            ffn_hidden_dim=16,
            hippo_hidden_dim=8,
            rms_eps=1e-6,
        )

    @classmethod
    def micro(cls) -> "ModelConfig":
        return cls()

    @classmethod
    def step1(cls) -> "ModelConfig":
        return cls(
            n_macros=2,
            d_model=512,
            n_heads=8,
            head_dim=64,
            vocab_size=4096,
            window_size=64,
            ffn_hidden_dim=1120,
            hippo_hidden_dim=128,
            rms_eps=1e-6,
        )

    @classmethod
    def production(cls) -> "ModelConfig":
        return cls(
            n_macros=6,
            d_model=2048,
            n_heads=32,
            head_dim=64,
            vocab_size=32768,
            window_size=512,
            ffn_hidden_dim=5504,
            hippo_hidden_dim=4096,
            rms_eps=1e-6,
        )

    @property
    def num_layers(self) -> int:
        return self.n_macros * 4


# ---------------------------------------------------------------------------
# 1.58-bit QAT Modules
# ---------------------------------------------------------------------------


class BitLinearQAT(nn.Module):
    """1.58-bit ternary linear projection with STE and symmetric INT8 activations."""

    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.weight = nn.Parameter(torch.randn(rows, cols) * (1.0 / math.sqrt(cols)) * 0.5)
        self.row_scale = nn.Parameter(torch.ones(rows))

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        use_ste = self.training if training is None else training
        w = self.weight
        gamma = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        w_q = round_half_away(w / gamma).clamp(-1.0, 1.0)
        if use_ste:
            w_ste = w + (w_q * gamma - w).detach()
            x_ste, _ = quantize_act_ste(x, training=True)
            return F.linear(x_ste, w_ste) * self.row_scale
        q_int, act_scale = quantize_act_ste(x, training=False)
        eff_row = (self.row_scale * gamma.squeeze(1)).to(torch.float32)
        acc = (q_int.to(torch.int32) @ w_q.to(torch.int32).T).to(torch.float32)
        return (acc * act_scale) * eff_row

    @torch.no_grad()
    def export_trit(self) -> tuple[np.ndarray, np.ndarray]:
        w = self.weight.detach()
        gamma = w.abs().mean(dim=1).clamp_min(1e-8)
        trit = round_half_away(w / gamma[:, None]).clamp(-1.0, 1.0).to(torch.int8)
        scale = (self.row_scale.detach() * gamma).to(torch.float32)
        return trit.cpu().numpy(), scale.cpu().numpy()


class SsmBlock(nn.Module):
    """Macro layers 1–3: BitLinear QKV + O(d_head^2) Gated DeltaNet + BitSwiGLU."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, h, hd, ffn = cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.ffn_hidden_dim
        self.cfg = cfg
        self.norm1 = nn.Parameter(torch.ones(d))
        self.norm2 = nn.Parameter(torch.ones(d))
        self.q = BitLinearQAT(d, d)
        self.k = BitLinearQAT(d, d)
        self.v = BitLinearQAT(d, d)
        self.o = BitLinearQAT(d, d)
        self.alpha_w = nn.Parameter(torch.randn(h, d) * 0.02)
        # Matches Rust `BitSsmBlock`: sigmoid(4.6) ≈ 0.990 slow retention by default.
        self.alpha_b = nn.Parameter(torch.full((h,), 4.6))
        self.beta_w = nn.Parameter(torch.randn(h, d) * 0.02)
        self.beta_b = nn.Parameter(torch.full((h,), -1.1))
        self.gate = BitLinearQAT(ffn, d)
        self.up = BitLinearQAT(ffn, d)
        self.down = BitLinearQAT(d, ffn)

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        bsz, steps, d = x.shape
        h, hd = self.cfg.n_heads, self.cfg.head_dim
        xn = rmsnorm(x, self.norm1, self.cfg.rms_eps)
        q = l2_norm_last(self.q(xn, training).view(bsz, steps, h, hd))
        k = l2_norm_last(self.k(xn, training).view(bsz, steps, h, hd))
        v = self.v(xn, training).view(bsz, steps, h, hd)
        alpha = torch.sigmoid(F.linear(xn, self.alpha_w, self.alpha_b))
        beta = torch.sigmoid(F.linear(xn, self.beta_w, self.beta_b))

        s = x.new_zeros(bsz, h, hd, hd)
        outs = []
        for t in range(steps):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            at = alpha[:, t].unsqueeze(-1)
            bt = beta[:, t].unsqueeze(-1)
            # Rank-1 O(d_head^2) update: u = S @ k, S = a S + b (v - a u) k^T, o = S @ q
            u = torch.einsum("bhij,bhj->bhi", s, kt)
            s = at.unsqueeze(-1) * s + bt.unsqueeze(-1) * torch.einsum(
                "bhi,bhj->bhij", vt - at * u, kt
            )
            outs.append(torch.einsum("bhij,bhj->bhi", s, qt))

        o = torch.stack(outs, dim=1).reshape(bsz, steps, d)
        x1 = x + self.o(o, training)
        xn2 = rmsnorm(x1, self.norm2, self.cfg.rms_eps)
        hidden = F.silu(self.gate(xn2, training)) * self.up(xn2, training)
        return x1 + self.down(hidden, training)


class MemBlock(nn.Module):
    """Macro layer 4: INT8 ring window attention + online neural hippocampus + BitSwiGLU."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, h, hd, ffn, hip = (
            cfg.d_model,
            cfg.n_heads,
            cfg.head_dim,
            cfg.ffn_hidden_dim,
            cfg.hippo_hidden_dim,
        )
        self.cfg = cfg
        self.norm1 = nn.Parameter(torch.ones(d))
        self.norm2 = nn.Parameter(torch.ones(d))
        self.wq = BitLinearQAT(d, d)
        self.wk = BitLinearQAT(d, d)
        self.wv = BitLinearQAT(d, d)
        self.wo = BitLinearQAT(d, d)
        self.hip_proj = BitLinearQAT(d, d)
        self.hip_k = BitLinearQAT(d, d)
        self.hip_v = BitLinearQAT(d, d)
        self.w1 = nn.Parameter(torch.randn(hip, d) * (1.0 / math.sqrt(d)))
        # 0.1x cold-start output scaling, matching Rust `HippocampusWeights::new_seeded`.
        self.w2 = nn.Parameter(torch.randn(d, hip) * (0.1 / math.sqrt(hip)))
        self.hip_version: int = 0
        self.gate_w = nn.Parameter(torch.randn(d) * 0.02)
        self.gate_b = nn.Parameter(torch.zeros(d))
        self.fg = BitLinearQAT(ffn, d)
        self.fu = BitLinearQAT(ffn, d)
        self.fd = BitLinearQAT(d, ffn)

    def _ring_attn(self, xn: torch.Tensor, training: bool) -> torch.Tensor:
        bsz, steps, d = xn.shape
        h, hd = self.cfg.n_heads, self.cfg.head_dim
        # Rust keeps `q` in FP32 and quantizes `k, v` per head to symmetric INT8 [-127, 127].
        qh = self.wq(xn, training).view(bsz, steps, h, hd)
        kh_raw = self.wk(xn, training).view(bsz, steps, h, hd)
        vh_raw = self.wv(xn, training).view(bsz, steps, h, hd)
        if training:
            kh, _ = quantize_act_ste(kh_raw, training=True)
            vh, _ = quantize_act_ste(vh_raw, training=True)
        else:
            kq, ks = quantize_act_ste(kh_raw, training=False)
            vq, vs = quantize_act_ste(vh_raw, training=False)
            kh = kq * ks
            vh = vq * vs

        scores = torch.einsum("bthd,bshd->bhts", qh, kh) * (1.0 / math.sqrt(hd))
        pos = torch.arange(steps, device=xn.device)
        valid = (pos[None, :] <= pos[:, None]) & ((pos[:, None] - pos[None, :]) < self.cfg.window_size)
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("bhts,bshd->bthd", attn, vh).reshape(bsz, steps, d)
        return self.wo(mixed, training)

    def _hippo_read_write(
        self, xn: torch.Tensor, training: bool, hippo_train: bool, online_lr: float = 0.25
    ) -> torch.Tensor:
        bsz, steps, d = xn.shape
        qh = l2_norm_last(self.hip_proj(xn, training))
        if not (training and hippo_train and steps >= 2):
            return F.silu(F.linear(qh, self.w1)) @ self.w2.T

        # Joint read/write interface meta-training:
        # First half performs differentiable analytic outer-product fast-weight writes
        # using (L2Norm(hip_k(xn)), hip_v(xn)); second half reads back via hip_proj(xn),
        # so LM loss trains hip_proj, hip_k, hip_v, and gate jointly!
        kh = l2_norm_last(self.hip_k(xn, training))
        vh = self.hip_v(xn, training)
        w1_fast = self.w1
        w2_fast = self.w2
        half = steps // 2
        outs = []
        for t in range(steps):
            qt = qh[:, t]
            mt = F.silu(qt @ w1_fast.T) @ w2_fast.T
            outs.append(mt)
            if t < half:
                kt = kh[:, t]
                vt = vh[:, t]
                z = kt @ w1_fast.T
                h_act = F.silu(z)
                pred = h_act @ w2_fast.T
                err = (pred - vt) / float(d)
                sig = torch.sigmoid(z)
                dsilu = sig * (1.0 + z * (1.0 - sig))
                gz = (err @ w2_fast) * dsilu
                grad_w2 = torch.einsum("bd,bh->dh", err, h_act) / float(bsz)
                grad_w1 = torch.einsum("bh,bd->hd", gz, kt) / float(bsz)
                w2_fast = w2_fast - online_lr * grad_w2
                w1_fast = w1_fast - online_lr * grad_w1
        return torch.stack(outs, dim=1)

    def forward(
        self, x: torch.Tensor, training: bool | None = None, hippo_train: bool = False
    ) -> torch.Tensor:
        use_ste = self.training if training is None else training
        xn = rmsnorm(x, self.norm1, self.cfg.rms_eps)
        a = self._ring_attn(xn, use_ste)
        m = self._hippo_read_write(xn, use_ste, hippo_train)
        g = torch.sigmoid(self.gate_w * xn + self.gate_b)
        y = x + g * a + (1.0 - g) * m
        yn = rmsnorm(y, self.norm2, self.cfg.rms_eps)
        hidden = F.silu(self.fg(yn, use_ste)) * self.fu(yn, use_ste)
        return y + self.fd(hidden, use_ste)


class CortexSeedModel(nn.Module):
    """Full macro-cycle 1.58-bit Cortex-Local QAT model matching Rust `CortexModel`."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_model) * 0.02)
        self.ssms = nn.ModuleList(SsmBlock(cfg) for _ in range(cfg.n_macros * 3))
        self.mems = nn.ModuleList(MemBlock(cfg) for _ in range(cfg.n_macros))
        self.final_norm = nn.Parameter(torch.ones(cfg.d_model))
        res_scale = 1.0 / math.sqrt(2.0 * cfg.num_layers)
        with torch.no_grad():
            for ssm in self.ssms:
                ssm.o.row_scale.fill_(res_scale)
                ssm.down.row_scale.fill_(res_scale)
            for mem in self.mems:
                mem.wo.row_scale.fill_(res_scale)
                mem.fd.row_scale.fill_(res_scale)

    def _quant_embed(self, use_ste: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        table = self.embed
        amax = table.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / 127.0
        q = round_half_away(table / scale).clamp(-127.0, 127.0)
        table_hat = q * scale
        table_ste = table + (table_hat - table).detach() if use_ste else table_hat
        return table_ste, q, scale.squeeze(1)

    def forward(
        self,
        ids: torch.Tensor,
        training: bool | None = None,
        hippo_train: bool = False,
    ) -> torch.Tensor:
        use_ste = self.training if training is None else training
        table_ste, q_emb, scale_emb = self._quant_embed(use_ste)
        x = table_ste[ids]
        for m in range(self.cfg.n_macros):
            for s in range(3):
                x = self.ssms[m * 3 + s](x, use_ste)
            x = self.mems[m](x, use_ste, hippo_train=hippo_train)
        hn = rmsnorm(x, self.final_norm, self.cfg.rms_eps)
        if use_ste:
            return F.linear(hn, table_ste)
        # Exact match with Rust `logits_from_hidden`: `scales[v] * dot(emb_i8[v], hn)`
        dots = F.linear(hn.to(torch.float32), q_emb.to(torch.float32))
        return dots * scale_emb.to(torch.float32)

    def loss(self, ids: torch.Tensor, hippo_train: bool = False) -> torch.Tensor:
        logits = self.forward(ids[:, :-1], training=True, hippo_train=hippo_train)
        return F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            ids[:, 1:].reshape(-1),
        )

    @torch.no_grad()
    def export_embed(self) -> tuple[np.ndarray, np.ndarray]:
        _, q, scale = self._quant_embed(use_ste=False)
        return q.to(torch.int8).cpu().numpy(), scale.to(torch.float32).cpu().numpy()

    @torch.no_grad()
    def quantized_logits_seq(self, tokens: list[int]) -> np.ndarray:
        """Return `[T, vocab]` eval quantized logits for `tokens`."""
        was_training = self.training
        self.eval()
        try:
            ids = torch.tensor([tokens], dtype=torch.long)
            logits = self.forward(ids, training=False, hippo_train=False)[0]
            return logits.cpu().numpy().astype(np.float32)
        finally:
            if was_training:
                self.train()


# Aliases used by `distill_train.py` and `distill_qwen.py`.
CortexModelTorch = CortexSeedModel


def Mini() -> CortexSeedModel:
    return CortexSeedModel(ModelConfig.mini())


# ---------------------------------------------------------------------------
# Binary `.ctxb` v1 (`flags = 1`) Exporter
# ---------------------------------------------------------------------------


def _u32(buf: bytearray, v: int) -> None:
    buf.extend(struct.pack("<I", int(v)))


def _u64(buf: bytearray, v: int) -> None:
    buf.extend(struct.pack("<Q", int(v)))


def _f32s(buf: bytearray, xs) -> None:
    arr = np.asarray(xs, dtype="<f4").reshape(-1)
    buf.extend(arr.tobytes())


def _pad64(buf: bytearray) -> None:
    while len(buf) % 64 != 0:
        buf.append(0)


def _write_bit(buf: bytearray, layer: BitLinearQAT) -> None:
    trit, scale = layer.export_trit()
    rows, cols, rows_pad, cols_pad, raw = pack_ternary(trit)
    _u32(buf, rows)
    _u32(buf, cols)
    _u32(buf, rows_pad)
    _u32(buf, cols_pad)
    _pad64(buf)
    buf.extend(raw)
    _f32s(buf, scale)


@torch.no_grad()
def export_ctxb(model: CortexSeedModel) -> bytes:
    """Serialize `model` into the 64B-aligned `CTXB` (`flags = 1`) format."""
    cfg = model.cfg
    emb_i8, emb_scale = model.export_embed()
    buf = bytearray()
    buf.extend(b"CTXB")
    _u32(buf, 1)
    _u32(buf, cfg.n_macros)
    _u32(buf, cfg.d_model)
    _u32(buf, cfg.n_heads)
    _u32(buf, cfg.head_dim)
    _u32(buf, cfg.vocab_size)
    _u32(buf, cfg.window_size)
    _u32(buf, cfg.ffn_hidden_dim)
    _u32(buf, cfg.hippo_hidden_dim)
    _f32s(buf, np.array([cfg.rms_eps], dtype=np.float32))
    _u32(buf, CTXB_FLAG_FULL_MEM_FINAL_NORM)
    _pad64(buf)
    assert len(buf) == 64

    buf.extend(np.asarray(emb_i8, dtype=np.int8).tobytes())
    _f32s(buf, emb_scale)

    for m in range(cfg.n_macros):
        for s in range(3):
            ssm = model.ssms[m * 3 + s]
            _f32s(buf, ssm.norm1.detach().cpu().numpy())
            _f32s(buf, ssm.norm2.detach().cpu().numpy())
            _f32s(buf, ssm.alpha_w.detach().cpu().numpy())
            _f32s(buf, ssm.alpha_b.detach().cpu().numpy())
            _f32s(buf, ssm.beta_w.detach().cpu().numpy())
            _f32s(buf, ssm.beta_b.detach().cpu().numpy())
            for bit in (ssm.q, ssm.k, ssm.v, ssm.o, ssm.gate, ssm.up, ssm.down):
                _write_bit(buf, bit)

        mem = model.mems[m]
        _f32s(buf, mem.norm1.detach().cpu().numpy())
        _f32s(buf, mem.norm2.detach().cpu().numpy())
        _f32s(buf, mem.gate_w.detach().cpu().numpy())
        _f32s(buf, mem.gate_b.detach().cpu().numpy())
        for bit in (mem.wq, mem.wk, mem.wv, mem.wo, mem.hip_proj, mem.hip_k, mem.hip_v):
            _write_bit(buf, bit)
        _u32(buf, cfg.d_model)
        _u32(buf, cfg.hippo_hidden_dim)
        _u32(buf, cfg.d_model)
        _u64(buf, mem.hip_version)
        _f32s(buf, mem.w1.detach().cpu().numpy())
        _f32s(buf, mem.w2.detach().cpu().numpy())
        for bit in (mem.fg, mem.fu, mem.fd):
            _write_bit(buf, bit)

    _f32s(buf, model.final_norm.detach().cpu().numpy())
    crc = zlib.crc32(bytes(buf)) & 0xFFFFFFFF
    _u32(buf, crc)
    return bytes(buf)


def export_to_ctxb(model: CortexSeedModel, filepath: str | Path) -> Path:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(export_ctxb(model))
    return path


# ---------------------------------------------------------------------------
# Training & Cross-Language Alignment Self-Check
# ---------------------------------------------------------------------------


def load_sequences(
    data_path: str | None, vocab_size: int, seq_len: int, n_seqs: int, seed: int
) -> torch.Tensor:
    rng = random.Random(seed)
    if data_path:
        raw = list(Path(data_path).read_bytes())
        seqs = []
        for _ in range(n_seqs):
            start = rng.randrange(0, max(1, len(raw) - seq_len))
            chunk = [b % vocab_size for b in raw[start : start + seq_len]]
            if len(chunk) < seq_len:
                chunk.extend([0] * (seq_len - len(chunk)))
            seqs.append(chunk)
        return torch.tensor(seqs, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (n_seqs, seq_len), generator=gen)


def train_seed(
    cfg: ModelConfig,
    steps: int,
    seed: int = 1,
    data_path: str | None = None,
    joint_hippocampus: bool = False,
) -> tuple[CortexSeedModel, dict[str, float]]:
    torch.manual_seed(seed)
    model = CortexSeedModel(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    seqs = load_sequences(data_path, cfg.vocab_size, 16, 4, seed)
    grad_norms: dict[str, float] = {}
    first_loss = None
    last_loss = None

    for step in range(max(1, steps)):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(seqs, hippo_train=joint_hippocampus)
        loss.backward()
        if step == 0 and joint_hippocampus:
            m0 = model.mems[0]
            for name, layer in (
                ("hip_proj", m0.hip_proj),
                ("hip_k", m0.hip_k),
                ("hip_v", m0.hip_v),
            ):
                g = layer.weight.grad
                grad_norms[name] = 0.0 if g is None else float(g.norm().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        val = float(loss.detach())
        if first_loss is None:
            first_loss = val
        last_loss = val
        if joint_hippocampus:
            for mem in model.mems:
                mem.hip_version = step + 1

    print(
        f"QAT seed trained {max(1, steps)} steps (joint_hippo={joint_hippocampus}): "
        f"loss {first_loss:.4f} -> {last_loss:.4f}"
    )
    if joint_hippocampus:
        print(f"joint hippocampus interface grad norms: {grad_norms}")
    return model, grad_norms


def self_check(model: CortexSeedModel, rust_bin: Path, prompt: str = "Hello") -> None:
    """Compare full PyTorch quantized forward vs Rust `cortex-cli` across all positions."""
    # Verify joint-hippocampus gradients flow into hip_proj, hip_k, and hip_v.
    probe_cfg = ModelConfig.mini()
    _, gnorms = train_seed(probe_cfg, steps=2, seed=7, joint_hippocampus=True)
    for key in ("hip_proj", "hip_k", "hip_v"):
        if gnorms.get(key, 0.0) <= 0.0:
            raise SystemExit(f"joint-hippocampus gradient check failed: {key} grad norm is 0")

    # Test token sequence spanning multiple steps (and exceeding `window` on mini!)
    tokens = [b % model.cfg.vocab_size for b in prompt.encode("utf-8")]
    if len(tokens) < 12:
        tokens = (tokens + list(range(1, 13)))[:12]
        tokens = [t % model.cfg.vocab_size for t in tokens]

    py_seq = model.quantized_logits_seq(tokens)
    blob = export_ctxb(model)

    with tempfile.TemporaryDirectory() as td:
        ctxb_path = Path(td) / "seed.ctxb"
        ctxb_path.write_bytes(blob)

        # 1. Position-by-position check via `cortex-cli logits --ids ...`
        proc = subprocess.run(
            [
                str(rust_bin),
                "logits",
                "--model",
                str(ctxb_path),
                "--ids",
                ",".join(map(str, tokens)),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            sys.stderr.write(proc.stdout)
            raise SystemExit(f"cortex-cli logits failed ({proc.returncode})")

        got_rows = []
        for line in proc.stdout.splitlines():
            if not line.startswith("pos="):
                continue
            floats = [float(x) for x in line.split()[2:]]
            got_rows.append(np.array(floats, dtype=np.float32))

        if len(got_rows) != len(tokens):
            raise SystemExit(f"expected {len(tokens)} logit rows, got {len(got_rows)}")

        worst = 0.0
        for pos, (py_row, rs_row) in enumerate(zip(py_seq, got_rows)):
            err = float(np.max(np.abs(py_row.astype(np.float64) - rs_row.astype(np.float64))))
            worst = max(worst, err)

        # 2. Also verify `--check-ctxb FILE --prompt ...` when vocab >= 256
        if model.cfg.vocab_size >= 256:
            proc2 = subprocess.run(
                [str(rust_bin), "--check-ctxb", str(ctxb_path), "--prompt", prompt],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc2.returncode != 0:
                raise SystemExit(f"cortex-cli --check-ctxb failed: {proc2.stderr}")
            rs_last = np.array(
                [float(v) for v in proc2.stdout.strip().splitlines()], dtype=np.float32
            )
            py_last = model.quantized_logits_seq(list(prompt.encode("utf-8")))[-1]
            err_last = float(
                np.max(np.abs(py_last.astype(np.float64) - rs_last.astype(np.float64)))
            )
            worst = max(worst, err_last)

    print(
        f"self-check ({len(tokens)} steps, layers={model.cfg.num_layers}, "
        f"d={model.cfg.d_model}, window={model.cfg.window_size}): "
        f"max abs logit err = {worst:.3e} (tolerance 1.000e-03)"
    )
    if worst >= 1e-3:
        raise SystemExit(f"self-check FAILED: max abs logit err {worst:.3e} >= 1e-3")
    print("SELF-CHECK: PASS")


def resolve_rust_bin(explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    root = Path(__file__).resolve().parents[1]
    for cand in (root / "target/release/cortex-cli", root / "target/debug/cortex-cli"):
        if cand.exists():
            return cand
    raise SystemExit("cortex-cli not found; pass --rust-bin or cargo build -p cortex-runtime")


def main() -> None:
    if "--describe-preset" in sys.argv:
        from presets import describe

        describe(sys.argv)
        return
    if "--step1" in sys.argv:
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--step1"]
        from step1_train import main as step1_main

        step1_main()
        return

    p = argparse.ArgumentParser(description="Cortex-Local 1.58-bit QAT seed trainer & .ctxb exporter")
    p.add_argument(
        "--config",
        choices=["mini", "micro", "step1", "production"],
        default="micro",
        help="model shape preset (default: micro 8-layer d=256)",
    )
    p.add_argument("--steps", type=int, default=10, help="QAT AdamW optimization steps")
    p.add_argument("--data", default=None, help="optional local file for byte sequences")
    p.add_argument(
        "--joint-hippocampus",
        "--hippo-train",
        dest="joint_hippocampus",
        action="store_true",
        help="jointly train hippocampus read/write projections via analytic fast weights",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="compare PyTorch quantized logits vs Rust cortex-cli, requiring err < 1e-3",
    )
    p.add_argument("--out", type=Path, default=None, help="output .ctxb file path")
    p.add_argument("--rust-bin", type=Path, default=None, help="path to cortex-cli binary")
    p.add_argument("--prompt", default="Hello Cortex!")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    cfg_map = {
        "mini": ModelConfig.mini,
        "micro": ModelConfig.micro,
        "step1": ModelConfig.step1,
        "production": ModelConfig.production,
    }
    cfg = cfg_map[args.config]()
    model, _ = train_seed(
        cfg,
        steps=args.steps,
        seed=args.seed,
        data_path=args.data,
        joint_hippocampus=args.joint_hippocampus,
    )

    if args.out is not None:
        out_path = export_to_ctxb(model, args.out)
        print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")

    if args.self_check:
        self_check(model, resolve_rust_bin(args.rust_bin), prompt=args.prompt)


if __name__ == "__main__":
    main()
