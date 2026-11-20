#!/usr/bin/env python3
"""One-shot offline QAT seed tool for cortex-local.

This is not a 1.6B trainer and it does not download data. It:

1. Defines BitLinearQAT — ternary weight + symmetric INT8 activation with a
   straight-through estimator (forward uses the quantized values, backward is
   the identity). Activations are clamped to [-127, 127], never -128.
2. Optionally joint-trains the hippocampus MLP `W2 SiLU(W1 k)` with MSE for a
   few steps (`--joint-hippocampus`). That autograd lives only in this tool.
   The Rust runtime never builds a graph.
3. Exports a mini `.ctxb` (magic CTXB, hyper-parameters, packed Tile16x4 /
   CacheLine16x16 blocks, CRC32).
4. `--self-check` runs `cortex-cli logits` and requires max abs logit error
   < 1e-3 against an fp32 NumPy forward of the same quantized weights.

The NumPy forward is the numerical contract shared with `cortex-runtime`:
per-row abs-mean is folded into the exported row scale, the GEMV itself is
`acc * (absmax(x)/127) * row_scale` on trits in {-1,0,+1}.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    print("PyTorch is required for tools/train_seed_qat.py:", exc, file=sys.stderr)
    sys.exit(1)


def round_half_away(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def quantize_act(x: np.ndarray) -> tuple[np.ndarray, np.float32]:
    x = np.asarray(x, dtype=np.float32)
    absmax = np.max(np.abs(x)) if x.size else np.float32(0)
    if not np.isfinite(absmax) or absmax == 0:
        return np.zeros_like(x, dtype=np.int8), np.float32(1.0)
    scale = np.float32(absmax / np.float32(127.0))
    q = np.clip(round_half_away(x / scale), -127, 127).astype(np.int8)
    return q, scale


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = np.float32(1.0) / (np.float32(1.0) + np.exp(-x[pos]))
    z = np.exp(x[~pos])
    out[~pos] = z / (np.float32(1.0) + z)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x * sigmoid(x)


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    ms = np.mean(x * x).astype(np.float32)
    inv = np.float32(1.0) / np.sqrt(ms + np.float32(eps))
    return (x * inv * w).astype(np.float32)


def bitlinear(trit: np.ndarray, scales: np.ndarray, x: np.ndarray) -> np.ndarray:
    q, x_scale = quantize_act(x)
    acc = trit.astype(np.float32) @ q.astype(np.float32)
    return (acc * np.float32(x_scale) * scales.astype(np.float32)).astype(np.float32)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).copy()
    n = np.sqrt(np.sum(v * v)).astype(np.float32)
    if n > np.float32(1e-8):
        v /= n
    return v


def delta_step(s: np.ndarray, q: np.ndarray, k: np.ndarray, v: np.ndarray, alpha: float, beta: float):
    """Rank-1 Gated DeltaNet. s is [d, d], row-major S[i, j]."""
    q = l2_normalize(q)
    k = l2_normalize(k)
    v = np.asarray(v, dtype=np.float32)
    u = s @ k
    s_new = np.float32(alpha) * s + np.float32(beta) * np.outer(v - np.float32(alpha) * u, k)
    o = s_new.T @ q
    return s_new.astype(np.float32), o.astype(np.float32)


# --- STE BitLinear ----------------------------------------------------------


def _round_half_away_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


class BitLinearQAT(nn.Module):
    """1.58-bit linear with a straight-through estimator.

    Forward value: per-row abs-mean ternary, times that abs-mean, times a
    learned row scale, on symmetrically quantized activations.
    Backward: identity w.r.t. the latent weight and the pre-quant activation
    (the quantization error is detached).
    """

    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(rows, cols) * 0.05)
        self.row_scale = nn.Parameter(torch.ones(rows))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        gamma = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        w_hat = w_q * gamma
        w_ste = w + (w_hat - w).detach()
        flat = x.reshape(-1, x.shape[-1])
        amax = flat.abs().amax().clamp_min(1e-8)
        scale = amax / 127.0
        q = torch.clamp(_round_half_away_torch(x / scale), -127.0, 127.0)
        x_hat = q * scale
        x_ste = x + (x_hat - x).detach()
        y = F.linear(x_ste, w_ste) * self.row_scale
        return y

    @torch.no_grad()
    def export_trit(self) -> tuple[np.ndarray, np.ndarray]:
        w = self.weight.detach()
        gamma = w.abs().mean(dim=1).clamp_min(1e-8)
        trit = torch.clamp(torch.round(w / gamma[:, None]), -1, 1).to(torch.int8)
        scale = (self.row_scale.detach() * gamma).to(torch.float32)
        return trit.cpu().numpy(), scale.cpu().numpy()


def train_bitlinear(steps: int, rows: int, cols: int, seed: int) -> BitLinearQAT:
    torch.manual_seed(seed)
    layer = BitLinearQAT(rows, cols)
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)
    for _ in range(steps):
        x = torch.randn(8, cols)
        target = torch.randn(8, rows)
        loss = F.mse_loss(layer(x), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return layer


def train_hippocampus(steps: int, d_in: int, hidden: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Joint-train mode: a few SGD steps on ||W2 SiLU(W1 k) - v||^2."""
    torch.manual_seed(seed)
    w1 = nn.Parameter(torch.randn(hidden, d_in) / (d_in**0.5))
    # W2 starts an order of magnitude smaller, matching the Rust 0.1 scale.
    w2 = nn.Parameter(torch.randn(d_in, hidden) * 0.1 / (hidden**0.5))
    opt = torch.optim.SGD([w1, w2], lr=0.05)
    last = None
    for _ in range(steps):
        k = torch.randn(8, d_in)
        v = torch.randn(8, d_in)
        y = F.silu(k @ w1.T) @ w2.T
        loss = ((y - v) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    print(f"joint hippocampus MSE after {steps} steps: {last:.6f}")
    return w1.detach().cpu().numpy().astype(np.float32), w2.detach().cpu().numpy().astype(np.float32)


# --- tiny model matching CortexConfig::mini --------------------------------


class Mini:
    def __init__(self) -> None:
        self.n_macros = 1
        self.d = 16
        self.heads = 2
        self.hd = 8
        self.vocab = 32
        self.window = 8
        self.ffn = 16
        self.hip_h = 8
        self.eps = np.float32(1e-6)
        rng = np.random.default_rng(0)
        self.emb = rng.integers(-20, 21, size=(self.vocab, self.d), dtype=np.int8)
        self.emb_scale = rng.uniform(0.02, 0.08, size=(self.vocab,)).astype(np.float32)
        self.ssms = [self._ssm(rng) for _ in range(3)]
        self.mem = self._mem(rng)

    def _trit(self, rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
        return rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(rows, cols))

    def _bit(self, rng: np.random.Generator, rows: int, cols: int) -> dict:
        return {
            "trit": self._trit(rng, rows, cols),
            "scale": rng.uniform(0.02, 0.08, size=(rows,)).astype(np.float32),
        }

    def _ssm(self, rng: np.random.Generator) -> dict:
        d = self.d
        return {
            "norm1": np.ones(d, np.float32),
            "norm2": np.ones(d, np.float32),
            "alpha_w": (rng.normal(0, 0.02, size=(self.heads, d))).astype(np.float32),
            "alpha_b": np.zeros(self.heads, np.float32),
            "beta_w": (rng.normal(0, 0.02, size=(self.heads, d))).astype(np.float32),
            "beta_b": np.full(self.heads, -1.1, np.float32),
            "q": self._bit(rng, d, d),
            "k": self._bit(rng, d, d),
            "v": self._bit(rng, d, d),
            "o": self._bit(rng, d, d),
            "gate": self._bit(rng, self.ffn, d),
            "up": self._bit(rng, self.ffn, d),
            "down": self._bit(rng, d, self.ffn),
        }

    def _mem(self, rng: np.random.Generator) -> dict:
        d = self.d
        return {
            "norm1": np.ones(d, np.float32),
            "norm2": np.ones(d, np.float32),
            "gate_w": (rng.normal(0, 0.02, size=(d,))).astype(np.float32),
            "gate_b": np.zeros(d, np.float32),
            "wq": self._bit(rng, d, d),
            "wk": self._bit(rng, d, d),
            "wv": self._bit(rng, d, d),
            "wo": self._bit(rng, d, d),
            "hip_proj": self._bit(rng, d, d),
            "w1": (rng.normal(0, 0.1, size=(self.hip_h, d))).astype(np.float32),
            "w2": (rng.normal(0, 0.02, size=(d, self.hip_h))).astype(np.float32),
            "version": 0,
            "fg": self._bit(rng, self.ffn, d),
            "fu": self._bit(rng, self.ffn, d),
            "fd": self._bit(rng, d, self.ffn),
        }


def forward_tokens(model: Mini, tokens: list[int]) -> list[np.ndarray]:
    d = model.d
    states = [np.zeros((model.heads, model.hd, model.hd), np.float32) for _ in model.ssms]
    window = model.window
    k_ring = np.zeros((window, model.heads, model.hd), np.int8)
    v_ring = np.zeros((window, model.heads, model.hd), np.int8)
    k_scale = np.ones((window, model.heads), np.float32)
    v_scale = np.ones((window, model.heads), np.float32)
    pos = 0
    filled = 0
    outs = []
    for tok in tokens:
        x = model.emb[tok].astype(np.float32) * model.emb_scale[tok]
        for si, layer in enumerate(model.ssms):
            x = ssm_forward(model, layer, states[si], x)
        x = mem_forward(model, x, k_ring, v_ring, k_scale, v_scale, pos, filled)
        # mem_forward returns updated ring counters via attributes on a small box
        x, pos, filled = x
        logits = np.empty(model.vocab, np.float32)
        for v in range(model.vocab):
            logits[v] = np.float32(model.emb_scale[v]) * np.dot(
                model.emb[v].astype(np.float32), x
            )
        outs.append(logits.astype(np.float32))
    return outs


def _sig_scalar(x: float) -> float:
    x = float(x)
    if x >= 0.0:
        z = float(np.exp(np.float32(-x)))
        return float(np.float32(1.0) / np.float32(1.0 + z))
    z = float(np.exp(np.float32(x)))
    return float(np.float32(z) / np.float32(1.0 + z))


def ssm_forward(model: Mini, layer: dict, state: np.ndarray, x: np.ndarray) -> np.ndarray:
    xn = rmsnorm(x, layer["norm1"], model.eps)
    q = bitlinear(layer["q"]["trit"], layer["q"]["scale"], xn)
    k = bitlinear(layer["k"]["trit"], layer["k"]["scale"], xn)
    v = bitlinear(layer["v"]["trit"], layer["v"]["scale"], xn)
    o = np.zeros(model.d, np.float32)
    for h in range(model.heads):
        sl = slice(h * model.hd, (h + 1) * model.hd)
        a = _sig_scalar(float(np.dot(layer["alpha_w"][h], xn) + layer["alpha_b"][h]))
        b = _sig_scalar(float(np.dot(layer["beta_w"][h], xn) + layer["beta_b"][h]))
        state[h], oh = delta_step(state[h], q[sl], k[sl], v[sl], a, b)
        o[sl] = oh
    o = bitlinear(layer["o"]["trit"], layer["o"]["scale"], o)
    x1 = (np.asarray(x, dtype=np.float32) + o).astype(np.float32)
    xn2 = rmsnorm(x1, layer["norm2"], model.eps)
    hidden = silu(bitlinear(layer["gate"]["trit"], layer["gate"]["scale"], xn2)) * bitlinear(
        layer["up"]["trit"], layer["up"]["scale"], xn2
    )
    ffn = bitlinear(layer["down"]["trit"], layer["down"]["scale"], hidden)
    return (x1 + ffn).astype(np.float32)


def mem_forward(model, x, k_ring, v_ring, k_scale, v_scale, pos, filled):
    mem = model.mem
    xn = rmsnorm(x, mem["norm1"], model.eps)
    q = bitlinear(mem["wq"]["trit"], mem["wq"]["scale"], xn)
    k = bitlinear(mem["wk"]["trit"], mem["wk"]["scale"], xn)
    v = bitlinear(mem["wv"]["trit"], mem["wv"]["scale"], xn)
    slot = pos % model.window
    for h in range(model.heads):
        sl = slice(h * model.hd, (h + 1) * model.hd)
        qk, sk = quantize_act(k[sl])
        qv, sv = quantize_act(v[sl])
        k_ring[slot, h] = qk
        v_ring[slot, h] = qv
        k_scale[slot, h] = sk
        v_scale[slot, h] = sv
    pos += 1
    if filled < model.window:
        filled += 1
    mixed = attend(model, q, k_ring, v_ring, k_scale, v_scale, pos, filled)
    mixed = bitlinear(mem["wo"]["trit"], mem["wo"]["scale"], mixed)
    qh = bitlinear(mem["hip_proj"]["trit"], mem["hip_proj"]["scale"], xn)
    pre = mem["w1"] @ qh
    m = mem["w2"] @ silu(pre)
    g = sigmoid(mem["gate_w"] * xn + mem["gate_b"])
    y = (np.asarray(x, dtype=np.float32) + g * mixed + (np.float32(1.0) - g) * m).astype(np.float32)
    xn2 = rmsnorm(y, mem["norm2"], model.eps)
    hidden = silu(bitlinear(mem["fg"]["trit"], mem["fg"]["scale"], xn2)) * bitlinear(
        mem["fu"]["trit"], mem["fu"]["scale"], xn2
    )
    ffn = bitlinear(mem["fd"]["trit"], mem["fd"]["scale"], hidden)
    return (y + ffn).astype(np.float32), pos, filled


def attend(model, q, k_ring, v_ring, k_scale, v_scale, pos, filled):
    out = np.zeros(model.d, np.float32)
    if filled == 0:
        return out
    start = (pos % model.window) if filled == model.window else 0
    scale = np.float32(1.0 / np.sqrt(np.float32(model.hd)))
    for h in range(model.heads):
        qh = q[h * model.hd : (h + 1) * model.hd]
        scores = np.empty(filled, np.float32)
        for t in range(filled):
            slot = (start + t) % model.window
            kd = k_ring[slot, h].astype(np.float32) * np.float32(k_scale[slot, h])
            scores[t] = np.dot(qh, kd) * scale
        m = np.max(scores)
        ex = np.exp(scores - m)
        ex = ex / np.sum(ex)
        acc = np.zeros(model.hd, np.float32)
        for t in range(filled):
            slot = (start + t) % model.window
            vd = v_ring[slot, h].astype(np.float32) * np.float32(v_scale[slot, h])
            acc += np.float32(ex[t]) * vd
        out[h * model.hd : (h + 1) * model.hd] = acc
    return out


# --- ctxb -------------------------------------------------------------------


def pack_ternary(trit: np.ndarray) -> tuple[int, int, int, int, bytes]:
    rows, cols = trit.shape
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
                raise ValueError(f"non-ternary {w}")
            rg = r // 16
            cb = c // 16
            tile = (c % 16) // 4
            col_in = c % 4
            row_in = r % 16
            off = (rg * ncb + cb) * 64 + tile * 16 + row_in
            blocks[off] |= code << (2 * col_in)
    return rows, cols, rows_pad, cols_pad, bytes(blocks)


def _u32(buf: bytearray, v: int) -> None:
    buf.extend(struct.pack("<I", v))


def _u64(buf: bytearray, v: int) -> None:
    buf.extend(struct.pack("<Q", v))


def _f32s(buf: bytearray, xs: np.ndarray) -> None:
    arr = np.asarray(xs, dtype=np.float32).reshape(-1)
    buf.extend(arr.tobytes())


def _pad64(buf: bytearray) -> None:
    while len(buf) % 64:
        buf.append(0)


def _write_bit(buf: bytearray, layer: dict) -> None:
    rows, cols, rows_pad, cols_pad, raw = pack_ternary(layer["trit"])
    _u32(buf, rows)
    _u32(buf, cols)
    _u32(buf, rows_pad)
    _u32(buf, cols_pad)
    _pad64(buf)
    buf.extend(raw)
    _f32s(buf, layer["scale"])


def export_ctxb(model: Mini) -> bytes:
    buf = bytearray()
    buf.extend(b"CTXB")
    _u32(buf, 1)
    _u32(buf, model.n_macros)
    _u32(buf, model.d)
    _u32(buf, model.heads)
    _u32(buf, model.hd)
    _u32(buf, model.vocab)
    _u32(buf, model.window)
    _u32(buf, model.ffn)
    _u32(buf, model.hip_h)
    _f32s(buf, np.array([model.eps], np.float32))
    _u32(buf, 0)
    _pad64(buf)
    buf.extend(model.emb.astype(np.int8).tobytes())
    _f32s(buf, model.emb_scale)
    for layer in model.ssms:
        _f32s(buf, layer["norm1"])
        _f32s(buf, layer["norm2"])
        _f32s(buf, layer["alpha_w"])
        _f32s(buf, layer["alpha_b"])
        _f32s(buf, layer["beta_w"])
        _f32s(buf, layer["beta_b"])
        for key in ("q", "k", "v", "o", "gate", "up", "down"):
            _write_bit(buf, layer[key])
    mem = model.mem
    _f32s(buf, mem["norm1"])
    _f32s(buf, mem["norm2"])
    _f32s(buf, mem["gate_w"])
    _f32s(buf, mem["gate_b"])
    for key in ("wq", "wk", "wv", "wo", "hip_proj"):
        _write_bit(buf, mem[key])
    _u32(buf, model.d)
    _u32(buf, model.hip_h)
    _u32(buf, model.d)
    _u64(buf, mem["version"])
    _f32s(buf, mem["w1"])
    _f32s(buf, mem["w2"])
    for key in ("fg", "fu", "fd"):
        _write_bit(buf, mem[key])
    crc = zlib.crc32(bytes(buf)) & 0xFFFFFFFF
    _u32(buf, crc)
    return bytes(buf)


def self_check(model: Mini, rust_bin: Path, tokens: list[int]) -> None:
    ref = forward_tokens(model, tokens)
    blob = export_ctxb(model)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "seed.ctxb"
        path.write_bytes(blob)
        proc = subprocess.run(
            [str(rust_bin), "logits", "--model", str(path), "--ids", ",".join(map(str, tokens))],
            check=False,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        raise SystemExit(f"cortex-cli logits failed ({proc.returncode})")
    got = []
    for line in proc.stdout.splitlines():
        if not line.startswith("pos="):
            continue
        parts = line.split()
        floats = [float(x) for x in parts[2:]]
        got.append(np.array(floats, np.float32))
    if len(got) != len(ref):
        raise SystemExit(f"expected {len(ref)} logit rows, got {len(got)}")
    worst = 0.0
    for i, (a, b) in enumerate(zip(ref, got)):
        err = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        worst = max(worst, err)
        print(f"pos {i}: max abs logit err {err:.3e}")
    if worst >= 1e-3:
        raise SystemExit(f"self-check failed: max abs err {worst:.3e} >= 1e-3")
    print(f"self-check ok: max abs logit err {worst:.3e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Tiny BitLinear QAT seed exporter (no corpus download).")
    p.add_argument("--steps", type=int, default=2, help="QAT / hippocampus SGD steps")
    p.add_argument("--joint-hippocampus", action="store_true", help="joint-train W1/W2 with the seed")
    p.add_argument("--self-check", action="store_true", help="compare Rust logits, require err < 1e-3")
    p.add_argument("--out", type=Path, default=None, help="write the .ctxb here")
    p.add_argument(
        "--rust-bin",
        type=Path,
        default=None,
        help="path to cortex-cli (default: target/release then target/debug)",
    )
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    print("BitLinearQAT STE: per-row abs-mean ternary, INT8 acts in [-127, 127], identity backward")
    layer = train_bitlinear(args.steps, rows=16, cols=16, seed=args.seed)
    trit, scale = layer.export_trit()
    print(f"exported QAT projection trit {trit.shape} scale mean {float(scale.mean()):.4f}")

    model = Mini()
    # Drop the trained projection into layer-0 Q so the checkpoint actually
    # contains a QAT-derived matrix, not only synthetic trits.
    model.ssms[0]["q"]["trit"] = trit.astype(np.int8)
    model.ssms[0]["q"]["scale"] = scale.astype(np.float32)

    if args.joint_hippocampus:
        w1, w2 = train_hippocampus(args.steps, model.d, model.hip_h, seed=args.seed + 3)
        model.mem["w1"] = w1
        model.mem["w2"] = w2
        model.mem["version"] = args.steps

    blob = export_ctxb(model)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(blob)
        print(f"wrote {args.out} ({len(blob)} bytes)")

    if args.self_check:
        rust = args.rust_bin
        if rust is None:
            root = Path(__file__).resolve().parents[1]
            for cand in (root / "target/release/cortex-cli", root / "target/debug/cortex-cli"):
                if cand.exists():
                    rust = cand
                    break
        if rust is None or not Path(rust).exists():
            raise SystemExit("cortex-cli not found; pass --rust-bin or cargo build -p cortex-runtime")
        self_check(model, Path(rust), [1, 2, 3])


if __name__ == "__main__":
    main()
