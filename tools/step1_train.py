#!/usr/bin/env python3
"""Step 1 sandbox trainer: ~25M 1.58-bit Cortex on synthetic Chinese stories.

Invoked from tools/train_seed_qat.py --step1. The forward matches the Rust
CortexModel block order (3 BitSSM + 1 memory block per macro) so the exported
.ctxb loads in cortex-cli.

No GPU is required. The run is time-boxed and logs the real loss.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

# 2 macros × 4 layers, d=512, ffn=1120, vocab=4096 → about 25.09M weight entries.
N_MACROS = 2
D = 512
HEADS = 8
HD = 64
VOCAB = 4096
WINDOW = 64
FFN = 1120
HIP = 128
EPS = 1e-6
SEQ = 64


def weight_matrix_count() -> int:
    emb = VOCAB * D
    ssm = 4 * D * D + (2 * D * FFN + FFN * D) + 2 * D + (2 * HEADS * D + 2 * HEADS)
    mem = 4 * D * D + D * D + (2 * D * FFN + FFN * D) + 4 * D + 2 * HIP * D
    return emb + N_MACROS * 3 * ssm + N_MACROS * mem


class BitLinear(nn.Module):
    """Per-row abs-mean ternary STE. Activations clamp to [-127, 127]."""

    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rows, cols))
        self.row_scale = nn.Parameter(torch.ones(rows))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        gamma = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        w_q = torch.clamp(torch.round(w / gamma), -1, 1)
        w_ste = w + (w_q * gamma - w).detach()
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = amax / 127.0
        q = torch.clamp(torch.round(x / scale), -127, 127)
        x_ste = x + (q * scale - x).detach()
        return F.linear(x_ste, w_ste) * self.row_scale

    @torch.no_grad()
    def export_trit(self) -> tuple[np.ndarray, np.ndarray]:
        w = self.weight.detach()
        gamma = w.abs().mean(dim=1).clamp_min(1e-8)
        trit = torch.clamp(torch.round(w / gamma[:, None]), -1, 1).to(torch.int8)
        scale = (self.row_scale.detach() * gamma).to(torch.float32)
        return trit.cpu().numpy(), scale.cpu().numpy()


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + EPS) * weight


def quant_ste(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return x + (q * scale - x).detach()


def delta_sequence(q, k, v, alpha, beta) -> torch.Tensor:
    """q,k,v [B,T,H,Dh]; alpha,beta [B,T,H]. Returns [B,T,H,Dh]."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    bsz, steps, heads, dh = q.shape
    s = q.new_zeros(bsz, heads, dh, dh)
    outs = []
    for t in range(steps):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        at = alpha[:, t].unsqueeze(-1)
        bt = beta[:, t].unsqueeze(-1)
        u = torch.einsum("bhij,bhj->bhi", s, kt)
        vk = vt - at * u
        s = at.unsqueeze(-1) * s + bt.unsqueeze(-1) * torch.einsum("bhi,bhj->bhij", vk, kt)
        outs.append(torch.einsum("bhij,bhi->bhj", s, qt))
    return torch.stack(outs, dim=1)


def ring_attn(q, k, v) -> torch.Tensor:
    """Causal window attention. q,k,v are [B,T,D] with D = heads * hd."""
    bsz, steps, _ = q.shape
    qh = quant_ste(q).view(bsz, steps, HEADS, HD)
    kh = quant_ste(k).view(bsz, steps, HEADS, HD)
    vh = quant_ste(v).view(bsz, steps, HEADS, HD)
    scale = HD**-0.5
    scores = torch.einsum("bthd,bshd->bhts", qh, kh) * scale
    mask = torch.triu(torch.ones(steps, steps, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", attn, vh)
    return out.reshape(bsz, steps, HEADS * HD)


class SsmBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.Parameter(torch.ones(D))
        self.norm2 = nn.Parameter(torch.ones(D))
        self.q = BitLinear(D, D)
        self.k = BitLinear(D, D)
        self.v = BitLinear(D, D)
        self.o = BitLinear(D, D)
        self.alpha_w = nn.Parameter(torch.zeros(HEADS, D))
        self.alpha_b = nn.Parameter(torch.zeros(HEADS))
        self.beta_w = nn.Parameter(torch.zeros(HEADS, D))
        self.beta_b = nn.Parameter(torch.full((HEADS,), -1.1))
        self.gate = BitLinear(FFN, D)
        self.up = BitLinear(FFN, D)
        self.down = BitLinear(D, FFN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = rmsnorm(x, self.norm1)
        q, k, v = self.q(xn), self.k(xn), self.v(xn)
        alpha = torch.sigmoid(xn @ self.alpha_w.T + self.alpha_b)
        beta = torch.sigmoid(xn @ self.beta_w.T + self.beta_b)
        o = delta_sequence(
            q.view(*q.shape[:2], HEADS, HD),
            k.view(*k.shape[:2], HEADS, HD),
            v.view(*v.shape[:2], HEADS, HD),
            alpha,
            beta,
        ).reshape_as(q)
        x = x + self.o(o)
        xn2 = rmsnorm(x, self.norm2)
        hidden = F.silu(self.gate(xn2)) * self.up(xn2)
        return x + self.down(hidden)


class MemBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.Parameter(torch.ones(D))
        self.norm2 = nn.Parameter(torch.ones(D))
        self.wq = BitLinear(D, D)
        self.wk = BitLinear(D, D)
        self.wv = BitLinear(D, D)
        self.wo = BitLinear(D, D)
        self.hip_proj = BitLinear(D, D)
        self.w1 = nn.Parameter(torch.empty(HIP, D))
        self.w2 = nn.Parameter(torch.empty(D, HIP))
        nn.init.normal_(self.w1, std=1.0 / D**0.5)
        nn.init.normal_(self.w2, std=0.1 / HIP**0.5)
        self.gate_w = nn.Parameter(torch.zeros(D))
        self.gate_b = nn.Parameter(torch.zeros(D))
        self.fg = BitLinear(FFN, D)
        self.fu = BitLinear(FFN, D)
        self.fd = BitLinear(D, FFN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = rmsnorm(x, self.norm1)
        a = self.wo(ring_attn(self.wq(xn), self.wk(xn), self.wv(xn)))
        qh = self.hip_proj(xn)
        m = F.silu(qh @ self.w1.T) @ self.w2.T
        g = torch.sigmoid(self.gate_w * xn + self.gate_b)
        y = x + g * a + (1.0 - g) * m
        yn = rmsnorm(y, self.norm2)
        hidden = F.silu(self.fg(yn)) * self.fu(yn)
        return y + self.fd(hidden)


class Step1Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Parameter(torch.empty(VOCAB, D))
        nn.init.normal_(self.embed, std=0.02)
        self.ssms = nn.ModuleList(SsmBlock() for _ in range(N_MACROS * 3))
        self.mems = nn.ModuleList(MemBlock() for _ in range(N_MACROS))

    def tied(self) -> torch.Tensor:
        table = self.embed
        amax = table.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / 127.0
        q = torch.clamp(torch.round(table / scale), -127, 127)
        return table + (q * scale - table).detach()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.tied()
        x = table[ids]
        ssm_i = 0
        for m in range(N_MACROS):
            for _ in range(3):
                x = self.ssms[ssm_i](x)
                ssm_i += 1
            x = self.mems[m](x)
        return F.linear(x, table)

    @torch.no_grad()
    def export_embed(self) -> tuple[np.ndarray, np.ndarray]:
        table = self.embed.detach()
        amax = table.abs().amax(dim=1).clamp_min(1e-8)
        scale = amax / 127.0
        q = torch.clamp(torch.round(table / scale[:, None]), -127, 127).to(torch.int8)
        return q.cpu().numpy(), scale.to(torch.float32).cpu().numpy()


def train_bpe(sample: bytes, n_merges: int) -> list[tuple[int, int]]:
    from collections import Counter

    seqs = [list(line) for line in sample.split(b"\n") if line]
    merges: list[tuple[int, int]] = []
    t0 = time.time()
    for i in range(n_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for s in seqs:
            counts.update(zip(s, s[1:]))
        if not counts:
            break
        (a, b), _ = counts.most_common(1)[0]
        new_id = 256 + i
        merges.append((a, b))
        nxt = []
        for s in seqs:
            out = []
            j = 0
            while j < len(s):
                if j + 1 < len(s) and s[j] == a and s[j + 1] == b:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(s[j])
                    j += 1
            nxt.append(out)
        seqs = nxt
        if i % 250 == 0 or i + 1 == n_merges:
            print(f"bpe {i + 1}/{n_merges} pair=({a},{b}) elapsed={time.time() - t0:.1f}s", flush=True)
    return merges


def sample_bytes_for_bpe(text: str, limit: int) -> bytes:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    rng = random.Random(21)
    rng.shuffle(lines)
    buf = bytearray()
    for ln in lines:
        raw = (ln + "\n").encode("utf-8")
        if len(buf) + len(raw) > limit:
            break
        buf.extend(raw)
    return bytes(buf)


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
            code = {0: 0, 1: 0b01, -1: 0b11}[w]
            rg, cb = r // 16, c // 16
            tile = (c % 16) // 4
            off = (rg * ncb + cb) * 64 + tile * 16 + (r % 16)
            blocks[off] |= code << (2 * (c % 4))
    return rows, cols, rows_pad, cols_pad, bytes(blocks)


def _u32(buf: bytearray, v: int) -> None:
    buf.extend(struct.pack("<I", int(v)))


def _u64(buf: bytearray, v: int) -> None:
    buf.extend(struct.pack("<Q", int(v)))


def _f32s(buf: bytearray, xs) -> None:
    arr = np.asarray(xs, dtype=np.float32).reshape(-1)
    buf.extend(arr.tobytes())


def _pad64(buf: bytearray) -> None:
    while len(buf) % 64:
        buf.append(0)


def _write_bit(buf: bytearray, layer: BitLinear) -> None:
    trit, scale = layer.export_trit()
    rows, cols, rows_pad, cols_pad, raw = pack_ternary(trit)
    _u32(buf, rows)
    _u32(buf, cols)
    _u32(buf, rows_pad)
    _u32(buf, cols_pad)
    _pad64(buf)
    buf.extend(raw)
    _f32s(buf, scale)


def export_ctxb(model: Step1Model) -> bytes:
    emb, emb_scale = model.export_embed()
    buf = bytearray()
    buf.extend(b"CTXB")
    _u32(buf, 1)
    _u32(buf, N_MACROS)
    _u32(buf, D)
    _u32(buf, HEADS)
    _u32(buf, HD)
    _u32(buf, VOCAB)
    _u32(buf, WINDOW)
    _u32(buf, FFN)
    _u32(buf, HIP)
    _f32s(buf, np.array([EPS], np.float32))
    _u32(buf, 0)
    _pad64(buf)
    buf.extend(np.asarray(emb, dtype=np.int8).tobytes())
    _f32s(buf, emb_scale)
    ssm_i = 0
    for _m in range(N_MACROS):
        for _s in range(3):
            layer = model.ssms[ssm_i]
            ssm_i += 1
            _f32s(buf, layer.norm1.detach().cpu().numpy())
            _f32s(buf, layer.norm2.detach().cpu().numpy())
            _f32s(buf, layer.alpha_w.detach().cpu().numpy())
            _f32s(buf, layer.alpha_b.detach().cpu().numpy())
            _f32s(buf, layer.beta_w.detach().cpu().numpy())
            _f32s(buf, layer.beta_b.detach().cpu().numpy())
            for bit in (layer.q, layer.k, layer.v, layer.o, layer.gate, layer.up, layer.down):
                _write_bit(buf, bit)
        mem = model.mems[_m]
        _f32s(buf, mem.norm1.detach().cpu().numpy())
        _f32s(buf, mem.norm2.detach().cpu().numpy())
        _f32s(buf, mem.gate_w.detach().cpu().numpy())
        _f32s(buf, mem.gate_b.detach().cpu().numpy())
        for bit in (mem.wq, mem.wk, mem.wv, mem.wo, mem.hip_proj):
            _write_bit(buf, bit)
        _u32(buf, D)
        _u32(buf, HIP)
        _u32(buf, D)
        _u64(buf, 0)
        _f32s(buf, mem.w1.detach().cpu().numpy())
        _f32s(buf, mem.w2.detach().cpu().numpy())
        for bit in (mem.fg, mem.fu, mem.fd):
            _write_bit(buf, bit)
    crc = zlib.crc32(bytes(buf)) & 0xFFFFFFFF
    _u32(buf, crc)
    return bytes(buf)


def rust_bin(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for cand in (ROOT / "target/release/cortex-cli", ROOT / "target/debug/cortex-cli"):
        if cand.exists():
            return cand
    raise SystemExit("cortex-cli not found; cargo build --release -p cortex-runtime")


def ensure_corpus(path: Path, nbytes: int) -> None:
    if path.exists() and path.stat().st_size >= nbytes * 9 // 10:
        print(f"corpus exists {path} ({path.stat().st_size} bytes)")
        return
    subprocess.check_call(
        [sys.executable, str(ROOT / "tools/make_step1_stories.py"), "--out", str(path), "--bytes", str(nbytes)]
    )


def ensure_tokens(corpus: Path, merges: Path, tokens: Path, rust: Path, bpe_bytes: int) -> None:
    text = corpus.read_text(encoding="utf-8")
    if not merges.exists():
        sample = sample_bytes_for_bpe(text, bpe_bytes)
        print(f"training BPE on {len(sample)} bytes toward vocab {VOCAB}")
        pairs = train_bpe(sample, VOCAB - 256)
        merges.write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8")
        print(f"wrote {merges} ({len(pairs)} merges, vocab {256 + len(pairs)})")
    if tokens.exists() and tokens.stat().st_size > 0:
        print(f"tokens exist {tokens}")
        return
    print("encoding corpus with cortex-cli")
    t0 = time.time()
    subprocess.check_call(
        [str(rust), "encode", "--vocab", str(merges), "--input", str(corpus), "--output", str(tokens)]
    )
    print(f"encode wall {time.time() - t0:.1f}s")


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    corpus = Path(args.corpus)
    merges = Path(args.merges)
    tokens_path = Path(args.tokens)
    out = Path(args.out)
    rust = rust_bin(args.rust_bin)
    ensure_corpus(corpus, args.corpus_bytes)
    ensure_tokens(corpus, merges, tokens_path, rust, args.bpe_bytes)
    token_arr = np.memmap(tokens_path, dtype=np.uint32, mode="r")
    print(f"token stream {len(token_arr)} ids, rust={rust}")
    if int(token_arr.max()) >= VOCAB:
        raise SystemExit(f"token id {int(token_arr.max())} outside vocab {VOCAB}")

    model = Step1Model()
    nparam = sum(p.numel() for p in model.parameters())
    print(
        f"config macros={N_MACROS} layers={N_MACROS * 4} d={D} heads={HEADS} "
        f"ffn={FFN} hip={HIP} window={WINDOW} vocab={VOCAB}"
    )
    print(f"parameters {nparam} ({nparam / 1e6:.3f}M) matrix_formula {weight_matrix_count()}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".loss.jsonl")
    curve = []
    t0 = time.time()
    deadline = t0 + args.minutes * 60
    step = 0
    model.train()
    while step < args.max_steps and time.time() < deadline:
        starts = rng.integers(0, len(token_arr) - SEQ, size=args.batch)
        batch = np.stack([token_arr[s : s + SEQ] for s in starts]).astype(np.int64)
        ids = torch.from_numpy(batch)
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        val = float(loss.detach())
        step += 1
        elapsed = time.time() - t0
        row = {"step": step, "loss": val, "elapsed_sec": round(elapsed, 2)}
        curve.append(row)
        print(f"step {step} loss {val:.4f} elapsed {elapsed:.1f}s", flush=True)
        if not np.isfinite(val):
            raise SystemExit("loss became non-finite")
    log_path.write_text("".join(json.dumps(r) + "\n" for r in curve), encoding="utf-8")
    print(f"exporting {out}")
    model.eval()
    blob = export_ctxb(model)
    out.write_bytes(blob)
    meta = {
        "n_macros": N_MACROS,
        "layers": N_MACROS * 4,
        "d_model": D,
        "n_heads": HEADS,
        "head_dim": HD,
        "ffn_hidden": FFN,
        "hip_hidden": HIP,
        "window": WINDOW,
        "vocab": VOCAB,
        "seq": SEQ,
        "batch": args.batch,
        "parameters": nparam,
        "matrix_formula": weight_matrix_count(),
        "tokens_in_corpus": int(len(token_arr)),
        "steps": step,
        "lr": args.lr,
        "minutes_budget": args.minutes,
        "wall_sec": round(time.time() - t0, 2),
        "loss_start": curve[0]["loss"] if curve else None,
        "loss_end": curve[-1]["loss"] if curve else None,
        "curve": curve,
        "ctxb": str(out),
        "merges": str(merges),
        "hardware": "cpu",
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in meta if k != "curve"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Step 1 Chinese-story QAT")
    p.add_argument("--corpus", type=Path, default=ROOT / "data/step1/corpus.txt")
    p.add_argument("--merges", type=Path, default=ROOT / "data/step1/merges.txt")
    p.add_argument("--tokens", type=Path, default=ROOT / "data/step1/tokens.u32")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/step1/step1.ctxb")
    p.add_argument("--corpus-bytes", type=int, default=60_000_000)
    p.add_argument("--bpe-bytes", type=int, default=400_000)
    p.add_argument("--minutes", type=float, default=22.0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--rust-bin", default=None)
    p.add_argument("--probe", action="store_true", help="one step then exit")
    args = p.parse_args()
    if args.probe:
        args.minutes = 10**9
        args.max_steps = 1
    train(args)


if __name__ == "__main__":
    main()
