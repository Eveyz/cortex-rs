#!/usr/bin/env python3
"""Step 1 sandbox trainer: ~25M 1.58-bit Cortex on synthetic Chinese stories.

Invoked from `tools/train_seed_qat.py --step1`. Uses the exact numerically
aligned `CortexSeedModel` (`round_half_away`, `o = S @ q`, per-head INT8 K/V
sliding-window attention with `WINDOW` mask, L2-normalized hippocampus read,
`final_norm`, and `CTXB` `flags = 1` export) so PyTorch quantized logits match
`cortex-cli` within `< 1e-3` even past `WINDOW = 64`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from train_seed_qat import CortexSeedModel, ModelConfig, export_ctxb  # noqa: E402

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
    mem = 7 * D * D + (2 * D * FFN + FFN * D) + 4 * D + 2 * HIP * D
    return emb + N_MACROS * 3 * ssm + N_MACROS * mem + D


class Step1Model(CortexSeedModel):
    def __init__(self) -> None:
        super().__init__(ModelConfig.step1())


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
            print(
                f"bpe {i + 1}/{n_merges} pair=({a},{b}) elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
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
        [
            sys.executable,
            str(ROOT / "tools/make_step1_stories.py"),
            "--out",
            str(path),
            "--bytes",
            str(nbytes),
        ]
    )


def ensure_tokens(corpus: Path, merges: Path, tokens: Path, rust: Path, bpe_bytes: int) -> None:
    text = corpus.read_text(encoding="utf-8")
    if not merges.exists():
        sample = sample_bytes_for_bpe(text, bpe_bytes)
        print(f"training BPE on {len(sample)} bytes toward vocab {VOCAB}")
        pairs = train_bpe(sample, VOCAB - 256)
        merges.parent.mkdir(parents=True, exist_ok=True)
        merges.write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8")
        print(f"wrote {merges} ({len(pairs)} merges, vocab {256 + len(pairs)})")
    if tokens.exists() and tokens.stat().st_size > 0:
        print(f"tokens exist {tokens}")
        return
    tokens.parent.mkdir(parents=True, exist_ok=True)
    print("encoding corpus with cortex-cli")
    t0 = time.time()
    subprocess.check_call(
        [
            str(rust),
            "encode",
            "--vocab",
            str(merges),
            "--input",
            str(corpus),
            "--output",
            str(tokens),
        ]
    )
    print(f"encode wall {time.time() - t0:.1f}s")


def self_check_step1(rust: Path) -> None:
    """Verify Step1Model (`25.9M`, `d=512`, `vocab=4096`) matches Rust `< 1e-3`."""
    torch.manual_seed(21)
    model = Step1Model()
    model.eval()
    tokens = [(i * 37 + 11) % VOCAB for i in range(8)]
    py_rows = model.quantized_logits_seq(tokens)
    blob = export_ctxb(model)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "step1_probe.ctxb"
        p.write_bytes(blob)
        proc = subprocess.run(
            [str(rust), "logits", "--model", str(p), "--ids", ",".join(map(str, tokens))],
            check=False,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise SystemExit(f"cortex-cli logits failed: {proc.stderr}")
    rs_rows = [
        np.array([float(v) for v in ln.split()[2:]], dtype=np.float32)
        for ln in proc.stdout.splitlines()
        if ln.startswith("pos=")
    ]
    if len(rs_rows) != len(tokens):
        raise SystemExit(f"row mismatch {len(rs_rows)} != {len(tokens)}")
    worst = max(
        float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        for a, b in zip(py_rows, rs_rows)
    )
    print(f"step1 self-check ({len(tokens)} steps, d={D}, vocab={VOCAB}): max abs logit err = {worst:.3e}")
    if worst >= 1e-3:
        raise SystemExit(f"step1 self-check failed: {worst:.3e} >= 1e-3")
    print("step1 self-check ok")


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
        logits = model(ids, training=True)
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
    p.add_argument("--self-check", action="store_true", help="check PyTorch vs Rust past window=64")
    args = p.parse_args()
    if args.self_check:
        self_check_step1(rust_bin(args.rust_bin))
        return
    if args.probe:
        args.minutes = 10**9
        args.max_steps = 1
    train(args)


if __name__ == "__main__":
    main()
