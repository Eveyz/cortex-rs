#!/usr/bin/env python3
"""Distillation training: Qwen3 teacher JSONL -> 1.58-bit Cortex student.

Trains `CortexModelTorch` (`CortexSeedModel` from `tools/train_seed_qat.py`) on
JSONL records emitted by `tools/distill_generate.py` using sequence-level SFT
with strict prompt loss masking (`labels = -100` on prompt/BOS/padding tokens,
supervising only assistant completion + `<|im_end|>`).

Special token alignment with `crates/cortex-runtime/src/tokenizer.rs`
(`Tokenizer::byte_level()`, `vocab_with_specials = 266`):
  * `BOS = 256` (`<|im_start|>`)
  * `EOS = 257` (`<|im_end|>` — triggers immediate stop in `cortex-cli`)
  * `PAD = 0`   (appended only at sequence tail; causal masks prevent leakage)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seed_qat import CortexModelTorch, ModelConfig, export_to_ctxb  # noqa: E402

BOS, EOS, PAD = 256, 257, 0
IGNORE = -100


def encode_text(s: str) -> list[int]:
    return list(s.encode("utf-8"))


def build_example(prompt: str, response: str, seq_len: int):
    """Return `(ids, labels)` with `-100` prompt loss masking, or `None` if too long."""
    p_ids = encode_text(prompt)
    r_ids = encode_text(response) + [EOS]
    if 1 + len(r_ids) > seq_len:
        return None
    keep_prompt = seq_len - 1 - len(r_ids)
    p_ids = p_ids[-keep_prompt:] if keep_prompt < len(p_ids) else p_ids
    ids = [BOS] + p_ids + r_ids
    labels = [IGNORE] * (1 + len(p_ids)) + r_ids
    return ids, labels


def load_dataset(
    path: Path,
    seq_len: int,
    eval_frac: float,
    max_examples: int | None,
    seed: int,
):
    examples, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ex = build_example(rec.get("prompt", ""), rec.get("response", ""), seq_len)
            if ex is None:
                skipped += 1
                continue
            examples.append(ex)
            if max_examples and len(examples) >= max_examples:
                break
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_frac)) if eval_frac > 0 and len(examples) > 1 else 0
    eval_set = examples[:n_eval]
    train_set = examples[n_eval:]
    print(
        f"loaded {len(examples)} examples "
        f"({len(train_set)} train / {len(eval_set)} eval, {skipped} skipped: too long)",
        flush=True,
    )
    return train_set, eval_set


def collate(batch):
    max_len = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE, dtype=torch.long)
    for i, (ex_ids, ex_labels) in enumerate(batch):
        ids[i, : len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)
        labels[i, : len(ex_labels)] = torch.tensor(ex_labels, dtype=torch.long)
    return ids, labels


def batches(examples, batch_size: int, rng: random.Random):
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [examples[i] for i in idx[start : start + batch_size]]


@torch.no_grad()
def eval_loss(model, eval_set, batch_size: int, device) -> float:
    if not eval_set:
        return float("nan")
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(eval_set), batch_size):
        ids, labels = collate(eval_set[start : start + batch_size])
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids[:, :-1], training=False)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=IGNORE,
        )
        n_tok = int((labels[:, 1:] != IGNORE).sum().item())
        total += loss.item() * n_tok
        count += n_tok
    model.train()
    return total / max(1, count)


@torch.no_grad()
def generate_sample(model, prompt: str, max_new: int, device) -> str:
    model.eval()
    ids = [BOS] + encode_text(prompt)
    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x, training=False)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == EOS:
            break
        ids.append(nxt)
    model.train()
    body = bytes(b for b in ids[1:] if b < 256)
    return body.decode("utf-8", errors="replace")


SAMPLE_PROMPTS = [
    "请一步一步解答这道题，最后一个字给出答案：小明有 12 个苹果，吃了 3 个，又买了 5 个，现在有几个苹果？",
    "写出满足要求的 Python 代码：\n实现一个函数 binary_search(arr, target)，arr 为升序列表，返回 target 的下标，找不到返回 -1。",
    "请写一段关于「光合作用的过程」（生物）的教科书式讲解。",
]


def lr_schedule(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


def train(args) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    cfg = ModelConfig.micro() if args.config == "micro" else ModelConfig.production()
    if device.type == "cpu" and args.config == "production":
        print(
            "WARNING: production config on CPU will be extremely slow; "
            "use --config micro for a smoke test.",
            flush=True,
        )

    model = CortexModelTorch(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Resume: restore model + optimizer + step, so the LR schedule and the
    # epoch loop continue where the last run stopped instead of restarting.
    # (Previously only the model weights were restored, so a resume re-ran
    # warmup and the full epoch count — useless for a multi-day Colab run.)
    start_step = 0
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"resumed model from {args.init_from}", flush=True)
        if "optimizer" in state:
            opt.load_state_dict(state["optimizer"])
            print("resumed optimizer state", flush=True)
        start_step = int(state.get("step", 0))
        print(f"resumed at step {start_step}", flush=True)

    train_set, eval_set = load_dataset(
        Path(args.data),
        args.seq_len,
        args.eval_frac,
        args.max_examples,
        args.seed,
    )
    if not train_set:
        print("no training examples", file=sys.stderr)
        return 2

    steps_per_epoch = max(1, len(train_set) // max(1, args.batch_size * args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(
        f"total optimizer steps: {total_steps} "
        f"({steps_per_epoch}/epoch x {args.epochs}), warmup {warmup_steps}",
        flush=True,
    )
    if start_step >= total_steps:
        print(f"already at step {start_step}/{total_steps}; nothing to train", flush=True)
        return 0

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(path: Path) -> None:
        # Write to a temp file then rename, so a session killed mid-write
        # (Colab's 12h/24h cap) never leaves a corrupt checkpoint behind.
        tmp = path.with_name(path.name + ".tmp")
        torch.save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step},
            tmp,
        )
        tmp.replace(path)

    rng = random.Random(args.seed + 1)
    step = start_step
    micro = 0
    t0 = time.time()
    tokens_seen = 0
    model.train()
    done = False
    # Skip the already-trained part: whole epochs, then micro-batches into
    # the current epoch. (The resumed epoch is reshuffled — harmless.)
    start_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    skip_micro = (start_step % steps_per_epoch) * args.grad_accum if start_step > 0 else 0
    if start_step > 0:
        print(
            f"skipping to epoch {start_epoch}, {skip_micro} micro-batches into it",
            flush=True,
        )
    last_ckpt = time.time()
    for epoch in range(start_epoch, args.epochs):
        if done:
            break
        for batch in batches(train_set, args.batch_size, rng):
            if skip_micro > 0:
                skip_micro -= 1
                continue
            ids, labels = collate(batch)
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids[:, :-1], training=True)
            loss = (
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=IGNORE,
                )
                / args.grad_accum
            )
            loss.backward()
            tokens_seen += int((labels[:, 1:] != IGNORE).sum().item())
            micro += 1
            if micro % args.grad_accum != 0:
                continue
            lr = lr_schedule(step, total_steps, warmup_steps, args.lr, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                el = time.time() - t0
                print(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {loss.item() * args.grad_accum:.4f} "
                    f"lr {lr:.2e} tok/s {tokens_seen / max(1e-6, el):.0f}",
                    flush=True,
                )
            if args.eval_every and step % args.eval_every == 0:
                ev = eval_loss(model, eval_set, args.batch_size, device)
                print(f"--- eval step {step}: loss {ev:.4f}", flush=True)
                for p in SAMPLE_PROMPTS:
                    print(
                        f"[sample] {p[:40]}... -> "
                        f"{generate_sample(model, p, 64, device)[:200]}",
                        flush=True,
                    )
                save_ckpt(ckpt_dir / f"step-{step}.pt")
                save_ckpt(ckpt_dir / "latest.pt")
                last_ckpt = time.time()
                if args.export_every:
                    export_to_ctxb(model, ckpt_dir / f"step-{step}.ctxb")
            # Time-based checkpoint: a Colab session can be killed at any
            # moment (12h free / 24h Pro cap), so never go longer than
            # --ckpt-every-min without a resume point on disk.
            if args.ckpt_every_min and time.time() - last_ckpt >= args.ckpt_every_min * 60:
                save_ckpt(ckpt_dir / "latest.pt")
                last_ckpt = time.time()
                print(f"--- checkpoint step {step} -> latest.pt", flush=True)
            if step >= total_steps:
                done = True
                break

    ev = eval_loss(model, eval_set, args.batch_size, device)
    print(f"final eval loss: {ev:.4f}", flush=True)
    save_ckpt(ckpt_dir / "final.pt")
    save_ckpt(ckpt_dir / "latest.pt")
    out = export_to_ctxb(model, args.out)
    print(f"exported {out} ({out.stat().st_size} bytes)")
    return 0


def self_check() -> int:
    ids, labels = build_example("2+2?", "4", seq_len=32)
    assert ids[0] == BOS and ids[-1] == EOS
    assert all(l == IGNORE for l in labels[: 1 + len(b"2+2?")])
    assert labels[1 + len(b"2+2?") :] == [ord("4"), EOS]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        data_file = td_path / "toy.jsonl"
        rows = [
            {"prompt": "6*7?", "response": "42"},
            {"prompt": "3+5?", "response": "8"},
            {"prompt": "hello", "response": "world"},
            {"prompt": "rust", "response": "simd"},
        ]
        data_file.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        out_ctxb = td_path / "student.ctxb"
        rc = main(
            [
                "--data",
                str(data_file),
                "--config",
                "micro",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--grad-accum",
                "1",
                "--seq-len",
                "32",
                "--eval-frac",
                "0.25",
                "--eval-every",
                "0",
                "--ckpt-dir",
                str(td_path / "ckpts"),
                "--out",
                str(out_ctxb),
            ]
        )
        assert rc == 0 and out_ctxb.exists() and out_ctxb.read_bytes()[:4] == b"CTXB"
    print("distill_train self-check ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Distill Qwen3-teacher JSONL into the 1.58-bit Cortex student"
    )
    ap.add_argument("--self-check", action="store_true", help="smoke-test masking & CTXB export")
    ap.add_argument("--data", default=None, help="JSONL from distill_generate.py")
    ap.add_argument("--config", choices=["micro", "production"], default="production")
    ap.add_argument("--out", default="student.ctxb")
    ap.add_argument("--ckpt-dir", default="ckpts")
    ap.add_argument(
        "--init-from", default=None,
        help="resume from a .pt checkpoint (restores model + optimizer + step; "
        "use <ckpt-dir>/latest.pt to continue after a killed Colab session)",
    )
    ap.add_argument(
        "--ckpt-every-min", type=float, default=30,
        help="save <ckpt-dir>/latest.pt at least this often (minutes); "
        "0 disables the timer (eval checkpoints still save latest.pt)",
    )
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--eval-frac", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--export-every", action="store_true", help="also export .ctxb at each eval")
    ap.add_argument("--max-examples", type=int, default=None, help="cap dataset size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.self_check:
        return self_check()
    if not args.data:
        ap.error("--data is required unless --self-check is passed")
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
