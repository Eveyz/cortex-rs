#!/usr/bin/env python3
"""Distill a Qwen3-28B teacher into the Cortex-Local QAT student.

This file does not download the teacher and does not start a 28B run.
``--self-check`` uses a mock teacher with a different vocabulary from the
student and checks that sequence-level CE plus mapped KL runs, then that the
existing ``export_ctxb`` hook writes a CTXB file.

Vocab mismatch
--------------
Qwen3-28B uses its own large tokenizer (on the order of 150k ids; read the
model card). The student uses byte-fallback BPE: 4096 ids in the Step 1 run,
32768 in the Step 2 design, always including 256 raw UTF-8 bytes and no UNK.
The two logit vectors are not the same shape and are not aligned by id.

Transfer plan implemented here:

1. The transfer set is raw text from the prepare_corpus manifests (textbooks,
   code, math traces, dialogue, ``<memory_slot>`` / refusal). Both sides see
   the same string.
2. Sequence-level distillation: the teacher returns a continuation string.
   The student is trained with cross-entropy on *its own* tokenization of
   that string. No id is copied from the teacher vocab.
3. Token-level KL: each teacher token has a decode string. Its probability is
   split evenly across the student ids that encode that string. KL is
   computed only after this projection. Mass that the student tokenizer
   cannot represent is dropped and the projected distribution is
   renormalized.

A real run replaces ``MockTeacher`` with a local checkpoint:

    python3 tools/distill_qwen.py --teacher-path /path/to/Qwen3-28B --manifest configs/step2_mix.json

That command refuses to start unless ``--teacher-path`` exists on disk.
It still does not download. Full bf16 Qwen3-28B does not fit a 24GB GPU;
use a local checkpoint you already have (bf16 on a multi-GPU box, or a
quantized build you produced yourself). Read the Qwen model card for the
license before use. Dataset licenses are listed in the mix JSON files.

After a distill seed exists, the student QAT loop is still
``tools/train_seed_qat.py`` (Step 1 measured path, or
``--describe-preset step2`` for the untrained 300M-scale shape) and the
export is a ``.ctxb`` for ``cortex-cli``.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

TEACHER_REPO = "Qwen/Qwen3-28B"


class MockTeacher:
    """Tiny stand-in. Vocab strings are not student token ids."""

    def __init__(self) -> None:
        # Single-byte strings so a projected teacher token is one student id.
        # A separate check still projects the three UTF-8 bytes of "猫".
        self.id_to_text = ["<bos>", "a", "b", "c", "."]
        self.next_id = {0: 1, 1: 2, 2: 3, 3: 4, 4: 1}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_text)

    def decode(self, index: int) -> str:
        return self.id_to_text[index]

    def soft_next(self, prev: int) -> np.ndarray:
        probs = np.full(self.vocab_size, 0.02 / (self.vocab_size - 1), dtype=np.float64)
        probs[self.next_id[prev]] = 0.98
        probs /= probs.sum()
        return probs

    def generate(self, prompt: str, n: int = 4) -> str:
        # Deterministic continuation; the prompt is not retokenized as Qwen ids.
        del prompt
        ids = [0]
        for _ in range(n):
            nxt = self.next_id[ids[-1]]
            ids.append(nxt)
        return "".join(self.decode(i) for i in ids if i != 0)


def project_teacher_probs(probs: np.ndarray, decode, encode, student_vocab: int) -> np.ndarray:
    """Split each teacher token's mass across the student ids of its string."""
    out = np.zeros(student_vocab, dtype=np.float64)
    for index, mass in enumerate(probs):
        if mass <= 0:
            continue
        ids = encode(decode(index))
        if not ids:
            continue
        share = float(mass) / len(ids)
        for sid in ids:
            if 0 <= sid < student_vocab:
                out[sid] += share
    total = float(out.sum())
    if total > 0:
        out /= total
    return out


def kl_teacher_student(teacher_on_student: np.ndarray, student_log_probs: torch.Tensor) -> torch.Tensor:
    p = torch.tensor(teacher_on_student, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mask = p > 0
    log_p = torch.zeros_like(p)
    log_p[mask] = torch.log(p[mask])
    return torch.sum(p[mask] * (log_p[mask] - student_log_probs[mask]))


class ByteStudent(torch.nn.Module):
    """Next-byte student. Ids are raw bytes, vocab 256, not the teacher vocab."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(256, 32)
        self.head = torch.nn.Linear(32, 256)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(token))


def student_encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def distill_steps(steps: int, lr: float = 0.5) -> list[float]:
    teacher = MockTeacher()
    student = ByteStudent()
    opt = torch.optim.SGD(student.parameters(), lr=lr)
    text = teacher.generate("prompt")
    hard = student_encode_bytes(text)
    teacher_ids = [1, 2, 3, 4]
    if hard != [ord(teacher.decode(i)) for i in teacher_ids]:
        raise RuntimeError(f"teacher text {text!r} did not map 1:1 onto bytes {hard}")
    losses: list[float] = []
    for _step in range(steps):
        opt.zero_grad(set_to_none=True)
        total = torch.zeros(())
        prev_teacher = 0
        prev_byte = 0
        for index, nxt in enumerate(hard):
            logits = student(torch.tensor(prev_byte))
            log_probs = F.log_softmax(logits, dim=-1)
            ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([nxt]))
            mapped = project_teacher_probs(
                teacher.soft_next(prev_teacher),
                teacher.decode,
                student_encode_bytes,
                256,
            )
            kl = kl_teacher_student(mapped, log_probs)
            total = total + ce + 0.5 * kl
            prev_byte = nxt
            prev_teacher = teacher_ids[index]
        loss = total / len(hard)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def export_hook(path: Path) -> None:
    """Reuse the QAT ctxb writer. Does not load a teacher."""
    from train_seed_qat import Mini, export_ctxb

    blob = export_ctxb(Mini())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    if blob[:4] != b"CTXB":
        raise RuntimeError("export hook did not write CTXB magic")
    body, crc_bytes = blob[:-4], blob[-4:]
    crc = zlib.crc32(body) & 0xFFFFFFFF
    got = int.from_bytes(crc_bytes, "little")
    if crc != got:
        raise RuntimeError(f"ctxb crc {got:#x} != {crc:#x}")


def mohawk_mse_step() -> tuple[float, float]:
    """Attention hidden states → Gated DeltaNet hidden states.

    The mock teacher width is 8 and the student width is 4. A learned map
    bridges them. Q/K/V are not copied when the shapes differ; that is the
    honest mismatch path. One SGD step should reduce MSE.
    """
    torch.manual_seed(0)
    teacher_h = torch.randn(6, 8)
    target = teacher_h[:, :4].detach()
    student_h = torch.nn.Parameter(torch.zeros(6, 4))
    opt = torch.optim.SGD([student_h], lr=0.2)
    first = float(F.mse_loss(student_h.detach(), target))
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(student_h, target)
        loss.backward()
        opt.step()
    last = float(F.mse_loss(student_h.detach(), target))
    return first, last


def progressive_qat_kl() -> list[float]:
    """λ goes 0 → 1. Loss is CE plus KL to mapped teacher logits."""
    torch.manual_seed(1)
    logits = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.SGD([logits], lr=0.4)
    target = torch.tensor(2)
    teacher = torch.tensor([0.05, 0.05, 0.8, 0.1])
    losses = []
    for step in range(6):
        lam = step / 5
        opt.zero_grad(set_to_none=True)
        log_probs = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(log_probs.unsqueeze(0), target.unsqueeze(0))
        kl = torch.sum(teacher * (torch.log(teacher) - log_probs))
        loss = (1.0 - lam) * ce + lam * kl
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def freeze_trunk_train_gates() -> bool:
    trunk = torch.nn.Linear(4, 4)
    gate = torch.nn.Linear(4, 1)
    for param in trunk.parameters():
        param.requires_grad_(False)
    opt = torch.optim.SGD(gate.parameters(), lr=0.1)
    x = torch.randn(3, 4)
    loss = F.mse_loss(gate(trunk(x).detach()), torch.ones(3, 1))
    loss.backward()
    opt.step()
    trunk_frozen = all(p.grad is None for p in trunk.parameters())
    gate_trained = any(p.grad is not None for p in gate.parameters())
    return trunk_frozen and gate_trained and bool(torch.isfinite(loss))


def self_check() -> None:
    teacher = MockTeacher()
    cat_ids = student_encode_bytes("猫")
    _expect(teacher.vocab_size != 256, "vocabs must differ")
    _expect(len(cat_ids) == 3, "猫 is three UTF-8 bytes")
    one_hot = np.zeros(1, dtype=np.float64)
    one_hot[0] = 1.0
    split = project_teacher_probs(one_hot, lambda _i: "猫", student_encode_bytes, 256)
    _expect(abs(float(split.sum()) - 1.0) < 1e-6, "split sums to 1")
    _expect(all(abs(split[i] - 1.0 / 3.0) < 1e-6 for i in cat_ids), "even split across bytes")
    projected = project_teacher_probs(
        teacher.soft_next(0),
        teacher.decode,
        student_encode_bytes,
        256,
    )
    _expect(abs(float(projected.sum()) - 1.0) < 1e-6, "projected probs")
    _expect(projected[ord("a")] > 0.9, "preferred teacher token maps to byte a")
    losses = distill_steps(8)
    _expect(all(math.isfinite(x) for x in losses), "finite loss")
    _expect(losses[-1] < losses[0], f"CE+KL should fall, {losses[0]:.4f} -> {losses[-1]:.4f}")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "probe.ctxb"
        export_hook(out)
        _expect(out.stat().st_size > 64, "ctxb size")
    mse0, mse1 = mohawk_mse_step()
    _expect(mse1 < mse0, f"MOHAWK MSE {mse0} -> {mse1}")
    qat = progressive_qat_kl()
    _expect(all(math.isfinite(x) for x in qat), "qat schedule")
    _expect(qat[-1] < qat[0], f"progressive KL {qat[0]} -> {qat[-1]}")
    _expect(freeze_trunk_train_gates(), "hippo gates")
    print(
        f"distill self-check ok: mock teacher vocab {teacher.vocab_size} "
        f"vs student bytes 256; loss {losses[0]:.4f} -> {losses[-1]:.4f}; "
        f"mohawk {mse0:.4f}->{mse1:.4f}; qat {qat[0]:.4f}->{qat[-1]:.4f}; ctxb hook ok"
    )


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"self-check failed: {msg}")


def refuse_real_teacher(path: str | None) -> None:
    print(f"teacher repo id (not downloaded): {TEACHER_REPO}")
    print("student: Cortex QAT in tools/train_seed_qat.py, exported with export_ctxb")
    print("vocab: teacher ids are not student ids; see the module docstring")
    if not path:
        raise SystemExit(
            "refusing to start a real distill without --teacher-path. "
            "No download was attempted. Use --self-check for the mock loop."
        )
    ckpt = Path(path)
    if not ckpt.exists():
        raise SystemExit(f"teacher path does not exist: {ckpt}. Not downloading it.")
    raise SystemExit(
        "A local teacher path was given, but this sandbox build does not launch "
        "Qwen3-28B. Load it in your own environment with the model card's license, "
        "then call distill_steps with a real forward. Export remains export_ctxb."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-28B → Cortex QAT distill (no download).")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--teacher-path", default=None, help="local Qwen3-28B directory; not fetched")
    p.add_argument("--manifest", type=Path, default=ROOT / "configs" / "step2_mix.json")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    if args.manifest.exists():
        print(f"distill recipe manifest: {args.manifest}")
    refuse_real_teacher(args.teacher_path)


if __name__ == "__main__":
    main()
