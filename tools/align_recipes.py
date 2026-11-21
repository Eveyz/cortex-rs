#!/usr/bin/env python3
"""Instruction and tool-calling scaffolds. No downloads and no training.

Prompt-loss masking, DPO/SimPO, and GRPO/RLVR are checked on dummy tensors.
Manifest weights are checked against the committed JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

SPECIALS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|thought|>",
    "</|thought|>",
    "<|tools|>",
    "</|tools|>",
    "<|tool_call|>",
    "</|tool_call|>",
    "<|tool_response|>",
    "</|tool_response|>",
)


def render_chat(turns: list[tuple[str, str]]) -> str:
    parts = []
    for role, content in turns:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def prompt_loss_mask(turns: list[tuple[str, str]]) -> list[int]:
    """1 on assistant content characters, 0 elsewhere. Loss hooks use this mask."""
    mask: list[int] = []
    for role, content in turns:
        prefix = f"<|im_start|>{role}\n"
        suffix = "<|im_end|>\n"
        mask.extend([0] * len(prefix))
        bit = 1 if role == "assistant" else 0
        mask.extend([bit] * len(content))
        mask.extend([0] * len(suffix))
    return mask


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float = 0.1):
    """Standard DPO on log-probabilities. No optimizer step is required to call it."""
    pi = policy_chosen - policy_rejected
    ref = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi - ref))


def simpo_loss(policy_chosen, policy_rejected, gamma: float = 1.0, beta: float = 2.0):
    """SimPO margin on the policy log-prob gap. Reference-free."""
    return -F.logsigmoid(beta * (policy_chosen - policy_rejected) - gamma)


def grpo_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    scale = var**0.5
    if scale < 1e-8:
        return [0.0 for _ in rewards]
    return [(r - mean) / scale for r in rewards]


def rlvr_reward(parsed: bool, schema_ok: bool, sandbox_ok: bool | None) -> float:
    """Deterministic reward. Sandbox is optional."""
    score = 0.0
    if parsed:
        score += 0.4
    if schema_ok:
        score += 0.4
    if sandbox_ok is True:
        score += 0.2
    return score


def load_mix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def self_check() -> None:
    turns = [
        ("system", "You are local."),
        ("user", "add 2 and 3"),
        ("assistant", "I will call add."),
    ]
    text = render_chat(turns)
    for token in SPECIALS:
        assert token in text or token.startswith("<|tool") or token.startswith("<|thought") or token.startswith("</")
    assert "<|im_start|>assistant" in text
    mask = prompt_loss_mask(turns)
    assert len(mask) == len(text)
    assert sum(mask) == len("I will call add.")
    assert mask[0] == 0

    chosen = torch.tensor(0.0)
    rejected = torch.tensor(-1.0)
    loss_near = dpo_loss(chosen, rejected, torch.tensor(0.0), torch.tensor(0.0), beta=0.1)
    loss_far = dpo_loss(torch.tensor(2.0), torch.tensor(-2.0), torch.tensor(0.0), torch.tensor(0.0), beta=0.1)
    assert torch.isfinite(loss_near) and float(loss_far) < float(loss_near)
    sim = simpo_loss(torch.tensor(1.0), torch.tensor(-1.0))
    assert torch.isfinite(sim)

    rewards = [
        rlvr_reward(True, True, True),
        rlvr_reward(True, False, None),
        rlvr_reward(False, False, None),
    ]
    assert rewards == [1.0, 0.4, 0.0]
    adv = grpo_advantages(rewards)
    assert abs(sum(adv)) < 1e-6
    assert adv[0] > adv[-1]

    tool = load_mix(ROOT / "configs" / "tool_mix.json")
    weights = [c["weight"] for c in tool["components"]]
    assert abs(sum(weights) - 1.0) < 1e-9
    assert weights == [0.5, 0.2, 0.15, 0.15]
    assert tool["download"] is False
    instruct = load_mix(ROOT / "configs" / "instruct_mix.json")
    assert abs(sum(c["weight"] for c in instruct["components"]) - 1.0) < 1e-9
    assert instruct["download"] is False
    print("align_recipes self-check ok")


def main() -> None:
    p = argparse.ArgumentParser(description="Instruction and tool scaffolds. No download.")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    print(json_dump(ROOT / "configs" / "tool_mix.json"))
    print(json_dump(ROOT / "configs" / "instruct_mix.json"))


def json_dump(path: Path) -> str:
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
