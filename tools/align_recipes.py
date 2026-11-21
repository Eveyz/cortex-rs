#!/usr/bin/env python3
"""Instruction and tool-calling alignment recipes (no network downloads).

Implements:
- ChatML rendering with the 10 dense atomic control tokens (`SPECIALS`),
- Character-level `prompt_loss_mask` AND token-level `masked_sft_labels`
  (`ignore_index = -100` on system/user/tool turns, supervising only assistant
  completions + `<|im_end|>`),
- `masked_ce_loss` over `[B, T, V]` logits,
- DPO / SimPO preference losses and GRPO / RLVR verifiable tool-use rewards.
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

IGNORE_INDEX = -100


def render_chat(turns: list[tuple[str, str]]) -> str:
    parts = []
    for role, content in turns:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def prompt_loss_mask(turns: list[tuple[str, str]]) -> list[int]:
    """1 on assistant content characters, 0 elsewhere."""
    mask: list[int] = []
    for role, content in turns:
        prefix = f"<|im_start|>{role}\n"
        suffix = "<|im_end|>\n"
        mask.extend([0] * len(prefix))
        bit = 1 if role == "assistant" else 0
        mask.extend([bit] * len(content))
        mask.extend([0] * len(suffix))
    return mask


def encode_with_specials(text: str, special_base: int = 256) -> list[int]:
    """Byte-fallback encoding with dense atomic special IDs (`256..266`)."""
    raw = text.encode("utf-8")
    special_bytes = [(i, tok.encode("utf-8")) for i, tok in enumerate(SPECIALS)]
    ids: list[int] = []
    pos = 0
    while pos < len(raw):
        matched = None
        for idx, sb in special_bytes:
            if raw[pos:].startswith(sb):
                if matched is None or len(sb) > matched[0]:
                    matched = (len(sb), special_base + idx)
        if matched is not None:
            ids.append(matched[1])
            pos += matched[0]
        else:
            ids.append(raw[pos])
            pos += 1
    return ids


def masked_sft_labels(
    turns: list[tuple[str, str]], special_base: int = 256
) -> tuple[list[int], list[int]]:
    """Tokenize `turns` and mask non-assistant tokens with `IGNORE_INDEX = -100`."""
    ids: list[int] = []
    labels: list[int] = []
    im_end_id = special_base + 1
    for role, content in turns:
        prefix_ids = encode_with_specials(f"<|im_start|>{role}\n", special_base)
        body_ids = encode_with_specials(content, special_base)
        suffix_ids = [im_end_id, ord("\n")]
        ids.extend(prefix_ids)
        labels.extend([IGNORE_INDEX] * len(prefix_ids))
        ids.extend(body_ids)
        if role == "assistant":
            labels.extend(body_ids)
            ids.extend(suffix_ids)
            labels.extend([im_end_id, IGNORE_INDEX])
        else:
            labels.extend([IGNORE_INDEX] * len(body_ids))
            ids.extend(suffix_ids)
            labels.extend([IGNORE_INDEX] * len(suffix_ids))
    return ids, labels


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Autoregressive next-token CE ignoring `IGNORE_INDEX = -100`."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float = 0.1):
    """Standard DPO on log-probabilities."""
    pi = policy_chosen - policy_rejected
    ref = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi - ref))


def simpo_loss(policy_chosen, policy_rejected, gamma: float = 1.0, beta: float = 2.0):
    """Reference-free SimPO margin on the policy log-prob gap."""
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
    """Deterministic verifiable tool-use reward."""
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
        ("assistant", "<|thought|>sum</|thought|><|tool_call|>{\"name\":\"add\"}</|tool_call|>"),
    ]
    text = render_chat(turns)
    assert "<|im_start|>assistant" in text
    mask = prompt_loss_mask(turns)
    assert len(mask) == len(text)
    assert sum(mask) == len(turns[2][1])
    assert mask[0] == 0

    ids, labels = masked_sft_labels(turns)
    assert len(ids) == len(labels)
    assert max(ids) < 266
    supervised = [l for l in labels if l != IGNORE_INDEX]
    assert len(supervised) > 0 and 257 in supervised  # <|im_end|> supervised on assistant turn
    dummy_logits = torch.randn(1, len(ids), 266, requires_grad=True)
    loss = masked_ce_loss(dummy_logits, torch.tensor([labels], dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss) and dummy_logits.grad is not None

    chosen = torch.tensor(0.0)
    rejected = torch.tensor(-1.0)
    loss_near = dpo_loss(chosen, rejected, torch.tensor(0.0), torch.tensor(0.0), beta=0.1)
    loss_far = dpo_loss(
        torch.tensor(2.0), torch.tensor(-2.0), torch.tensor(0.0), torch.tensor(0.0), beta=0.1
    )
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
