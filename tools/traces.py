#!/usr/bin/env python3
"""L3 golden-trace hook. Drops failed detours. Does not train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_golden(episode: dict) -> dict | None:
    if not episode.get("verified"):
        return None
    problem = str(episode.get("problem", "")).strip()
    if not problem:
        return None
    chain = []
    for step in episode.get("steps", []):
        if step.get("ok") and not step.get("detour"):
            text = str(step.get("text", "")).strip()
            if text:
                chain.append(text)
    if not chain:
        return None
    return {"problem": problem, "chain": chain}


def write_goldens(episodes: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            gold = extract_golden(episode)
            if gold is None:
                continue
            handle.write(json.dumps(gold, ensure_ascii=False) + "\n")
            n += 1
    return n


def self_check() -> None:
    episode = {
        "problem": "add two numbers",
        "verified": True,
        "steps": [
            {"text": "wrong tool", "ok": False, "detour": True},
            {"text": "call add", "ok": True, "detour": False},
            {"text": "report 5", "ok": True, "detour": False},
        ],
    }
    gold = extract_golden(episode)
    assert gold is not None
    assert gold["chain"] == ["call add", "report 5"]
    episode["verified"] = False
    assert extract_golden(episode) is None
    print("traces self-check ok")


def main() -> None:
    p = argparse.ArgumentParser(description="Extract verified traces. No training.")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.self_check or args.input is None:
        self_check()
        return
    episodes = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    dest = args.output or Path("artifacts/golden.jsonl")
    n = write_goldens(episodes, dest)
    print(f"wrote {n} golden pairs to {dest}")


if __name__ == "__main__":
    main()
