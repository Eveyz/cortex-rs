#!/usr/bin/env python3
"""Prepare corpus manifests and clean text for later Step 2 / Step 3 runs.

Default paths never contact the network. ``--self-check`` and ``--dry-run``
read the committed fixtures only. ``--download`` writes a fetch plan and
still does not download.

Three hard filters from the data spec:

1. Drop acausal list-like text (phone books, SKU tables, tickers, communiqués).
2. Drop fragment comments and context-free replies shorter than 100 characters.
3. Invert answer-first textbook items so the reason precedes the answer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "step2" / "fixtures"

PHONE = re.compile(r"(?:\+?\d{1,3}[- ]?)?(?:\d{3}[- ]\d{3,4}[- ]\d{4})")
SKU = re.compile(r"^[A-Z0-9]{2,}[-_][A-Z0-9]{2,}\b", re.I)
TICKER = re.compile(r"^[A-Z]{1,5}\s+\d+\.\d+")
COMMUNIQUE = re.compile(r"会议纪要|兹定于|新闻通稿|参会人员|communique", re.I)

ANSWER_EN = re.compile(
    r"(?is)^answer\s*:\s*(.+?)\n+because\s+(.+)$"
)
ANSWER_ZH = re.compile(
    r"^答案\s*[:：]\s*(.+?)。\s*因为\s*(.+)$",
    re.S,
)
CHOICE_BECAUSE = re.compile(
    r"(?is)^([A-D])[\.\)]\s*(?:because|因为)\s+(.+)$"
)


def step2_manifest() -> dict:
    """~2GB bilingual textbooks + basic Python. Design mix, not a dump."""
    total_bytes = 2_000_000_000
    parts = [
        ("bilingual_textbooks", 0.55, [
            "Cosmopedia v2",
            "FineWeb-Edu score>=4.2",
            "CCI3-HQ textbook subset",
        ], "Apache-2.0 (Cosmopedia v2); ODC-By 1.0 (FineWeb-Edu); CCI3-HQ: read the BAAI card before use"),
        ("basic_python", 0.30, [
            "StarCoder2-Edu / The Stack Python subset with docstrings",
        ], "Per-file original licenses; BigCode / Software Heritage terms"),
        ("short_explanations", 0.15, [
            "synthetic vernacular explainers (this repo's generator style)",
        ], "Original synthetic text, no third-party claim"),
    ]
    components = []
    for name, weight, sources, license_note in parts:
        components.append({
            "name": name,
            "weight": weight,
            "bytes_design": int(total_bytes * weight),
            "sources": sources,
            "license": license_note,
        })
    return {
        "name": "step2_mix",
        "step": 2,
        "purpose": "Design rung for short QA, simple explanations, and small scripts. Not a measured training result.",
        "total_bytes_design": total_bytes,
        "token_budget_design": 500_000_000,
        "token_budget_note": "Planning figure at about 4 bytes/token. Not counted on a real download.",
        "hours_design": "8-12",
        "hours_note": "Curriculum design budget, not a wall-clock measurement.",
        "download": False,
        "components": components,
    }


def step3_manifest() -> dict:
    """20B-token / ~80GB recipe. Design only."""
    rows = [
        ("bilingual_textbooks", 0.35, 7_000_000_000, 28_000_000_000, [
            "Cosmopedia v2",
            "FineWeb-Edu score>=4.2",
            "CCI3-HQ textbook subset",
            "rewritten explainer style",
        ], "Apache-2.0 (Cosmopedia v2); ODC-By 1.0 (FineWeb-Edu); verify CCI3-HQ card"),
        ("code_traces", 0.25, 5_000_000_000, 20_000_000_000, [
            "StarCoder2-Edu",
            "Python/Rust/TypeScript with docstrings and passing tests",
        ], "Per-file original licenses; BigCode dataset terms"),
        ("cot_self_correction", 0.20, 4_000_000_000, 16_000_000_000, [
            "OpenMathInstruct-2",
            "traces that contain backtracking language",
        ], "CC-BY-4.0 (OpenMathInstruct-2, confirm the card)"),
        ("dialogue_vernacular", 0.12, 2_400_000_000, 9_600_000_000, [
            "synthetic vernacular dialogue",
        ], "Original synthetic; no officialese"),
        ("memory_interface_honesty", 0.08, 1_600_000_000, 6_400_000_000, [
            "<memory_slot> QA",
            "refuse-when-no-evidence",
        ], "Original synthetic"),
    ]
    components = []
    for name, weight, tokens, nbytes, sources, license_note in rows:
        components.append({
            "name": name,
            "weight": weight,
            "tokens": tokens,
            "bytes_design": nbytes,
            "sources": sources,
            "license": license_note,
        })
    return {
        "name": "pretrain_mix_20b",
        "step": 3,
        "purpose": "Design recipe for a 1.6B seed. This script does not fetch it.",
        "total_tokens": 20_000_000_000,
        "total_bytes_design": 80_000_000_000,
        "download": False,
        "components": components,
    }


def manifest_for(step: int) -> dict:
    if step == 2:
        return step2_manifest()
    if step == 3:
        return step3_manifest()
    raise SystemExit(f"step must be 2 or 3, got {step}")


def is_list_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if PHONE.search(s) and len(re.findall(r"\d", s)) >= 8:
        return True
    if SKU.match(s):
        return True
    if TICKER.match(s):
        return True
    if re.match(r"^[\w\u4e00-\u9fff]{1,8}\s+\d{3}[- ]\d{3,4}[- ]\d{4}$", s):
        return True
    return False


def drop_acausal(text: str) -> bool:
    """True when the document should be discarded."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    listed = sum(1 for ln in lines if is_list_line(ln))
    if listed >= 4 and listed / len(lines) >= 0.5:
        return True
    if COMMUNIQUE.search(text) and listed / len(lines) >= 0.3 and len(lines) >= 4:
        return True
    # A communiqué that is mostly short labeled lines.
    if COMMUNIQUE.search(text):
        short = sum(1 for ln in lines if len(ln) < 40)
        if short / len(lines) >= 0.6 and len(lines) >= 4:
            return True
    return False


_SOCIAL = (
    "lol", "lmao", "thanks", "thank you", "收到", "哈哈", "好的",
    "+1", "同意", "666", "顶", "沙发",
)


def drop_fragment(text: str) -> bool:
    s = text.strip()
    if len(s) >= 100:
        return False
    if s.startswith("#") or s.startswith("//") or s.startswith("/*"):
        return True
    low = s.lower().strip("!！.。 ")
    if low in _SOCIAL:
        return True
    if any(low == item or low.startswith(item + " ") for item in _SOCIAL):
        return True
    return False


def invert_answer_first(text: str) -> str:
    """Move a leading answer behind its reason. Other texts pass through."""
    stripped = text.strip()
    m = ANSWER_EN.match(stripped)
    if m:
        answer = m.group(1).strip()
        reason = m.group(2).strip()
        return f"Because {reason}\nAnswer: {answer}"
    m = ANSWER_ZH.match(stripped)
    if m:
        answer = m.group(1).strip()
        reason = m.group(2).strip().rstrip("。")
        return f"因为{reason}。答案：{answer}。"
    m = CHOICE_BECAUSE.match(stripped)
    if m:
        letter = m.group(1)
        reason = m.group(2).strip().rstrip(".")
        return f"Because {reason}. Therefore {letter}."
    # Already reason-first, or not a textbook item.
    return text


def clean_document(text: str) -> tuple[str | None, str]:
    """Return (cleaned_text or None, reason)."""
    if drop_acausal(text):
        return None, "acausal_list"
    if drop_fragment(text):
        return None, "fragment"
    rewritten = invert_answer_first(text)
    if rewritten != text:
        return rewritten, "inverted"
    return text, "kept"


def write_manifest(path: Path, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_for(step), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_plan(step: int) -> str:
    man = manifest_for(step)
    lines = [
        f"# Fetch plan for step {step} ({man['name']}).",
        "# This file is a plan. prepare_corpus.py does not contact the network.",
        "# Run huggingface-cli yourself only after you accept each dataset card.",
        "",
    ]
    for comp in man["components"]:
        lines.append(f"# {comp['name']} weight={comp['weight']} license={comp['license']}")
        for src in comp["sources"]:
            lines.append(f"#   source: {src}")
        lines.append("")
    if step == 3:
        lines.append("# Example, not executed here:")
        lines.append("# huggingface-cli download HuggingFaceTB/cosmopedia-v2 --repo-type dataset")
        lines.append("# huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset")
        lines.append("# huggingface-cli download nvidia/OpenMathInstruct-2 --repo-type dataset")
    else:
        lines.append("# Example, not executed here:")
        lines.append("# huggingface-cli download HuggingFaceTB/cosmopedia-v2 --repo-type dataset --include 'data/train*' ")
    lines.append("")
    return "\n".join(lines)


def run_fixtures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for path in sorted(FIXTURES.glob("*.txt")) + sorted(FIXTURES.glob("*.py")):
        raw = path.read_text(encoding="utf-8")
        cleaned, reason = clean_document(raw)
        record = {"source": path.name, "reason": reason, "kept": cleaned is not None}
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if cleaned is not None:
            (out_dir / path.name).write_text(cleaned, encoding="utf-8")
            kept += 1
    print(f"fixtures cleaned into {out_dir} ({kept} kept)")


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"self-check failed: {msg}")


def self_check() -> None:
    phone = (FIXTURES / "drop_phonebook.txt").read_text(encoding="utf-8")
    comment = (FIXTURES / "drop_comment.txt").read_text(encoding="utf-8")
    story = (FIXTURES / "keep_story.txt").read_text(encoding="utf-8")
    zh = (FIXTURES / "invert_zh.txt").read_text(encoding="utf-8")
    en = (FIXTURES / "invert_en.txt").read_text(encoding="utf-8")
    choice = (FIXTURES / "invert_choice.txt").read_text(encoding="utf-8")
    code = (FIXTURES / "keep_code.py").read_text(encoding="utf-8")
    memory = (FIXTURES / "keep_memory_slot.txt").read_text(encoding="utf-8")

    _expect(clean_document(phone)[1] == "acausal_list", "phone book should drop")
    _expect(clean_document(comment)[1] == "fragment", "short comment should drop")
    _expect(clean_document(story)[1] == "kept", "story should stay")
    _expect("因为" in story and clean_document(story)[0] == story, "story already reason-first")

    zh_out, zh_why = clean_document(zh)
    _expect(zh_why == "inverted" and zh_out is not None, "zh invert")
    _expect(zh_out.index("因为") < zh_out.index("答案"), "reason before answer (zh)")

    en_out, en_why = clean_document(en)
    _expect(en_why == "inverted" and en_out is not None, "en invert")
    _expect(en_out.lower().index("because") < en_out.lower().index("answer"), "reason before answer (en)")

    ch_out, ch_why = clean_document(choice)
    _expect(ch_why == "inverted" and ch_out.startswith("Because"), choice)
    _expect(ch_out.rstrip(".").endswith("C") or "Therefore C" in ch_out, "choice letter last")

    _expect(clean_document(code)[1] == "kept", "docstring sample should stay")
    _expect(clean_document(memory)[1] == "kept", "memory slot should stay")
    _expect(not drop_fragment(story), "narrative is not a fragment")

    committed = {
        2: ROOT / "configs" / "step2_mix.json",
        3: ROOT / "configs" / "pretrain_mix_20b.json",
    }
    for step, path in committed.items():
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        live = manifest_for(step)
        _expect(on_disk["name"] == live["name"], f"{path} name")
        _expect(
            [c["name"] for c in on_disk["components"]] == [c["name"] for c in live["components"]],
            f"{path} components",
        )
        _expect(abs(sum(c["weight"] for c in on_disk["components"]) - 1) < 1e-9, f"{path} weights")

    for step in (2, 3):
        man = manifest_for(step)
        weights = [c["weight"] for c in man["components"]]
        _expect(abs(sum(weights) - 1.0) < 1e-9, f"step {step} weights")
        if step == 3:
            _expect(sum(c["tokens"] for c in man["components"]) == 20_000_000_000, "20B")
            _expect(man["total_bytes_design"] == 80_000_000_000, "80GB design")
        _expect(man["download"] is False, "download flag default")

    plan = download_plan(3)
    _expect("huggingface-cli" in plan and "does not contact the network" in plan, "plan")
    _expect("http" not in plan.split("huggingface-cli")[0], "plan header has no URL fetch")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        run_fixtures(out)
        phone_rec = json.loads((out / "drop_phonebook.json").read_text(encoding="utf-8"))
        story_rec = json.loads((out / "keep_story.json").read_text(encoding="utf-8"))
        _expect(phone_rec["kept"] is False, "fixture phone")
        _expect(story_rec["kept"] is True, "fixture story")
        inverted = (out / "invert_zh.txt").read_text(encoding="utf-8")
        _expect(inverted.index("因为") < inverted.index("答案"), "written invert")
    print("prepare_corpus self-check ok")


def main() -> None:
    p = argparse.ArgumentParser(description="Corpus prep for Step 2/3. No downloads by default.")
    p.add_argument("--step", type=int, choices=(2, 3), default=2)
    p.add_argument("--dry-run", action="store_true", help="print the manifest; do not fetch")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--download", action="store_true", help="write a fetch plan only; no network")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--fixture-out", type=Path, default=None, help="where to write cleaned fixtures")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    man = manifest_for(args.step)
    if args.manifest:
        write_manifest(args.manifest, args.step)
        print(f"wrote {args.manifest}")
    if args.download:
        plan = download_plan(args.step)
        dest = ROOT / "data" / f"step{args.step}" / "download_plan.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(plan, encoding="utf-8")
        print(plan)
        print(f"wrote {dest}")
        print("no network request was made")
        return
    if args.dry_run or not args.fixture_out:
        print(json.dumps(man, ensure_ascii=False, indent=2))
        if args.dry_run:
            print("dry-run: fixtures not required, network not used")
            return
    if args.fixture_out:
        run_fixtures(args.fixture_out)


if __name__ == "__main__":
    main()
