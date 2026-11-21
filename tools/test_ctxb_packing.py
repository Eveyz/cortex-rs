#!/usr/bin/env python3
"""Cross-check Python `CacheLine16x16` / `Tile16x4` packing against Rust `dump_tiles`.

Requires NumPy only (AST-extracts `pack_ternary` & `pack_tiles` from
`tools/train_seed_qat.py` without importing PyTorch) and verifies 100%
byte-for-byte parity across aligned and unaligned shapes.
"""

from __future__ import annotations

import ast
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "train_seed_qat.py"


def load_pack_tiles():
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = ("round_half_away_np", "pack_ternary", "pack_tiles")
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    ns = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), ns)
    return ns["pack_tiles"]


def main() -> int:
    pack_tiles = load_pack_tiles()
    rng = random.Random(7)
    for rows, cols in [(37, 523), (20, 10), (16, 4), (1, 1), (33, 17)]:
        w = [[rng.choice([-1, 0, 1]) for _ in range(cols)] for _ in range(rows)]
        flat = [v for row in w for v in row]
        inp = "\n".join(str(v) for v in flat) + f"\n{rows} {cols}\n"
        res = subprocess.run(
            ["cargo", "run", "-q", "-p", "cortex-kernels", "--example", "dump_tiles"],
            input=inp,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if res.returncode != 0:
            print("rust dump_tiles failed:", res.stderr, file=sys.stderr)
            return 1
        rust_hex = res.stdout.strip()
        py_bytes = pack_tiles(np.array(w, dtype=np.int8))
        py_hex = py_bytes.hex()
        if rust_hex != py_hex:
            print(
                f"MISMATCH {rows}x{cols}: rust {rust_hex[:64]}... py {py_hex[:64]}...",
                file=sys.stderr,
            )
            return 1
        print(f"pack_tiles {rows}x{cols}: OK ({len(py_bytes)} bytes)")
    print("ALL PACKING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
