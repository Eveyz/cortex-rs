"""Design-only model presets. Nothing here is a trained checkpoint."""

from __future__ import annotations


def weight_matrix_count(
    n_macros: int,
    d_model: int,
    ffn: int,
    vocab: int,
    hip: int,
    heads: int,
) -> int:
    """Same counting formula as tools/step1_train.py (weight matrices, not Adam state)."""
    emb = vocab * d_model
    ssm = (
        4 * d_model * d_model
        + (2 * d_model * ffn + ffn * d_model)
        + 2 * d_model
        + (2 * heads * d_model + 2 * heads)
    )
    mem = (
        4 * d_model * d_model
        + d_model * d_model
        + (2 * d_model * ffn + ffn * d_model)
        + 4 * d_model
        + 2 * hip * d_model
    )
    return emb + n_macros * 3 * ssm + n_macros * mem


# Measured Step 1 shape. Parameter total including row scales was 25,133,664.
STEP1 = {
    "name": "step1",
    "trained": True,
    "n_macros": 2,
    "layers": 8,
    "d_model": 512,
    "n_heads": 8,
    "head_dim": 64,
    "ffn_hidden": 1120,
    "hip_hidden": 128,
    "window": 64,
    "vocab": 4096,
    "note": "Measured run: 25,133,664 params, loss 8.317→2.689, 1400 steps, 568s, 4 vCPU Xeon.",
}

# Design target only. Not allocated or trained in this repo.
STEP2 = {
    "name": "step2",
    "trained": False,
    "n_macros": 4,
    "layers": 16,
    "d_model": 1024,
    "n_heads": 16,
    "head_dim": 64,
    "ffn_hidden": 3904,
    "hip_hidden": 256,
    "window": 128,
    "vocab": 32768,
    "tie_embeddings": True,
    "design_hours": "8-12 (curriculum design budget, not a measured run)",
    "data": "configs/step2_mix.json",
}


def step2_param_count() -> int:
    return weight_matrix_count(
        STEP2["n_macros"],
        STEP2["d_model"],
        STEP2["ffn_hidden"],
        STEP2["vocab"],
        STEP2["hip_hidden"],
        STEP2["n_heads"],
    )


def describe(argv: list[str]) -> None:
    name = "step2"
    if "--describe-preset" in argv:
        i = argv.index("--describe-preset")
        if i + 1 < len(argv):
            name = argv[i + 1]
    if name == "step1":
        print("preset step1 (measured)")
        for k, v in STEP1.items():
            print(f"  {k} = {v}")
        return
    if name != "step2":
        raise SystemExit(f"unknown preset {name}; choose step1 or step2")
    n = step2_param_count()
    print("preset step2 (design only, not trained, weights not allocated)")
    for k, v in STEP2.items():
        print(f"  {k} = {v}")
    print(f"  weight_matrix_count = {n} ({n / 1e6:.2f}M)")
    print("  vocab plan: 32768 byte-fallback BPE, embedding tied to the LM head")
    print("  continue from a distilled or from-scratch seed only on a machine with GPU:")
    print("    python3 tools/train_seed_qat.py --describe-preset step2")
    print("    # data: python3 tools/prepare_corpus.py --step 2 --dry-run")
    print("    # a real Step 2 fit is not wired to start from this flag")
