# Spec mapping

This tree follows the superseding implementation brief. Earlier notes (pure-Rust
kernels, the hippocampus arena, the data-mix addendum) were adapted where they
disagreed with that brief. There is no C toolchain and no autograd inside
`crates/`.

## Crates

| Brief piece | Code |
| --- | --- |
| 1.58-bit tiles, INT8 activations, scalar / AVX2 / NEON GEMV | `crates/cortex-kernels` |
| `Tile16x4` bit order `7..6 = w[r,3]` … `1..0 = w[r,0]`; codes `00/01/11` | `crates/cortex-kernels/src/tile.rs` |
| `CacheLine16x16` = 4 tiles, `align(64)` | `tile.rs` |
| Symmetric INT8 in `[-127, 127]`, promote `i16→i32` every 8 cache lines | `gemv.rs`, `bitlinear.rs` |
| Gated DeltaNet rank-1 update, state `[heads, d, d]` | `crates/cortex-ssm/src/deltanet.rs` |
| BitSSM block (layers 1–3): BitLinear QKV + DeltaNet + Bit SwiGLU | `crates/cortex-ssm/src/block.rs` |
| INT8 ring window attention (branch A, window 512 in the prod config) | `crates/cortex-ssm/src/ring_attn.rs` |
| Hippocampus `m = W2 SiLU(W1 q)`, `W2` init × 0.1 | `crates/cortex-hippocampus/src/mlp.rs` |
| `Surprise = (1/(2d)) ‖m − v‖²`, drop / local / high | `surprise.rs`, `lib.rs` `inject` |
| `SharedWeightArena`: two `UnsafeCell` slots, `active_idx`, `active_readers[2]` | `arena.rs` |
| WAL + mmap snapshot rollback | `wal.rs` (`memmap2`) |
| MeZO 3-pass, one PRNG seed, φ = row scales + RMS weights | `crates/cortex-sleep/src/mezo.rs` |
| Replay 80% WAL / 20% anchors | `crates/cortex-sleep/src/replay.rs` |
| Macro model, tied INT8 embedding / LM head | `crates/cortex-runtime/src/model.rs` |
| `.ctxb` (magic `CTXB`, CRC32, packed tiles) | `ctxb.rs` |
| Byte-fallback BPE, default 256-byte vocab | `tokenizer.rs` |
| Temperature, top-p, repetition penalty | `sample.rs` |
| `cortex-cli` `/status` `/sleep` `/reset-mem`, streaming generate | `main.rs`, `generate.rs` |
| Local HTTP + Unix socket | `api.rs` |
| QAT STE, joint hippocampus, `export_to_ctxb`, logit self-check | `tools/train_seed_qat.py` |
| Step 1 measured 25.1M run | `tools/step1_train.py`, `docs/step1.md` |
| Step 2/3 manifests and the three cleaners | `tools/prepare_corpus.py`, `configs/` |
| Qwen3-28B distill scaffold (mock teacher by default) | `tools/distill_qwen.py` |

## Recurrence actually implemented

After per-head L2-normalization of `q` and `k`:

```text
u = S_{t-1} k_t
S_t = α_t S_{t-1} + β_t (v_t − α_t u) k_t^T
o_t = S_t^T q_t
```

which is the same as

```text
S_t = α_t S_{t-1} (I − β_t k_t k_t^T) + β_t v_t k_t^T
```

`α` and `β` are per-head sigmoids of a light FP32 projection. `q, k, v, o`
are `BitLinearWeights`. State is caller-owned and does not grow with `T`.

## Macro cycle

| Config | Macros | Layers | d_model | Window |
| --- | --- | --- | --- | --- |
| `CortexConfig::prod` | 6 | 24 | 2048 | 512 |
| `CortexConfig::test_256` | 2 | 8 | 256 | 512 |
| `CortexConfig::mezo_tiny` | 2 | 8 | 32 | 8 |
| `CortexConfig::mini` | 1 | 4 | 16 | 8 |

Layer 4 of each macro is branch A (INT8 ring) + branch B (hippocampus read of a
BitLinear query) added through a diagonal sigmoid gate, then a 1.58-bit SwiGLU.
The prod shape is a config constant. Tests do not allocate it.

## DoD tests

| Test | Where |
| --- | --- |
| `test_simd_bitexact` (37×523, err < 1e-5) | `cortex-kernels` |
| `test_deltanet_constant_memory` (2048 steps) | `cortex-ssm` |
| `test_oneshot_hippocampus_binding` (MSE drop > 80%, repeat does not bump version) | `cortex-hippocampus` |
| `test_mezo_loss_descent` (8-layer probe, 30 steps) | `cortex-runtime` |
| `test_end_to_end_cli_generation` (32 streamed tokens, latency + RSS) | `cortex-runtime/tests/e2e.rs` |

## Explicitly not in this tree

* A trained 1.6B or 120M checkpoint, and any downloaded corpus.
* A 32,768-token trained BPE. The tokenizer is byte-level plus optional merges.
* GPU kernels, autograd, or a Python runtime on the inference path.
* Measured throughput of the production shape. Numbers in the README, if any,
  are for the mini model on the machine that ran them.
