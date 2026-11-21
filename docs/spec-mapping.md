# Spec Mapping (`Cortex-Local` Unified Architecture)

This repository implements **Cortex-Local** across 5 pure-Rust crates (`crates/`, zero external ML frameworks, zero runtime autograd, zero `cc` build scripts) and an offline PyTorch QAT + distillation toolchain (`tools/`).

## 1. Crate & Tool Mapping

| Design Contract | Implementation |
| --- | --- |
| 1.58-bit `Tile16x4` (16B) & `CacheLine16x16` (`align(64)`, 64B), codes `00->0, 01->+1, 11->-1` | `crates/cortex-kernels/src/tile.rs` |
| Symmetric INT8 dynamic activation quantization in `[-127, +127]` (`round_half_away`, never `-128`) | `crates/cortex-kernels/src/bitlinear.rs` |
| Multiplication-free SIMD GEMV (`x86_64` AVX2 `_mm256_shuffle_epi8` + `_mm256_sign_epi8`, `aarch64` NEON `vqtbl1q_u8` + `vaddl_s8`), `i16 -> i32` promotion every 8 cache lines (128 cols), single row-tail FP32 dequant | `crates/cortex-kernels/src/gemv.rs`, `bitlinear.rs` |
| `O(d_head^2)` Gated DeltaNet rank-1 recurrence (`u = S k`, `S = α S + β(v - α u) k^T`, `o = S q`) + WY lower-triangular chunkwise prefill (`Γ < 1e-3` sequential fallback) | `crates/cortex-ssm/src/deltanet.rs`, `block.rs` |
| Constant-capacity INT8 sliding-window ring attention (`W = 512`, `O(1)` memory over 10,000+ tokens) | `crates/cortex-ssm/src/ring_attn.rs` |
| Online neural hippocampus `m = W2 SiLU(W1 L2Norm(q))`, `0.1×` `W2` cold-start init, L2-normalized `q_t`/`k_t`, analytic outer-product momentum SGD | `crates/cortex-hippocampus/src/mlp.rs` |
| Lock-free double-buffered `SharedWeightArena` with Dekker `SeqCst` reader drain guard + background `AsyncHippocampusHandle` (`PlasticityWorker`) | `crates/cortex-hippocampus/src/arena.rs`, `lib.rs` |
| 3-band surprise gate (`Drop` / `Local` / `High` boosted LR + `WAL1` CRC32 episode log + torn-tail crash recovery + `HPSP` mmap snapshot rollback) | `crates/cortex-hippocampus/src/surprise.rs`, `wal.rs` |
| Zero-copy in-place 3-pass MeZO (`+ε z -> -2ε z -> restore & step`, 2 forwards/step, 0 parameter clones), 80% WAL + 20% anchor `ReplayBuffer`, Gram–Schmidt row orthogonalization & singular suppression | `crates/cortex-sleep/src/mezo.rs`, `replay.rs`, `orthogonalize.rs` |
| Macro-cycle `CortexModel` (`3 × BitSSM + 1 × MemoryBlock`), weight-tied INT8 embedding / LM head, `final_norm`, `.ctxb` (`CTXB`, 64B-aligned tiles, IEEE CRC32) | `crates/cortex-runtime/src/model.rs`, `ctxb.rs` |
| Lossless UTF-8 byte-fallback BPE + 10 dense collision-free atomic control tokens (`<|im_start|>`, `<|thought|>`, `<|tool_call|>`, etc.) | `crates/cortex-runtime/src/tokenizer.rs` |
| Constrained JSON Grammar FSM (`ToolCallFsm::mask_logits`), stop-token truncation, sandboxed 5-tool `AgentLoop`, day/night `RuntimeFsm`, `RwLock` concurrent HTTP/Unix server | `crates/cortex-runtime/src/grammar.rs`, `tools.rs`, `fsm.rs`, `generate.rs`, `api.rs`, `main.rs` |
| L2 sparse micro-expert packages (`.cxex`) & router + Nano/Pro/Max family configs + L3 verified trace cleaner | `crates/cortex-runtime/src/expert.rs`, `family.rs`, `traces.rs`, `docs/l3-self-distillation.md` |
| Offline STE QAT + joint hippocampus read/write meta-training + `.ctxb` exporter + `< 1e-3` Rust logit parity check | `tools/train_seed_qat.py`, `tools/test_ctxb_packing.py`, `tools/step1_train.py` |
| Qwen3 teacher data generation (7 categories incl. 50/20/15/15 tool-use verification), `-100` prompt-masked SFT distillation, DPO/SimPO/GRPO/RLVR recipes | `tools/distill_generate.py`, `tools/distill_train.py`, `tools/distill_qwen.py`, `tools/align_recipes.py`, `tools/test_tooluse_generators.py` |

## 2. Gated DeltaNet Orientation & Complexity

With per-head L2-normalized `q_t` and `k_t` and state `S` stored row-major as `[d_v, d_k]` (`d_v = d_k = d_head`):

```text
u_t = S_{t-1} k_t                                  -- O(d_head^2)
S_t = α_t S_{t-1} + β_t (v_t - α_t u_t) k_t^T     -- O(d_head^2) rank-1 outer product
o_t = S_t q_t                                      -- O(d_head^2)
```

No `I - β_t k_t k_t^T` dense matrix-matrix product (`O(d_head^3)`) is ever formed. Under `β_t = 1` from `S_0 = 0`, `S_1 = v_1 k_1^T` and querying `q_1 = k_1` with `||k_1||_2 = 1` yields `o_1 = (v_1 k_1^T) k_1 = v_1` exactly.

## 3. Model Presets

| Preset | Macros | Layers | `d_model` | Heads × `d_head` | Vocab | Window |
| --- | --- | --- | --- | --- | --- | --- |
| `CortexConfig::prod` | 6 | 24 | 2048 | 32 × 64 | 32768 | 512 |
| `CortexConfig::step1_stories` | 2 | 8 | 512 | 8 × 64 | 4096 | 64 |
| `CortexConfig::micro` | 2 | 8 | 256 | 4 × 64 | 266 | 512 |
| `CortexConfig::test_256` | 2 | 8 | 256 | 8 × 32 | 266 | 512 |
| `CortexConfig::mezo_tiny` | 2 | 8 | 32 | 4 × 8 | 16 | 8 |
| `CortexConfig::mini` | 1 | 4 | 16 | 2 × 8 | 32 | 8 |
