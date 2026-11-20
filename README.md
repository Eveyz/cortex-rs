# cortex-local

CPU-only personal runtime for a frozen 1.58-bit trunk (Gated DeltaNet), an
INT8 sliding-window attention branch, and an online two-layer hippocampus.
The inference binary is Rust. Python exists only as the offline QAT exporter
in `tools/train_seed_qat.py`.

This repository does **not** ship a trained 1.6B model and does not claim a
token rate for that model. The production *shape* (`6` macros, `24` layers,
`d_model = 2048`, ring window `512`) is a config constant. What runs in tests
and in `cortex-cli` is a mini or tiny instance of the same block layout.

## Layout

```text
Cargo.toml
crates/cortex-kernels/        pure-Rust 1.58-bit GEMV (scalar, AVX2, NEON)
crates/cortex-ssm/            Gated DeltaNet, ring attention, BitSSM blocks
crates/cortex-hippocampus/    W2 SiLU(W1 q), surprise gate, lock-free arena, WAL
crates/cortex-sleep/          in-place 3-pass MeZO, 80/20 replay
crates/cortex-runtime/        CortexModel, .ctxb, tokenizer, cortex-cli
tools/train_seed_qat.py       BitLinear STE + hippocampus joint-train + .ctxb
docs/spec-mapping.md
```

## Build and test

```bash
cargo test --workspace
cargo build --release -p cortex-runtime
```

No C compiler is required. `cargo test` passes on a host without AVX2; the
AVX2 kernel is selected at runtime when the CPU has it, and
`test_simd_bitexact` compares it with the scalar GEMV.

Kernels are pure Rust (`core::arch` intrinsics). There is no `cc` crate.

## What is real

* **Tiles.** `Tile16x4` is 16 bytes: 16 rows × 4 cols. `data[r]` packs
  `w[r,3]` in bits 7..6, `w[r,2]` in 5..4, `w[r,1]` in 3..2, `w[r,0]` in 1..0.
  `00 → 0`, `01 → +1`, `11 → -1`. `10` is never emitted. Four tiles make one
  64-byte `CacheLine16x16`.
* **GEMV.** Symmetric INT8 activations in `[-127, 127]`. Partial sums stay in
  `i16` and promote to `i32` every 8 cache-line blocks (128 columns;
  `128 * 127 = 16256`). The row scale is applied once at the end.
* **Gated DeltaNet.** Per head, after L2-normalizing `q` and `k`:

  ```text
  u = S k
  S = α S + β (v − α u) kᵀ
  o = Sᵀ q
  ```

  State is `[num_heads, head_dim, head_dim]` and does not grow with time.
  `q, k, v, o` are 1.58-bit projections. `α, β ∈ (0, 1)` come from a small
  FP32 gate.
* **Macro block.** Layers 1–3 are BitSSM (DeltaNet + 1.58-bit SwiGLU). Layer 4
  mixes an INT8 ring (branch A) with `m = W2 SiLU(W1 q)` (branch B) through a
  sigmoid gate, then another SwiGLU. The hippocampus query is a BitLinear, so
  the plastic FP32 memory and the ternary kernels actually call each other.
* **One-shot bind.** A Newton step on `W2` (analytic, no autograd) plus a
  momentum step on `W1`. Surprise below `τ_low` does not increment `version`.
* **Arena.** Two `UnsafeCell` slots, an atomic active index, and per-slot
  reader counts. Fast-path reads do not take a mutex. This is not an
  `AtomicPtr` swap.
* **MeZO.** Three in-place passes regenerate the same Rademacher vector from
  one seed. φ is embedding / BitLinear row scales and RMSNorm weights only.
* **Tokenizer.** 256 byte tokens plus optional BPE merges. Any UTF-8 string
  round-trips. There is no UNK id.
* **CLI sampling.** Temperature, top-p, repetition penalty.

## Commands

```bash
cargo run --release -p cortex-runtime --bin cortex-cli -- init-mini --out /tmp/mini.ctxb

./target/release/cortex-cli generate \
  --model /tmp/mini.ctxb --prompt "hello" --max-tokens 32 \
  --temperature 0.8 --top-p 0.9 --repetition-penalty 1.1 --seed 1

./target/release/cortex-cli /status --model /tmp/mini.ctxb
./target/release/cortex-cli /sleep --model /tmp/mini.ctxb
./target/release/cortex-cli /reset-mem --model /tmp/mini.ctxb

./target/release/cortex-cli serve \
  --model /tmp/mini.ctxb --bind 127.0.0.1:43123 --unix /tmp/cortex.sock
```

`/sleep` runs one MeZO step on a fixed anchor mix and writes the checkpoint
back. `/reset-mem` restores hippocampus weights to the deterministic init.

Streaming chat (SSE), same process:

```bash
curl -N http://127.0.0.1:43123/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":8,"stream":true}'
```

`GET /status`, `POST /sleep`, and `POST /reset-mem` are the same verbs as the CLI.

## QAT export

```bash
cargo build --release -p cortex-runtime
python3 tools/train_seed_qat.py --self-check --steps 2 --joint-hippocampus \
  --rust-bin target/release/cortex-cli --out /tmp/seed.ctxb
```

The script trains a 16×16 BitLinear with a straight-through estimator and, with
`--joint-hippocampus`, a few SGD steps of the memory MLP. It then writes a
mini `.ctxb` and checks Rust logits against its own fp32 forward. The gate is
max abs error `< 1e-3`. Nothing here is a 1.6B run.

## `.ctxb`

Magic `CTXB`, version 1, hyperparameters, embedding table (INT8 + per-row
FP32 scale, tied to the LM head), packed `CacheLine16x16` blobs padded to
64-byte file offsets, hippocampus `W1/W2`, and an IEEE CRC32. See
`crates/cortex-runtime/src/ctxb.rs`.

## Memory

`Aligned64<T>` and `Vec<CacheLine16x16>` are 64-byte aligned. DeltaNet state
and the attention ring are allocated once per session. MeZO does not allocate
a gradient buffer. A short test process stays far under an 800 MB virtual
memory ceiling; the production shape is not allocated by tests. To check a
binary under a limit when the sandbox allows it:

```bash
ulimit -v 800000
cargo test -p cortex-hippocampus --lib
```

If `ulimit` is forbidden, that check is skipped by not running it in CI. The
unit tests themselves do not depend on it.

## Measured here

Host: Linux x86_64, 4 vCPU, Intel Xeon (KVM), AVX2 and AVX-512 present. Release
binary, `CortexConfig::mini` (1 macro, 4 layers, `d_model = 16`, vocab 32),
greedy decode of 32 tokens after the prompt `hello`:

* about 7–8 µs per generated token (`dt_us` from `cortex-cli generate`)
* process RSS about 2.9 MB

That is the toy graph only. It is not a measurement of the 24-layer
`d_model = 2048` shape, and it is not a claim about a trained model.

## 中文

这是一个纯 Rust 的本地推理骨架：1.58-bit 三值 GEMV（无 C 编译器）、三层
Gated DeltaNet、INT8 滑窗注意力，以及在线海马体 `W2 SiLU(W1 q)`。
`cortex-cli` 在本机提供 `/status`、`/sleep`、`/reset-mem` 和流式生成。
Python 只用于离线 QAT 导出。仓库里没有训练好的大模型，也没有虚构的 tok/s。

## License

MIT. See `LICENSE`.
