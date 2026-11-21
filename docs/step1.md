# Step 1 — 25M toy Cortex on simple Chinese stories

This is the first curriculum rung: a small 1.58-bit model, a 4096-id
byte-fallback tokenizer, and continuous children's-story prose. It is not the
300M or 1.6B rung, and it is not the `d_model = 16` init-mini toy.

## Shape

| Field | Value |
| --- | --- |
| `N_MACROS` | 2 |
| layers | 8 (3 BitSSM + 1 memory block, twice) |
| `d_model` | 512 |
| heads / head dim | 8 / 64 |
| FFN hidden | 1120 |
| hippocampus hidden | 128 |
| ring window | 64 |
| vocab | 4096 (256 byte ids + 3840 BPE merges) |
| weight matrices | 25,094,240 (row scales add a few tens of thousands) |

The Rust side of this shape is `CortexConfig::step1_stories`. It does not
allocate the weights. Training lives in PyTorch and exports a `.ctxb` the
runtime can load.

## Data

`tools/make_step1_stories.py` writes original template stories (80 books).
It is not a copy of a published book. The fish-soup sentence used as the
generation prompt appears inside that synthetic text so the prompt is
in-distribution. A one-page sample is committed at
`data/step1/fixture_story.txt`. The full ~60MB corpus is generated locally
and is gitignored.

## Rerun

```bash
cargo build --release -p cortex-runtime
python3 tools/train_seed_qat.py --step1 \
  --minutes 22 --max-steps 400 --batch 4 --lr 1.5e-3 \
  --out artifacts/step1/step1.ctxb

./target/release/cortex-cli generate \
  --model artifacts/step1/step1.ctxb \
  --vocab data/step1/merges.txt \
  --prompt "一天早上，小猫看见桌子上有一碗鱼汤，它" \
  --max-tokens 40 --temperature 0.8 --top-p 0.9 --repetition-penalty 1.1 --seed 1
```

`--step1` builds the corpus if needed, trains the 4096 BPE on a sample of the
corpus, encodes the full text with `cortex-cli encode` (same merge rules as
generation), then runs AdamW QAT until the time budget or the step cap.
Loss lines are appended to `artifacts/step1/step1.loss.jsonl`.

The forward is the same block order as Rust: RMSNorm, 1.58-bit QKV, Gated
DeltaNet, SwiGLU, then the memory block (INT8 causal window attention,
`W2 SiLU(W1 q)`, gated residual, SwiGLU). BitLinear uses a straight-through
estimator. The exported file stores ternary tiles plus per-row scales, INT8
tied embeddings, and FP32 hippocampus weights.

## Run on this machine (2026-11-21)

Hardware: 4 vCPU Intel Xeon under KVM, AVX2, no GPU, no swap. PyTorch 2.15 CPU.

| Item | Value |
| --- | --- |
| parameters (`numel`, including row scales and gates) | 25,133,664 |
| weight-matrix formula | 25,094,240 |
| corpus | 60,000,814 bytes, 80 synthetic books |
| BPE fit | 399,882-byte sample, 3840 merges, 18 s |
| corpus encode | 1,636,315 tokens, 143 s (`cortex-cli encode`) |
| optimizer | AdamW, lr 1.5e-3, weight decay 0.01, grad clip 1 |
| batch × seq | 4 × 64 |
| steps | 1400 (hit the step cap; 25 min budget was not exhausted) |
| train wall time | 568 s (~9.5 min), about 0.4 s/step |
| loss | 8.317 → 2.689 nats (cross-entropy) |
| checkpoint | `artifacts/step1/step1.ctxb` (8.9 MB, gitignored) |
| merges | `data/step1/merges.txt` |

Loss along the way: step 1 = 8.317, 10 = 7.184, 50 = 6.200, 100 = 4.904,
200 = 3.759, 400 = 3.433, 800 = 3.070, 1200 = 2.674, 1400 = 2.689.
Initial loss matches `ln(4096)`.

Generation (release binary, same host), prompt
`一天早上，小猫看见桌子上有一碗鱼汤，它`, 40 new tokens, about 3.1 ms/token,
RSS about 18 MB. Greedy (`temperature 0`, repetition penalty 1.15) continuation,
copied from `cortex-cli` and not rewritten:

```text
第42章
一天早上，小青蛙来到井边，看见一壶热水。小青蛙小心翼翼地伸手。邻居也在旁边，安静地看着。小青蛙说：「看，天晴了。」于是小青蛙和邻居一起把东西抱在怀里，把地方收拾干净。风从操场吹过来，叶子轻轻响。小青蛙记得要把一壶热水放回原来的地方。
春天的中午，小青蛙来到山坡，看见一张图画。小青蛙停下来看了很久。爷爷也在旁边，安静地看着。小青蛙说：「先洗手再吃。」于是小青蛙和爷爷一起先想了想，和朋友一起回家。风从井边吹过来，叶子轻轻响。小青蛙记得要把一只纸船放回原来的地方。
暑假的第一天，小青蛙来到河边，看见一条围巾。小青蛙绕着圈跑了两步。爷爷也在旁边，安静地看着。小青蛙说：「我们慢慢来。」于是小青蛙和爷爷一起问旁边的朋友，把看见的事告诉了朋友。风从森林吹过来，叶子轻轻响。小青蛙记得要把一壶热水放回原来的地方。
放学以后，小青蛙来到菜地，看见一只风筝。小青蛙先想了想。奶奶
```

The sentence is grammatical template prose. It does not keep the kitten or
finish “它” with a verb; it jumps to a new chapter about 小青蛙. That matches
the synthetic books, which are built from a small set of sentence frames.
A sampled decode (`temperature 0.8`, `top-p 0.9`, penalty 1.1, seed 1) does
the same kind of thing and happens to mention 一碗鱼汤, still attached to 小青蛙
rather than 小猫.

3.1 ms/token is this 25M checkpoint only.

## Still missing for Step 2

Step 2 in the curriculum is the ~300M rung on a few gigabytes of broader text.
This run does not start that. Still missing: a real (non-template) corpus,
a wider model, a training budget measured in epochs rather than minutes, and
any evaluation that is not “can it echo the story frames.” The 1.6B shape
stays a config constant.

## What this is not

No GPU, no 300M run, no 1.6B run, no claimed 80 tok/s. A short CPU run on
repetitive synthetic stories will not produce a fluent novel. The generation
transcript in the run notes is the model output, copied as-is.
