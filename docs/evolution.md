# Evolution harness (v0.4)

Code and tests only. Nothing here trains a model or downloads a corpus.

## L1 / L2 / L3

L1 is the existing hippocampus. Its snapshot magic is `HPSP` and the WAL magic is `WAL1`. `ExpertPackage::export_refusing_private` rejects any sidecar that contains those bytes, and a normal package does not have a field for them.

L2 is `crates/cortex-runtime/src/expert.rs`. A router mounts a few 1.58-bit adapters and `route` returns only the slot for the requested domain. The slot cap and the disk budget evict the least recently used adapter. `adapter_plan` describes a packed tile buffer in the 20–50 MB band; tests do not allocate that buffer. Sleep scaffolding lives in `crates/cortex-sleep/src/scaffold.rs`. It groups high-surprise logs by domain and returns an `ExpertScaffold` with `trained: false` and a target size inside that band. It does not run an optimizer.

L3 is `crates/cortex-runtime/src/traces.rs` and `tools/traces.py`. A verified episode keeps steps with `ok` and not `detour`. Unverified episodes produce nothing. The Python hook writes JSONL and stops there.

## Family shapes

`crates/cortex-runtime/src/family.rs` names Nano (1.6B, ~380 MB design footprint), Pro (7.5B, ~1.7 GB), and Max (28B total, 4B active, ~6.5 GB on disk). Those disk numbers are the v0.4 design labels. 16 GB of CPU memory is described as comfortable for Nano, tighter for Pro, and tight for Max. 40+ tok/s is a target, not a number this tree measured.

## Distillation

`tools/distill_qwen.py --self-check` runs three mock steps:

1. MSE from a wider teacher hidden state into a narrower student state (shapes differ, so Q/K/V are not copied).
2. A λ schedule from 0 to 1 mixing cross-entropy and KL on a mapped distribution.
3. A frozen trunk linear and a trainable gate, checked by which tensors receive gradients.

The teacher id remains `Qwen/Qwen3-28B`. The script does not download it.

## Tools

Special tokens such as `<|tool_call|>` are scanned before BPE and decode back to the same string. `render_chat` covers system, user, assistant, and tool. `tools/align_recipes.py` masks loss to assistant characters and evaluates DPO, SimPO, and a deterministic GRPO/RLVR reward on dummy numbers. `configs/tool_mix.json` is the 50/20/15/15 recipe. `configs/instruct_mix.json` lists Tulu 3, Infinity-Instruct, and Magpie as cards to confirm, not as files on disk.

`crates/cortex-runtime/src/harness.rs` parses a tool call, checks required arguments, and runs `echo`, `add`, and `read_fixture` (a single file name under a root, no `..`). The grammar in `grammar.rs` masks bytes after a tool call so a dummy logit stream that prefers an illegal byte still emits parseable JSON. That 100% figure is a property of the mask.

CLI: `export-expert`, `install-expert`, `list-experts`, and `generate --tools`. HTTP chat completions accept a `tools` array and report `tools_accepted`.
