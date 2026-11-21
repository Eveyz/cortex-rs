# Cortex-Local

本地、只在 CPU 上跑的个人持续学习运行时。冻结的 1.58-bit 新皮层（Gated DeltaNet + INT8 滑窗）负责推理；海马体是固定尺寸的在线记忆，`θ` 在原位更新，参数量不会随着对话变长。推理二进制不依赖 Python。种子训练和蒸馏才用 PyTorch，而且不在这台机器的默认测试里启动。

这是一份可能尚未公开的草稿。下面没有仓库托管地址，也不要把它当成已经发布的上游。

## 仓库里有什么

```text
crates/cortex-kernels/       纯 Rust 1.58-bit GEMV（标量 / AVX2 / NEON，无 C 编译器）
crates/cortex-ssm/           Gated DeltaNet、INT8 环窗注意力、BitSSM
crates/cortex-hippocampus/   W2 SiLU(W1 q)、惊讶度门控、无锁双缓冲、WAL
crates/cortex-sleep/         原地三步 MeZO、80/20 回放
crates/cortex-runtime/       CortexModel、.ctxb、分词器、cortex-cli
tools/train_seed_qat.py      QAT 与 .ctxb 导出（含已测过的 Step 1）
tools/make_step1_stories.py  Step 1 合成故事
tools/prepare_corpus.py      Step 2/3 配方与三条清洗，默认不下载
tools/distill_qwen.py        Qwen3-28B → 学生 的蒸馏脚手架（默认只用假教师）
tools/presets.py             Step 1 实测形状 / Step 2 设计形状
configs/step2_mix.json       约 2GB 的设计配方
configs/pretrain_mix_20b.json 20B token / 约 80GB 的设计配方
docs/step1.md                Step 1 实测记录
docs/spec-mapping.md         规范到代码的对照
```

状态 `S` 的形状是 `[heads, head_dim, head_dim]`，不随序列变长。海马体权重有固定的 `d_in / d_hidden / d_out`。快路径只读当前快照，不拿互斥锁。

## 构建和本机推理

```bash
cargo test --workspace
cargo build --release -p cortex-runtime

cargo run --release -p cortex-runtime --bin cortex-cli -- init-mini --out /tmp/mini.ctxb
./target/release/cortex-cli generate \
  --model /tmp/mini.ctxb --prompt "hello" --max-tokens 32 \
  --temperature 0.8 --top-p 0.9 --repetition-penalty 1.1 --seed 1
./target/release/cortex-cli /status --model /tmp/mini.ctxb
./target/release/cortex-cli /sleep --model /tmp/mini.ctxb
./target/release/cortex-cli /reset-mem --model /tmp/mini.ctxb
./target/release/cortex-cli serve --model /tmp/mini.ctxb --bind 127.0.0.1:43123
```

`/sleep` 做一步 MeZO（只动 row scale 和 RMSNorm 的 γ）。`/reset-mem` 把海马体权重放回确定性初值。流式接口是 `POST /v1/chat/completions`，`"stream": true`。

内核是纯 Rust。没有 `cc` 构建脚本。没有 AVX2 的机器走标量 GEMV，`test_simd_bitexact` 仍要求它和标量一致。

## Step 1（已经跑过）

做法和数字写在 [docs/step1.md](docs/step1.md)。这里只重复实测，不外推：

- 参数量 **25,133,664**（2 个 macro、8 层、`d_model = 512`、词表 4096）
- 交叉熵 **8.317 → 2.689**，**1400** 步，**568 秒**，4 vCPU Xeon，无 GPU
- 语料 **60,000,814** 字节、**80** 本合成童书（`tools/make_step1_stories.py`，不是出版书）
- 续写语法通顺，但是在复读模板，不会接着「小猫……它」把动作写完

重跑：

```bash
cargo build --release -p cortex-runtime
python3 tools/train_seed_qat.py --step1 --minutes 25 --max-steps 1400 \
  --out artifacts/step1/step1.ctxb
./target/release/cortex-cli generate \
  --model artifacts/step1/step1.ctxb --vocab data/step1/merges.txt \
  --prompt "一天早上，小猫看见桌子上有一碗鱼汤，它" \
  --max-tokens 40 --temperature 0 --repetition-penalty 1.15 --seed 1
```

`artifacts/*.ctxb` 和 `data/step1/corpus.txt` 不进 git。

## Step 2 和 Step 3 的数据（不在这里训练）

Step 2 的目标是：大约 300M 的学生、大约 2GB 双语教材加基础 Python，用来做短问答、简单解释和小脚本。8–12 小时是课程里的设计预算，不是这次测到的时间。对应形状由预设打印，权重矩阵大约 2.993 亿，词表按 32768、嵌入和 LM 头绑定来设计。没有 300M 的训练记录。

Step 3 的目标是：规范里的 20B token、约 80GB 五路配比，用来准备 1.6B 种子。本仓库只有配方，没有这 80GB。

| 份额 | 设计 token | 来源 |
| --- | --- | --- |
| 35% | 7.0B | Cosmopedia v2、FineWeb-Edu（分数 ≥ 4.2）、CCI3-HQ |
| 25% | 5.0B | StarCoder2-Edu / 带 docstring 且测试通过的代码 |
| 20% | 4.0B | OpenMathInstruct-2 和带回溯的推理轨迹 |
| 12% | 2.4B | 白话对话（合成） |
| 8% | 1.6B | `<memory_slot>` 和没有证据就拒答 |

三条清洗都在 `tools/prepare_corpus.py`：无因果罗列（电话本、SKU、行情、通稿）丢掉；短于 100 字的注释和脱离上下文的回复丢掉；「先答案后因为」改成先写理由。

```bash
python3 tools/prepare_corpus.py --self-check
python3 tools/prepare_corpus.py --step 2 --dry-run
python3 tools/prepare_corpus.py --step 3 --dry-run
python3 tools/prepare_corpus.py --step 2 --fixture-out /tmp/step2-clean
python3 tools/prepare_corpus.py --step 3 --download
```

`--download` 只把 `huggingface-cli` 示例写进 `data/step3/download_plan.txt`，不会发网络请求。默认测试不要加这个旗标。许可证以各数据集卡片为准：Cosmopedia v2 常见为 Apache-2.0，FineWeb-Edu 为 ODC-By 1.0，OpenMathInstruct-2 为 CC-BY-4.0（以卡片为准），代码语料保留每个文件自己的许可证，CCI3-HQ 必须先读 BAAI 的说明。

## 从 Qwen3-28B 蒸馏

教师是官方仓库名 `Qwen/Qwen3-28B`（以模型卡上的变体名为准）。学生是 `tools/train_seed_qat.py` 里的 QAT Cortex，导出 `.ctxb` 给 `cortex-cli`。两边词表不一样：教师是十万级 id，学生是 4096 或设计中的 32768 字节回退 BPE，没有 UNK。脚本不把两套 logit 当成同一个向量。

对齐方式：同一段原文；序列级用教师生成的字符串、再用学生自己的分词做交叉熵；token 级把教师概率按「该 token 解码成的字符串」均分到学生 id 上，然后再做 KL。

```bash
python3 tools/distill_qwen.py --self-check
python3 tools/train_seed_qat.py --describe-preset step2
```

`--self-check` 用假教师，损失要下降，并调用现有的 `export_ctxb` 写出带 CRC 的 CTXB。它不下载 28B。

真正蒸馏要自己准备本地权重，脚本发现路径不存在会直接退出，不会代为下载：

```bash
python3 tools/distill_qwen.py --teacher-path /local/Qwen3-28B --manifest configs/step2_mix.json
```

当前这份代码即使路径存在也拒绝拉起 28B（见脚本末尾的说明）。bf16 的 28B 放不进一张 24GB 显卡；多卡或你自己做好的量化权重才是可行的加载方式。先读 Qwen 模型卡上的许可证。蒸馏或从零 QAT 需要 GPU 和 PyTorch。板上的推理和海马体更新只走 Rust。

Step 2 规模的训练入口还没有接上大循环。可以看设计形状，不要把它当成已经跑过：

```bash
python3 tools/train_seed_qat.py --describe-preset step2
```

已实现、并且对过 Rust logits 的小导出是：

```bash
python3 tools/train_seed_qat.py --self-check --steps 2 --joint-hippocampus \
  --rust-bin target/release/cortex-cli --out /tmp/seed.ctxb
```

门槛是最大绝对误差 `< 1e-3`。这是 16 维的迷你网，不是 1.6B。

## 设备上的演进和工具（v0.4，只写了代码）

细节在 [docs/evolution.md](docs/evolution.md)。L1 海马体不能打进可分享的专家包。L2 路由只激活选中的槽，有槽位数和磁盘上限，睡眠路径只搭一个未训练的适配器骨架（目标体积 20–50 MB 这一档）。L3 从通过校验的轨迹里丢掉失败绕路。Nano / Pro / Max 是未训练的形状：16 GB 对 Nano 宽裕，对 Pro 更紧，对 Max 很紧。40 tok/s 以上是目标，不是测量。

```bash
python3 tools/distill_qwen.py --self-check
python3 tools/align_recipes.py --self-check
python3 tools/traces.py --self-check
./target/release/cortex-cli export-expert --domain math --license MIT --d-model 512 --vocab 4096 --out /tmp/math.cxex
./target/release/cortex-cli install-expert --pack /tmp/math.cxex --dir /tmp/experts --d-model 512 --vocab 4096
./target/release/cortex-cli list-experts --dir /tmp/experts
./target/release/cortex-cli generate --model /tmp/mini.ctxb --prompt "hello" --tools tools.txt --max-tokens 8
```

`tools.txt` 每行一个工具：`add a,b`。指令数据和工具轨迹的配方在 `configs/instruct_mix.json` 与 `configs/tool_mix.json`，权重是 50/20/15/15 的设计比例，没有下载。

## 训练和推理

| | 种子训练 / 蒸馏 | 板上推理 |
| --- | --- | --- |
| 运行时 | Python + PyTorch | `cortex-cli`（无 Python） |
| 设备 | 需要 GPU 才能做 Step 2/3 或 28B 教师 | CPU |
| 产物 | `.ctxb` | 读 `.ctxb`，海马体原地更新 |

## 诚实边界

已经实现并且有测试或有实测的：三值 GEMV、Gated DeltaNet、环窗注意力、海马体绑定、MeZO、字节回退 BPE、`.ctxb`、Step 1 的那一次 CPU 训练、清洗器和假教师蒸馏自检。

规范里的设计目标，不是本仓库的测量结果：80 tok/s 以上、海马体更新小于 2 ms、1.6B 打包成 380MB、H100 上 3–4 天、以及任何 M3 的 tok/s。Step 1 的续写也没有达到「既合语法又合物理」——它只是在复读模板。

迷你图在本机 release 上大约 7–8 µs/token、RSS 约 2.9 MB（1 macro，`d_model = 16`）。Step 1 检查点大约 3.1 ms/token、RSS 约 18 MB。两句都只描述那两个具体模型。

## 不要提交的东西

`.gitignore` 排除了 `*.ctxb`、`artifacts/`、Step 1 全量语料、`tokens.u32`、以及 `data/step2/download_plan.txt`。教师权重和 2GB/80GB 语料也不要放进 git。

## English

Cortex-Local is a CPU inference runtime: a frozen 1.58-bit trunk plus a fixed-size hippocampus updated in place. Python is only for offline data prep, QAT, and distillation scaffolding.

```bash
cargo test --workspace
cargo build --release -p cortex-runtime
python3 tools/prepare_corpus.py --self-check
python3 tools/distill_qwen.py --self-check
python3 tools/train_seed_qat.py --describe-preset step2
```

Measured Step 1 facts, and only these: 25,133,664 parameters; loss 8.317→2.689 over 1400 steps in 568 s on a 4 vCPU Xeon; 60,000,814 bytes / 80 synthetic books; generation is grammatical and template-echoing. Step 2 (~300M, ~2GB, 8–12 h) and Step 3 (20B tokens, ~80GB, 1.6B seed) are recipes. Qwen3-28B distillation is specified in `tools/distill_qwen.py` and is not executed here. This draft does not claim a public clone URL.

## License

MIT. 见 `LICENSE`。第三方语料和 Qwen 权重另看各自的许可证，不因本仓库的 MIT 而改变。
