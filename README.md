# Cortex-Local (`cortex-rs`)

轻量级个人本地持续学习 AI 框架：**冻结 1.58-bit 三值新皮层（Gated DeltaNet + INT8 环形滑动窗口注意力）+ 无锁双缓冲在线可塑神经海马体 + 夜间零阶 MeZO 睡眠巩固**。

推理与在线持续学习全程运行于纯标准库级 Rust 核心（无 `candle`、`tch`、`ort`、`burn`，无 `cc` 构建脚本，无运行时计算图/Autograd）：
- **在线学习（白天）**：手推解析外积动量梯度 + 三级惊奇度门控 + 无锁 `SharedWeightArena`（Dekker `SeqCst` 读者排空保护）+ `WAL1` 崩溃恢复日志；
- **睡眠巩固（夜间）**：单种子三步原地扰动 MeZO（`+εz -> -2εz -> 还原并更新`，零参数拷贝、严格每步 2 次前向）+ 80% 高惊奇 WAL / 20% 锚点混合重放 + Gram–Schmidt 行正交去重与 `HPSP` 多宏快照回滚；
- **训练与蒸馏（离线 Python）**：`tools/` 提供 PyTorch STE 量化感知训练（QAT）、海马体读写接口联合元训练、Qwen3 7 类可验证蒸馏生成与 `-100` Prompt Masking SFT 导出 `.ctxb`（跨语言前向 Logits 误差 `< 1e-3`）。

---

## 目录结构

```text
cortex-rs/
├── Cargo.toml                         # Rust Workspace（5 大核心 Crate）
├── crates/
│   ├── cortex-kernels/                # 1.58-bit Tile16x4 / CacheLine16x16(align 64)、对称 INT8[-127,127]、
│   │                                  # 真 AVX2(_mm256_shuffle_epi8) / NEON(vqtbl1q_u8 + vaddl_s8) 无乘法 GEMV
│   ├── cortex-ssm/                    # O(d_head²) Gated DeltaNet 秩-1 递推 + WY 分块并行 Prefill + INT8 环形窗口注意力(W=512)
│   ├── cortex-hippocampus/            # 双层 SiLU 海马体(0.1x W2 冷启动、L2 归一化 q/k)、无锁双缓冲 Arena、WAL1 + HPSP 快照
│   ├── cortex-sleep/                  # 零拷贝原地 MeZOScaleTuner、80/20 ReplayBuffer、Gram-Schmidt 正交去重、L2 骨架
│   └── cortex-runtime/                # 3×BitSSM + 1×MemoryBlock 宏周期模型、权重绑定 INT8 词表、无 UNK 字节回退 BPE +
│                                      # 10 个紧凑原子控制 Token、受限 JSON Grammar FSM、5 工具沙箱 AgentLoop、昼夜 FSM、CLI/HTTP
├── tools/
│   ├── train_seed_qat.py              # STE QAT + 海马体读写接口联合训练 + 64B 对齐 .ctxb 导出 + <1e-3 Rust 对齐自检
│   ├── test_ctxb_packing.py           # 无需 PyTorch：Python CacheLine16x16 位打包 vs Rust dump_tiles 逐字节校验
│   ├── step1_train.py                 # Step 1 (~25.1M) 中文故事合成语料 QAT 与跨窗口(T > W)对齐验证
│   ├── make_step1_stories.py          # Step 1 合成语料生成器
│   ├── distill_generate.py            # Qwen3 教师 7 大类蒸馏数据生成（含 50/20/15/15 四类工具调用真实沙箱验证）
│   ├── distill_train.py               # 序列级 SFT 蒸馏训练（Prompt Loss Masking = -100）+ .ctxb 导出
│   ├── distill_qwen.py                # 跨词表字节投影 KL + MOHAWK 隐藏态换血蒸馏脚手架
│   ├── test_tooluse_generators.py     # 工具调用数据生成与幻觉清洗单测（9/9 通过）
│   ├── align_recipes.py               # ChatML 控制符、-100 Token Masking、DPO/SimPO、GRPO/RLVR 奖励函数自检
│   ├── prepare_corpus.py              # Step 2/3 语料配比清单与三条因果清洗规则
│   ├── presets.py                     # 模型尺寸预设描述
│   └── traces.py                      # L3 验证轨迹去绕路清洗自检
├── configs/                           # step2_mix.json / pretrain_mix_20b.json / instruct_mix.json / tool_mix.json
└── docs/
    ├── spec-mapping.md                # 核心设计契约到源码文件的一一映射表
    ├── step1.md                       # Step 1 25.1M CPU 实测训练记录
    ├── evolution.md                   # L1/L2/L3 持续演进与微专家包(.cxex)说明
    └── l3-self-distillation.md        # L3 慢思考试错 -> 快直觉自蒸馏完整架构设计
```

---

## 快速构建与验证

### 1. Rust 全量测试与端到端 CLI

```bash
cargo test --workspace --release
cargo build --workspace --examples --release

# Micro (8 层, d_model=256) 单步延迟与常驻内存基准
./target/release/examples/bench_forward

# 初始化模型并执行流式推理、昼夜睡眠巩固、快照回滚与沙箱 Agent
./target/release/cortex-cli init-mini --shape micro --out /tmp/micro.ctxb
./target/release/cortex-cli generate \
  --model /tmp/micro.ctxb --prompt "你好 Cortex — 龘靐齉" --max-tokens 32 \
  --temperature 0.8 --top-p 0.9 --repetition-penalty 1.1 --seed 1
./target/release/cortex-cli /status --model /tmp/micro.ctxb
./target/release/cortex-cli /sleep --model /tmp/micro.ctxb
./target/release/cortex-cli /reset-mem --model /tmp/micro.ctxb
./target/release/cortex-cli tools
./target/release/cortex-cli agent --model /tmp/micro.ctxb --prompt "What is 6 * 7?" --workspace /tmp/cortex-work
```

并发本地服务（基于 `RwLock<CortexModel>` + 无锁 `SharedWeightArena`，多路流式推理零全局互斥阻塞）：

```bash
./target/release/cortex-cli serve --model /tmp/micro.ctxb --bind 127.0.0.1:43123
```

### 2. Python 位打包与跨语言 `< 1e-3` Logits 对齐自检

```bash
python3 tools/test_ctxb_packing.py
python3 tools/test_tooluse_generators.py
python3 tools/train_seed_qat.py --config micro --steps 10 --joint-hippocampus \
  --self-check --rust-bin ./target/release/cortex-cli
python3 tools/step1_train.py --self-check
python3 tools/distill_train.py --self-check
python3 tools/align_recipes.py --self-check
python3 tools/distill_qwen.py --self-check
python3 tools/prepare_corpus.py --self-check
python3 tools/traces.py --self-check
```

---

## 核心技术要点

1. **真·SIMD 无乘法 1.58-bit 微内核**：
   - `Tile16x4`（16B）与 `CacheLine16x16`（`#[repr(C, align(64))]`，64B）严丝合缝对齐 CPU 缓存行，2-bit 编码 `00->0, 01->+1, 11->-1`；
   - 动态激活严格对称量化至 `[-127, +127]`（`round_half_away`，杜绝 `-128` 二补数取反溢出）；
   - `x86_64` AVX2 采用 `_mm256_shuffle_epi8` + `_mm256_sign_epi8`，`aarch64` NEON 采用 `vqtbl1q_u8` 掩码查表 + `vaddl_s8` 双列宽加，每 8 个缓存行块（128 列）从 `i16` 无损升位累加至 `i32`，仅在行尾做唯一一次浮点反量化缩放。Apple M3 Pro（NEON）实测 8 层 `micro` 模型单步延迟 **~0.90 ms/token**（> 1,100 tok/s）。
2. **$O(d_{\text{head}}^2)$ Gated DeltaNet + $O(1)$ 环形局部注意力**：
   - 状态 `S` 按 `[d_v, d_k]` 存储，严格按 `u = S k -> S = α S + β(v - α u) k^T -> o = S q` 执行秩-1 更新，并内置 WY 下三角前向代入分块并行 Prefill（`Γ < 1e-3` 自动回退顺序单步）；
   - 第 4 层局部注意力采用固定槽位（生产配置 `W = 512`）的逐头 INT8 环形缓冲区，10,000+ 步长程推理状态内存零增长。
3. **无锁并发在线海马体**：
   - 严格遵循 **“先 L2 归一化读旧记忆 -> 门控残差融合 -> 后台线程异步写新记忆”** 的因果顺序；
   - `W2` 初始化缩放 `0.1×` 保护冷启动残差流；`SharedWeightArena` 使用双槽 `UnsafeCell` + `SeqCst` 活跃读者计数排空保护，彻底消除悬垂指针与全局锁竞争；
   - 高惊奇度样本自动放大在线学习率并异步追加带 IEEE CRC32 校验与断尾恢复的 `WAL1` 日志。
4. **受限解码与沙箱工具调用**：
   - 256 字节无损 UTF-8 回退 BPE + 紧接词表尾部的 10 个无碰撞专用控制 Token（`<|im_start|>`、`<|im_end|>`、`<|thought|>`、`<|tool_call|>` 等）；
   - 生成 `<|tool_call|>` 后自动激活 `ToolCallFsm::mask_logits` JSON 语法状态机掩码非法 Token，遇到 `</|tool_call|>` 或 `<|im_end|>` 精准截断；
   - 内置 `calculator`、`read_file`、`write_file`、`list_dir`、`get_time` 五大路径越狱防护工具与 `AgentLoop`，System Prompt 与 Python 蒸馏器逐字钉死校验。

---

## License

MIT（见 [LICENSE](LICENSE)）。第三方公开数据集与教师模型权重遵循各自原始许可证。
