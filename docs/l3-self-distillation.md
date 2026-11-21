# L3 自蒸馏设计文档（慢思考试错 → 快直觉蒸馏）

- 状态：设计稿（2026-11-21），未实现
- 决策：两条线并行。A 线（轻量本地线）与 B 线（离线 SFT 线）共享采集、
  验证、提取三层，只在落点上分开。
- 关联文档：`README.md`「路线图」一节；`tools/distill_train.py`（SFT 管线）；
  `crates/cortex-runtime/src/tools.rs`（AgentLoop）；`crates/cortex-sleep`（睡眠引擎）。

## 1. 目标与非目标

**目标**：让本地模型在可验证的任务类上，把“靠多轮工具试错才跑通”的
慢思考，压缩成“一步写对”的快直觉。验收标准是同一类任务的一次成功率
（one-shot success rate）before / after 上升，而不是“记得更多”。

**非目标**：

- 不做通用能力的 SFT（那是 `distill_train.py` 的蒸馏管线，不是 L3）。
- 不碰不可验证的任务（闲聊、开放写作、主观判断）：没有验证器，就没有
  黄金轨迹，进不了 L3。
- 不追求“零人工”：验证器、评估集、回滚都需要人定规则和阈值。
- 不替代 L2：微专家 adapter（L2）是另一个落点，L3 的 SFT 数据可以拿去
  训 adapter，但 L3 本身不依赖 L2。

## 2. 总览：两条线 + 三层共享

```
                    ┌─ 采集 ─┐   ┌─ 验证 ─┐   ┌─ 提取 ─┐
  AgentLoop turn ─▶ │ trace  │─▶│Verifier│─▶│ golden │
                    │ journal│   │ trait  │   │ record │
                    └────────┘   └────────┘   └────┬───┘
                                                    │
              ┌─────────────────────────────────────┼─────────────────────┐
              │ A 线：轻量本地线（每晚睡眠）         │ B 线：离线 SFT 线    │
              │                                     │ （攒批，GPU 机器）   │
              │  ① ReplayBuffer 黄金池（高权重采样） │  ① export_golden.py │
              │  ② 海马体 anchor（情景记忆）        │     导出 JSONL       │
              │  ③ MeZO 调 LM 子空间 γ（已有）      │  ② distill_train.py │
              │  ④ φ 版本快照 + 回滚                │     跑 SFT，出新 ctxb│
              │                                     │  ③ 评估门 → 替换/回滚│
              └─────────────────────────────────────┴─────────────────────┘
```

两条线的分工是刻意的：A 线零 GPU、每晚都跑，负责“记住解法”；
B 线一周一次、跑 GPU，负责“学会技能”。A 线是 B 线的数据上游
（黄金轨迹先攒在 journal 里，B 线批量取用），B 线是 A 线的能力上限
（SFT 后的新 `.ctxb` 替换后，A 线的海马体重放继续在新主干上跑）。

## 3. 数据模型

### 3.1 Trace journal（采集层）

WAL 是二进制、字段固定（seq / surprise / context / k / v），装不下
transcript，所以新开一个文件：**与 WAL 同目录的 JSONL**，一行一条，
人类可读，Python 侧直接读，和蒸馏数据格式同构。

- 路径：`<data_dir>/traces.jsonl`（与 WAL、`cortex-snapshots` 同目录，
  由 CLI 的 `--data-dir` 或默认目录决定；默认 `./cortex-work` 的父目录
  逻辑沿用现有约定）。
- 写入时机：每个 agent turn 结束时同步写（`AgentLoop::run` 返回后，
  由调用方写 journal）。普通生成（非 agent）不写——没有工具轨迹，
  不是 L3 素材。
- 每行 schema：

```json
{
  "ts": "2026-11-21T10:30:00Z",
  "task": "修复 notes/queue.rs 里的死锁，单测通过为止",
  "transcript": "……完整 transcript（含 <tool_call> / <tool_result>）……",
  "answer": "……最终答案……",
  "steps": 6,
  "tool_calls": 9,
  "finished": true,
  "verified": {"verifier": "code", "ok": true, "detail": "cargo test: 12 passed"},
  "surprise": 0.42
}
```

- `verified` 由验证器在 turn 结束时同步填（见 §4）。验证失败或无
  适用验证器的 turn 也写 journal（`verified.ok=false` 或
  `verifier=null`），只作统计，不进提取。
- `surprise` 取该 turn 期间海马体的平均 surprise（已有信号，复用）。

### 3.2 Golden record（提取层输出）

提取器把一条 journal 记录变成一条训练样本，格式与
`distill_train.py` 的输入完全一致（prompt / response / verified /
meta），可直接混进蒸馏数据或单独成文件：

```json
{
  "category": "golden",
  "prompt": "<SYSTEM_PROMPT>\n\nUser: 修复 notes/queue.rs 里的死锁，单测通过为止\nAssistant:",
  "response": "……提炼后的推导 + 最终答案……",
  "verified": true,
  "meta": {
    "source": "l3",
    "task_class": "code",
    "rounds": 6,
    "diagnosis": "锁顺序反了：先拿 A 再拿 B，改成统一顺序",
    "journal_seq": 1234
  }
}
```

- `prompt` 里的 SYSTEM_PROMPT 用 `ToolRegistry::render_system_prompt()`
  的同一份文本（与工具调用数据一致，见 `test_render_system_prompt_exact`）。
- `meta.task_class` 用于按类统计和按类评估（见 §8）。
- `meta.diagnosis` 只在带修复的提炼形式下有值（见 §5.2）。

## 4. 验证器（Verifier）

### 4.1 原则

- 验证器只证“结果对”，不证“推理对”。模型可能瞎蒙对答案
  （测试覆盖不全、数字碰巧对），所以黄金轨迹还要过 §5.3 的
  推导质量关。
- 验证器必须是确定性的、可重跑的、沙箱内的。验证本身不许有
  副作用（不许联网、不许写 workspace 之外的路径）。
- 验证器按任务类注册，不是按工具注册：一个任务可能用多个工具，
  验证看的是最终状态（代码通过测试、文件内容正确、数字算对），
  不是调用序列。

### 4.2 Trait 设计（Rust，`crates/cortex-runtime/src/verify.rs`）

```rust
pub struct VerifyOutcome {
    pub ok: bool,
    pub detail: String,      // 如 "cargo test: 12 passed, 0 failed"
}

pub trait Verifier {
    /// 任务类名，如 "code" / "calc" / "file"。
    fn task_class(&self) -> &'static str;
    /// 这条任务是否适用本验证器（按任务文本 + 调用过的工具判断）。
    fn applies(&self, task: &str, outcome: &AgentOutcome) -> bool;
    /// 在 workspace 内验证最终状态。只读 + 受控执行，不许联网。
    fn verify(&self, task: &str, outcome: &AgentOutcome, workspace: &Path) -> VerifyOutcome;
}

pub struct VerifierRegistry {
    verifiers: Vec<Box<dyn Verifier>>,
    // verify(task, outcome, workspace) -> Option<VerifyOutcome>
    // （无适用验证器时返回 None）
}
```

- `AgentOutcome` 需补一个字段：`task: String`（现在只有 transcript /
  answer / steps / tool_calls / finished，缺任务原文，验证器和
  journal 都需要它）。
- `verify` 的执行预算：单次验证超时 30 秒（code 类跑测试可能慢，
  设 120 秒），超时算验证失败。

### 4.3 三个验证器（实现顺序：calc → file → code）

| 验证器 | 适用条件 | 验证方法 | 备注 |
|---|---|---|---|
| `calc` | turn 里调过 calculator，且任务含数字 | 重算 transcript 里的最后一个 calculator 表达式，与最终答案里的数字比对 | 最简单，先做，跑通整条管线 |
| `file` | turn 里调过 write_file | 重读目标文件，比对内容与最终答案引用的关键行 | 复用 harness 的 jail 逻辑 |
| `code` | 任务含“修复/写代码/单测”等关键词，或调过 write_file 且路径以 `.rs` / `.py` 结尾 | 在 workspace 的临时副本里跑测试命令（`cargo test` / `python -m pytest`），通过才算 ok | 最有价值，也最重：需要沙箱执行器（子进程 + 超时 + 无网络）；先做 Python 的 `run_tests` 思路的 Rust 版，见 `distill_generate.py` 的同名函数 |

顺序的理由：calc 和 file 是纯 Rust、无子进程，先把采集→验证→
提取→A 线落点的整条管线跑通；code 的价值最高（L3 的主战场是
代码任务），但沙箱执行器要单独做，不阻塞管线。

### 4.4 质量关（防瞎蒙）

验证通过后，黄金轨迹还要过两道关才算数：

1. **证据关**：最终答案的推导链必须引用真实工具结果。复用
   `distill_generate.py` 里 `_evidence_ok` 的逻辑（calculator 结果
   出现在终答里、read_file 的关键行被引用、write_file 的路径被提到）。
   引用了幻觉结果的轨迹直接丢掉。
2. **覆盖关（仅 code 类）**：测试必须真的跑过且覆盖了改动的代码。
   最低要求：测试命令退出 0 且“通过数 > 0”；进阶要求（v0.4 后）：
   改动文件在测试里被 import / 调用过。测试数为 0 的“通过”
   （比如跑错了目录）算验证失败。

## 5. 黄金轨迹提取（Golden-trace extraction）

提取器是纯函数：`journal 记录 → Option<GoldenRecord>`，放在
`crates/cortex-sleep/src/golden.rs`（A 线用）和
`tools/export_golden.py`（B 线用，两者共享同一套判定逻辑，
Rust 侧是权威实现，Python 侧是复刻——判定逻辑以 Rust 为准，
Python 只做导出格式转换，不重新判定）。

### 5.1 入选条件（缺一不可）

1. `verified.ok == true`（验证通过）。
2. 费了功夫：`steps > 1` **或** `surprise >= 0.25`（与 FSM 的
   plastic 阈值一致）。一步到位的成功不用蒸馏——模型已经会了。
3. 证据关通过（§4.4）。
4. 去重：同一 `task_class` 下，与已入选黄金轨迹的任务文本
   相似度过高（归一化后 SHA-256 相同，或编辑距离极小）的不重复
   入选。同一模式反复出现只留最早的一条，避免某个任务类
   flooding 训练集。

### 5.2 提炼形式（决策：带修复的推导链 + 压缩诊断）

两条路线已讨论，定为**带修复的推导链**，并加一道压缩：

- 保留“报错 → 诊断 → 修正”的关键转折，砍掉重复的无效尝试。
- 压缩方式：睡眠时用模型自身重写（固定模板 prompt，见下），
  把多轮拉扯压成一句话诊断，写进 `meta.diagnosis`，同时 response
  里只保留最后一次成功的干净推导。
- 重写模板（固定文本，睡眠时调用 `model.generate`）：

```
下面是一段工具调用记录。模型先犯了错，看到报错后修正，最终成功。
请用一句话总结：错在哪里、为什么错、改成了什么。只输出这一句话。
```

- 重写失败时（生成为空、超过 200 字、不含修正关键词）的回退：
  规则提取——取最后一条 error 结果 + 最终答案的第一句，拼成诊断。
  回退也失败则整条轨迹不入选（宁缺毋滥）。

这样既保留调试知识（和刚补的报错自愈数据同一方向），又不把
“先犯错”学成习惯：response 里没有错误轮次，只有成功推导；
错误只以一句话诊断的形式存在于 meta 里，不进训练文本。

### 5.3 输出

输出即 §3.2 的 Golden record。`prompt` 与 `distill_train.py`
的训练格式一致（SYSTEM_PROMPT + User 任务 + Assistant:），
`response` 是提炼后的推导 + 最终答案，训练时走 completion-only
loss（prompt 部分 label 为 -100，已实现）。

## 6. A 线：轻量本地线（每晚睡眠，零 GPU）

### 6.1 三个落点

1. **ReplayBuffer 黄金池**：`ReplayBuffer` 新增 `add_golden(rec)`，
   黄金记录单独成池。`mixed_batch` 改为三路混合：60% 当日高惊奇
   WAL + 20% 锚点 + **20% 黄金**（黄金不足时按比例回填当日 WAL）。
   黄金池按 `task_class` 轮询取，避免单类垄断。
2. **海马体 anchor**：每条黄金轨迹的（任务关键 token → 解法摘要）
   作为 anchor 存进海马体（`add_anchor` 已有接口，复用）。这是
   情景记忆：“这类问题，解法是 Y”，下次遇到相似任务先想起解法。
3. **MeZO 调 γ**：`/sleep` 的 MeZO 步骤（已有，LM 子空间 516 维）
   的训练序列改成 mixed_batch 的输出（含 20% 黄金），其余配置
   不变（K=4 方向、eps 0.002、lr 0.002、5% trust region）。
   容量小，只能微调输出分布——定位就是“微调”，不指望它教新技能。

### 6.2 睡眠流程（`/sleep` 的新顺序）

```
1. 扫 trace journal，提取当日黄金轨迹（§5）
2. 黄金轨迹 → ReplayBuffer 黄金池 + 海马体 anchor
3. mixed_batch（三路混合）→ MeZO 调 LM 子空间 γ（已有配置）
4. 海马体正交化 + 快照（已有，consolidate_hippocampus）
5. φ 版本快照（新增，见 §6.3）
6. 写睡眠统计（见 §8.3）
```

第 5 步之前，`/sleep` 还没有 WAL 重放接线（README 下一步第 5 条）。
A 线落地时把 WAL 重放和黄金池一起接进来，一次改完。

### 6.3 φ 版本快照 + 回滚（新增，A 线必需）

现在只有海马体有 HIP1 版本快照，`φ`（模型缩放因子）没有。
MeZO 改坏了 γ 没法回滚，A 线必须补：

- 新文件 `crates/cortex-runtime/src/phi_snapshot.rs`：
  `save_phi(version) / load_phi(version) / latest_phi()`，
  格式与 HIP1 对齐（magic `"PHI1"` + CRC32 + 版本号）。
- `/sleep` 在 MeZO 之前先 `save_phi`，MeZO 之后在 held-out
  黄金任务上跑快速评估（§8.2 的轻量版：5–10 条，loss 不升），
  升了就 `load_phi` 回滚并记一条统计。
- 磁盘占用：φ 全量 24,116 个 f32 ≈ 96 KB，留最近 5 个版本，
  不到 1 MB，可忽略。

## 7. B 线：离线 SFT 线（攒批，GPU 机器）

### 7.1 导出：`tools/export_golden.py`（新增）

- 输入：`traces.jsonl`（journal）+ 已导出的 golden 记录清单
  （防重复导出，存 `golden-exported.json`，记 journal_seq）。
- 逻辑：按 §5.1 的入选条件过滤（复刻 Rust 判定，不重新发明），
  输出 `golden-<date>.jsonl`，格式即 §3.2。
- 注意：提炼（§5.2 的压缩诊断）需要模型重写，离线导出时
  **不做重写**——重写是 A 线睡眠时做的，B 线直接取 journal 里
  已有的 `meta.diagnosis`（A 线写回 journal）。B 线不依赖 A 线
  跑过：若某条黄金轨迹还没有 diagnosis，导出时跳过并记数，
  等下次睡眠补上再导。

### 7.2 批量阈值与训练

- 攒够 **500 条**黄金轨迹（或 30 天，先到为准）触发一次 B 线训练。
  500 条的依据：`distill_train.py` 默认每 200 步评估一次，
  500 条 × 2 epoch ≈ 1000 步，刚好够一次完整训练 + 评估；
  太少则 SFT 噪声大、容易过拟合到几条轨迹。
- 训练：`distill_train.py --data golden-<date>.jsonl --init-from
  <当前 .ctxb>`，其余参数沿用蒸馏默认值（lr 2e-4、2 epoch、
  completion-only loss）。黄金数据**混 20% 通用蒸馏数据**
  （从 `distill-data.jsonl` 里抽），防灾难性遗忘：只训黄金
  轨迹会让模型在别的任务上退化。
- 训练在 GPU 机器上跑（与蒸馏同一台），不进 Rust 引擎。

### 7.3 评估门与部署

- 新 `.ctxb` 上线前必须过评估门（§8.2）：held-out 黄金任务的
  一次成功率不下降，且目标任务类上升。门不过 → 丢弃新 `.ctxb`，
  保留旧的，记一条统计。
- 通过后：替换 `.ctxb`（旧文件改名保留，`student.ctxb.bak-<date>`），
  CLI 下次启动加载新的；同时 journal 里记一条部署记录。
- 回滚：新 `.ctxb` 上线后 7 天内，若用户反馈明显退化或
  held-out 评估下降，手动恢复 `.bak` 文件即可（不需要代码支持）。

## 8. 评估

### 8.1 验收指标（唯一主指标）

**一次成功率**：给定任务类，模型在**一轮之内**（不许试错、
不许调工具超过 1 次）完成任务且验证通过的比例。

- 基线：在 L3 上线前，对每个任务类测一次（用当前模型 + 固定
  评估集），记为基线。
- 目标：目标任务类（先定 code 类）一次成功率上升，且其余类
  不下降。没有“涨多少算赢”的硬数字——先跑出第一组 before /
  after 数据再定阈值。

### 8.2 评估集

- 每个任务类 20–30 条 held-out 任务，**永不进训练**（不写进
  journal 的提取范围，单独文件 `eval/<task_class>.jsonl`）。
- 评估集来源：人工写 10 条 + 从 journal 里已验证的任务改写
  10–20 条（改数字、改文件名、改需求细节，保持任务类不变）。
- 评估方法：`--agent` 模式跑每条任务（max-steps 1，测一次成功率），
  验证器打分。A 线轻量版：每晚睡眠后抽 5–10 条跑 loss 对比
  （不跑完整 agent 评估，省时间）；B 线完整版：新 `.ctxb` 上线前
  全量跑。

### 8.3 统计（可观测性）

睡眠结束时写一行 JSON 到 `<data_dir>/sleep-stats.jsonl`：

```json
{"date": "2026-11-21", "traces": 42, "golden": 7,
 "by_class": {"code": 5, "calc": 2}, "mezo_loss_before": 5.24,
 "mezo_loss_after": 4.16, "phi_rollback": false, "exported": 0}
```

- `traces` 当日 journal 新增数，`golden` 当日入选数，
  `exported` 当日 B 线导出数。数字对不上时先查这里。

## 9. 风险与缓解

| 风险 | 现象 | 缓解 |
|---|---|---|
| 验证器被“骗” | 模型写出通过测试但逻辑错的代码（测试覆盖不全）；算式蒙对数字 | §4.4 证据关 + 覆盖关；评估集人工审一遍 |
| 奖励 hacking | 模型学会挑简单任务刷黄金轨迹（只做 steps=2 的 trivial 任务） | 入选条件要求 steps>1 或高 surprise；按类配额，单类不超过黄金池 50% |
| 分布漂移 | 黄金轨迹全来自某一类任务，SFT 后别的任务退化 | B 线混 20% 通用数据；评估门要求其余类不下降 |
| 灾难性遗忘 | MeZO / SFT 后通用能力下降 | A 线 trust region（已有 5%）+ 轻量评估；B 线混通用数据 + 评估门 |
| 提炼学坏习惯 | 模型从带修复的轨迹里学会“先犯错再改” | response 里只有成功推导，错误只进 meta.diagnosis，不进训练文本（§5.2） |
| Journal 膨胀 | traces.jsonl 无限增长，磁盘和扫描变慢 | 按月轮转（`traces-2026-11.jsonl`），提取时只扫当月 + 上月 |

## 10. 工作计划

### Phase L3-1：采集 + 验证器骨架（先做，约 1–2 天工作量）

- `tools.rs`：`AgentOutcome` 加 `task` 字段；`AgentLoop::run`
  返回后由 CLI 写 journal（`crates/cortex-runtime/src/journal.rs`，
  JSONL 追加写）。
- `verify.rs`：`Verifier` trait + `Registry` + `calc` 验证器
  （+ `file`，同批做，两者都无子进程）。
- 测试：journal  roundtrip（写→读→字段全）；calc 验证器
  （对/错/超时各一例）；`AgentOutcome.task` 贯穿到 CLI。
- DoD：跑一次 `--agent`，journal 里多一行，`verified` 字段正确。

### Phase L3-2：提取 + A 线（接着做）

- `golden.rs`（`cortex-sleep`）：入选条件 + 去重 + 提炼
  （压缩诊断用模型重写 + 规则回退）。
- `replay.rs`：`add_golden` + `mixed_batch` 三路混合。
- `phi_snapshot.rs`：`PHI1` 快照 + `/sleep` 前后快照与回滚。
- `/sleep` 流程改成 §6.2 的 6 步（含 WAL 重放接线，顺带把
  README 下一步第 5 条做了）。
- 测试：提取器（入选/拒绝/去重各一例）；mixed_batch 三路
  比例；φ 快照 roundtrip + 回滚；`/sleep` 端到端（fake journal）。
- DoD：fake journal → `/sleep` → 黄金池非空、MeZO loss 下降、
  φ 快照多一个版本。

### Phase L3-3：code 验证器（单做，不阻塞 L3-2）

- 沙箱执行器：子进程 + 超时 + 无网络 + 临时副本 workspace。
- `code` 验证器：跑测试命令，通过数 > 0 才算 ok。
- 测试：通过/失败/超时/测试数为 0 各一例。
- DoD：一个真实修 bug 的 agent turn 被 code 验证器判为通过。

### Phase L3-4：B 线（攒够 500 条后，或先写好脚本备用）

- `tools/export_golden.py`：读 journal → 按 §5.1 过滤 →
  写 `golden-<date>.jsonl`（跳过无 diagnosis 的，记数）。
- 评估集：`eval/code.jsonl` 等，每类 20–30 条（人工写 + 改写）。
- 第一次 B 线训练：混 20% 通用数据 → 新 `.ctxb` → 评估门 →
  替换或丢弃。
- DoD：新 `.ctxb` 在 held-out code 任务上一次成功率上升，
  其余类不下降。

## 11. 待决策（有推荐默认值，用户可推翻）

| # | 问题 | 推荐默认 | 推翻成本 |
|---|---|---|---|
| 1 | 提炼形式 | 带修复的推导链 + 一句话诊断（§5.2） | 低：改提取器一处 |
| 2 | 验证器顺序 | calc → file → code | 低：code 本来就是独立 phase |
| 3 | 评估门是否强制 | 强制：B 线新 `.ctxb` 不过门不上线；A 线轻量评估不过则 φ 回滚 | 中：关掉门等于去掉保险，不推荐 |
| 4 | 黄金池比例 | mixed_batch 里 20%（60/20/20） | 低：一个常数 |
| 5 | B 线触发阈值 | 500 条或 30 天，先到为准 | 低：一个常数 |
| 6 | Journal 轮转 | 按月轮转 | 低：文件名规则 |

## 12. 与 L1 / L2 的关系

- L1（海马体）是 A 线的落点之一：黄金轨迹的 anchor 存进海马体，
  是 L1 机制的直接复用，不新增记忆结构。
- L2（微专家）是 B 线的另一个出口：B 线的 SFT 数据（golden JSONL）
  可以直接用来训 adapter；L2 建好后，B 线的训练目标从“整模型
  SFT”改成“adapter SFT”，其余流程不变。
- 三层的关系：L1 记喜好与解法 → L2 长技能块 → L3 把试错压缩成
  直觉。L3 的黄金轨迹同时喂给 L1（anchor）和 L2（adapter 训练数据），
  是三层之间的连接件。
