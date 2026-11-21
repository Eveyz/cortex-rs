#!/usr/bin/env python3
"""Distillation data generation: Qwen3 teacher -> Cortex-Local student data.

Uses vLLM (offline) to run an instruction-tuned Qwen3 teacher
(default Qwen/Qwen3-30B-A3B-Instruct-2507, MoE, 3B active params
per token) and generates the training set for the 1.6B ternary
student. This is *sequence-level* distillation: the teacher writes
text, the student trains on that text.

Why sequence-level only (not logit KD):
  * The teacher is a Transformer with a 151K vocabulary; the
    student is a DeltaNet SSM with a 32K byte-level vocabulary.
    Logit distillation needs matching output spaces, which do not
    exist here, and weights cannot be transferred across
    architectures at all.
  * Sequence-level distillation sidesteps both problems: text is
    text. The student re-learns the *behaviour* from the teacher's
    outputs.

Categories (seven, matching the 20B recipe, re-balanced for
distillation -- the teacher's output *is* the quality filter,
so far fewer tokens are needed than from-scratch pretraining):
  math      -- CoT math traces. Seed problems with known answers
               are verified exactly; teacher-invented problems are
               kept only when two independent solutions agree.
  code      -- code + unit tests. The solution is executed in an
               isolated temp dir; only passing solutions are kept.
  textbook  -- textbook-style explanations of a topic.
  instruct  -- instruction following (summarise / explain /
               translate / rewrite a given passage).
  dialogue  -- multi-turn bilingual (zh/en) everyday dialogue.
  memory    -- <memory_slot> QA pairs: the script invents a fact,
               the teacher writes a question + an answer that
               quotes the fact inside <memory_slot> tags. Verifies
               the fact value appears in the answer.
  tooluse   -- tool-calling traces for the Rust harness
               (crates/cortex-runtime/src/tools.rs): the teacher
               solves a task with <tool_call> blocks; the script
               parses each call, executes it for real in a
               sandbox, substitutes the REAL result for the
               teacher's hallucinated one, and keeps the trace
               only if the final answer quotes the real results.
               Mixed 50/20/15/15: success / error recovery
               (first call must fail, then the model corrects
               itself) / clarification (a required parameter is
               missing: ask the user, don't guess) / no-tool chat
               (answer directly, call nothing). The tool set, the
               system prompt and the trace format are shared
               verbatim with the Rust harness -- change one side
               and you must change the other.

Verification is the whole point: an unverified teacher output is
no better than web text. Every record carries "verified": bool.

Requirements (run on the GPU machine, NOT on the small VM):
  * vLLM:  pip install vllm
  * Teacher weights (~60 GB BF16 for 30B-A3B): vLLM downloads
    them from Hugging Face on first use, or pre-fetch with
    `hf download Qwen/Qwen3-30B-A3B-Instruct-2507`.
  * 1x H100 80GB (BF16). Qwen3-32B also works via --teacher
    (needs ~64 GB); the MoE default is cheaper per token.
  * A HF token if the teacher repo is gated (it is not, but set
    HF_TOKEN anyway if downloads fail).

Safety note: the code category executes teacher-generated Python.
It runs in a fresh temp dir with a 15 s timeout, a stripped
environment and no network access from the child process. That is
enough for filtering, not a security boundary: for untrusted
teachers, run this script inside a container/VM.

Usage:
  python distill_generate.py --out distill-data.jsonl
  python distill_generate.py --out distill-data.jsonl --limit 20   # smoke test
  python distill_generate.py --out distill-data.jsonl --categories math,code
  python distill_generate.py --teacher Qwen/Qwen3-32B --out data.jsonl

Output: JSONL, one record per line:
  {"category", "prompt", "response", "verified", "meta", "teacher"}
The student trainer (distill_train.py) reads this file directly.
Generation is resumable: existing records in --out are counted
and only the remainder is generated.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TEACHER = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_COUNTS = {
    "math": 3000,
    "code": 2000,
    "textbook": 3000,
    "instruct": 2000,
    "dialogue": 1500,
    "memory": 1000,
    "tooluse": 2000,
}
CATEGORIES = list(DEFAULT_COUNTS)

# ---------------------------------------------------------------------------
# Seed data (small, real, hand-written -- the teacher expands from these)
# ---------------------------------------------------------------------------

# (problem, expected answer as a plain number string)
MATH_SEEDS: list[tuple[str, str]] = [
    ("小明有 12 个苹果，吃了 3 个，又买了 5 个，现在有几个苹果？", "14"),
    ("一个长方形的长是 8 厘米，宽是 5 厘米，它的面积是多少平方厘米？", "40"),
    ("小红每天看 15 页书，一本 120 页的书她几天能看完？", "8"),
    ("3 个工人 4 天修了 240 米路，照这样计算，5 个工人 6 天能修多少米？", "600"),
    ("A train travels 120 km in 2 hours, then 180 km in 3 hours. What is its average speed in km/h?", "60"),
    ("If x + 2x = 27, what is x?", "9"),
    ("一个数加上它的 3 倍等于 48，这个数是多少？", "12"),
    ("超市苹果每斤 6 元，买 3 斤 4 两要付多少钱？（1 斤 = 10 两）", "20.4"),
]

MATH_PROBLEM_STYLES = [
    "小学算术应用题（加减乘除、单位换算）",
    "一元一次方程应用题",
    "几何面积/体积计算题",
    "行程问题（速度、时间、路程）",
    "比例与百分比应用题",
    "grade-school arithmetic word problem in English",
]

# (task description, unit-test file content; the teacher's solution is
# saved as solution.py and must make the tests print ALL TESTS PASSED)
CODE_TASKS: list[tuple[str, str]] = [
    (
        "实现一个函数 quicksort(arr)，对整数列表做快速排序并返回新列表，不得使用内置 sort/sorted。",
        "from solution import quicksort\n"
        "assert quicksort([]) == []\n"
        "assert quicksort([3, 1, 2]) == [1, 2, 3]\n"
        "assert quicksort([5, 5, 1, 4, 5]) == [1, 4, 5, 5, 5]\n"
        "assert quicksort([9, 8, 7, 6]) == [6, 7, 8, 9]\n"
        "print('ALL TESTS PASSED')\n",
    ),
    (
        "实现一个函数 binary_search(arr, target)，arr 为升序列表，返回 target 的下标，找不到返回 -1。",
        "from solution import binary_search\n"
        "assert binary_search([1, 3, 5, 7, 9], 5) == 2\n"
        "assert binary_search([1, 3, 5, 7, 9], 4) == -1\n"
        "assert binary_search([], 1) == -1\n"
        "assert binary_search([2], 2) == 0\n"
        "print('ALL TESTS PASSED')\n",
    ),
    (
        "实现一个函数 count_words(text)，返回一个 dict，统计每个单词（按空格切分，转小写）出现的次数。",
        "from solution import count_words\n"
        "assert count_words('a b a') == {'a': 2, 'b': 1}\n"
        "assert count_words('') == {}\n"
        "assert count_words('Hello hello HELLO') == {'hello': 3}\n"
        "print('ALL TESTS PASSED')\n",
    ),
    (
        "实现一个函数 read_csv_sum(path, column)，读取 CSV 文件并返回指定列的数值之和（跳过表头，空值按 0 计）。",
        "import csv, os\n"
        "from solution import read_csv_sum\n"
        "with open('t.csv', 'w', newline='') as f:\n"
        "    w = csv.writer(f); w.writerow(['a', 'b']); w.writerow(['1', '2']); w.writerow(['3', ''])\n"
        "assert read_csv_sum('t.csv', 'a') == 4\n"
        "assert read_csv_sum('t.csv', 'b') == 2\n"
        "print('ALL TESTS PASSED')\n",
    ),
    (
        "实现一个类 LRUCache(capacity)，支持 get(key) 与 put(key, value)，容量满时淘汰最久未使用的项。",
        "from solution import LRUCache\n"
        "c = LRUCache(2); c.put(1, 1); c.put(2, 2)\n"
        "assert c.get(1) == 1\n"
        "c.put(3, 3)\n"
        "assert c.get(2) == -1\n"
        "assert c.get(3) == 3\n"
        "print('ALL TESTS PASSED')\n",
    ),
    (
        "实现一个函数 fibonacci(n)，返回第 n 个斐波那契数（f(0)=0, f(1)=1），n 最大为 30，要求时间复杂度优于指数级。",
        "from solution import fibonacci\n"
        "assert [fibonacci(i) for i in range(8)] == [0, 1, 1, 2, 3, 5, 8, 13]\n"
        "assert fibonacci(30) == 832040\n"
        "print('ALL TESTS PASSED')\n",
    ),
]

TEXTBOOK_TOPICS: list[tuple[str, str, str]] = [
    ("物理", "牛顿第二定律", "初中生"),
    ("物理", "为什么天会下雨（水循环）", "小学生"),
    ("生物", "光合作用的过程", "初中生"),
    ("生物", "DNA 是如何复制的", "高中生"),
    ("化学", "酸碱中和反应", "初中生"),
    ("数学", "勾股定理及其证明思路", "初中生"),
    ("数学", "什么是概率", "初中生"),
    ("计算机", "什么是递归", "编程初学者"),
    ("计算机", "哈希表的工作原理", "编程初学者"),
    ("历史", "丝绸之路的形成与影响", "中学生"),
    ("地理", "季风是如何形成的", "中学生"),
    ("经济", "供需关系如何决定价格", "普通读者"),
    ("天文", "为什么会有四季", "小学生"),
    ("计算机", "操作系统中的进程与线程", "编程初学者"),
    ("生物", "人体免疫系统的工作方式", "高中生"),
]

INSTRUCT_SEEDS: list[tuple[str, str]] = [
    ("把下面这段话总结成一句话", "人工智能正在改变软件开发的方式。过去程序员需要逐行手写每一行代码，现在他们可以描述需求，让模型生成初稿，再人工检查和修改。这种变化并没有让程序员失业，反而让他们把精力集中在设计、架构和解决真正困难的问题上。"),
    ("把下面这段话翻译成英文", "学习一门新语言最有效的方法不是背单词表，而是大量使用它。每天读一点、写一点、说一点，三个月后会有明显的进步。"),
    ("用通俗的语言解释下面这段话", "过拟合是指模型在训练数据上表现很好，但在没见过的新数据上表现很差，通常是因为模型记住了训练数据的噪声而不是学到了通用规律。"),
    ("把下面这段话改写得更正式", "这个功能做完了，你看一下，没问题就上线吧。"),
    ("列出下面这段话的三个要点", "要做出好的产品，需要三件事：第一，真正理解用户的问题而不是想象；第二，用最简单的方法解决它；第三，上线后根据真实反馈快速迭代。很多团队失败不是因为技术不行，而是因为在错误的问题上做得很努力。"),
    ("Summarize the following paragraph in one sentence", "The water cycle is driven by the sun. Water evaporates from oceans and lakes, condenses into clouds, and falls back as rain or snow. This cycle has run for billions of years and makes life on land possible."),
    ("把下面这段话改写成适合小学生读的版本", "光合作用是植物利用阳光、水和二氧化碳制造有机物并释放氧气的过程，是地球上几乎所有生命的能量来源。"),
    ("从下面这段话中提取出所有的数字和单位", "会议定在下周三下午 3 点，时长 90 分钟，地点是 5 楼 302 会议室，参会人数约 12 人，预算 5000 元。"),
]

DIALOGUE_SCENARIOS: list[tuple[str, str]] = [
    ("朋友向你请教如何开始健身", "zh"),
    ("向同事解释为什么项目要延期", "zh"),
    ("在餐厅向服务员询问过敏原信息", "zh"),
    ("和朋友讨论周末去哪里玩", "zh"),
    ("Asking a stranger for directions in a new city", "en"),
    ("A customer asking for a refund in a shop", "en"),
    ("向房东反映水管漏水，希望尽快维修", "zh"),
    ("和同学讨论小组作业的分工", "zh"),
]

# Memory-interface facts: the script invents the fact (template +
# random values); the teacher only writes the QA pair. Verification
# is exact: the fact value must appear inside <memory_slot> in the
# answer, so no teacher hallucination can pass the gate.
MEMORY_FACT_TEMPLATES: list[tuple[str, str]] = [
    ("API 限流", "内部订单 API 的限流是每分钟 {v} 次"),
    ("会议时间", "每周项目例会固定在周{v}下午 {h} 点"),
    ("项目代号", "新推荐系统的内部代号是 {v}"),
    ("文档位置", "部署手册放在内部 wiki 的 {v} 目录下"),
    ("联系人", "负责支付模块的同事是 {v}"),
    ("版本号", "当前线上服务的版本号是 {v}"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def dedup_key(s: str) -> str:
    return hashlib.sha256(norm_text(s).encode("utf-8")).hexdigest()


def strip_think(text: str) -> str:
    """Remove a Qwen3 <think>...</think> block, if present."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text: str):
    """Last number in the text (after an optional \\boxed{...}), or None."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        text = m.group(1)
    nums = _NUMBER_RE.findall(text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def answers_match(got, expected: str) -> bool:
    if got is None:
        return False
    try:
        return math.isclose(got, float(expected), rel_tol=1e-6, abs_tol=1e-9)
    except ValueError:
        return False


def chat(llm, messages: list[dict], sampling, think: bool):
    """One vLLM chat call. Returns the raw output text (with think block)."""
    out = llm.chat(
        messages,
        sampling,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": think},
    )
    return out[0].outputs[0].text


# ---------------------------------------------------------------------------
# Category generators. Each yields (prompt, response, verified, meta).
# ---------------------------------------------------------------------------

def gen_math(llm, sampling, n: int, rng: random.Random):
    """CoT math. Seeds are verified against known answers; invented
    problems are kept only when two independent solutions agree."""
    made = 0
    # 1) Seed problems (exact verification).
    for problem, expected in MATH_SEEDS:
        if made >= n:
            return
        msgs = [
            {"role": "system", "content": "你是一位耐心的小学数学老师。一步一步推导，最后给出答案。"},
            {"role": "user", "content": f"请一步一步解答这道题，最后一个字给出答案：{problem}"},
        ]
        text = chat(llm, msgs, sampling, think=True)
        got = extract_final_number(strip_think(text))
        if answers_match(got, expected):
            yield (f"请一步一步解答这道题，最后一个字给出答案：{problem}", text, True,
                   {"source": "seed", "expected": expected})
            made += 1
    # 2) Teacher-invented problems (self-consistency verification).
    while made < n:
        style = rng.choice(MATH_PROBLEM_STYLES)
        msgs = [
            {"role": "system", "content": "你是一位出题老师。"},
            {"role": "user", "content": f"出一道{style}，题目本身完整、条件充分，并给出标准答案（只给数字）。格式：\n题目：……\n答案：……"},
        ]
        text = chat(llm, msgs, sampling, think=False)
        m = re.search(r"题目[:：]\s*(.+?)\n答案[:：]\s*(.+)", text, flags=re.DOTALL)
        if not m:
            continue
        problem, expected = m.group(1).strip(), m.group(2).strip()
        if extract_final_number(expected) is None:
            continue
        # Solve it twice independently; keep only when both solutions
        # agree AND match the teacher-stated answer (guards against
        # a shared error in the two samples).
        solve_msgs = [
            {"role": "system", "content": "你是一位耐心的小学数学老师。一步一步推导，最后给出答案。"},
            {"role": "user", "content": f"请一步一步解答这道题，最后一个字给出答案：{problem}"},
        ]
        outs = llm.chat(
            [solve_msgs, solve_msgs], sampling, use_tqdm=False,
            chat_template_kwargs={"enable_thinking": True},
        )
        answers = [extract_final_number(strip_think(o.outputs[0].text)) for o in outs]
        expected_num = extract_final_number(expected)
        if (answers[0] is not None and answers[0] == answers[1]
                and expected_num is not None
                and math.isclose(answers[0], expected_num, rel_tol=1e-6, abs_tol=1e-9)):
            yield (f"请一步一步解答这道题，最后一个字给出答案：{problem}",
                   outs[0].outputs[0].text, True,
                   {"source": "invented", "style": style})
            made += 1


def gen_code(llm, sampling, n: int, rng: random.Random):
    """Code + unit tests. The solution is executed; only passing
    solutions are kept. The response stored is solution + a short
    explanation (the tests themselves are NOT in the response)."""
    made = 0
    tasks = list(CODE_TASKS)
    rng.shuffle(tasks)
    for task, tests in tasks:
        if made >= n:
            return
        msgs = [
            {"role": "system", "content": "你是一位 Python 程序员。只输出代码，不要解释。"},
            {"role": "user", "content": f"写出满足要求的 Python 代码，只给代码块：\n{task}"},
        ]
        text = chat(llm, msgs, sampling, think=False)
        code = extract_code_block(text) or text
        if run_tests(code, tests):
            explain_msgs = [
                {"role": "system", "content": "你是一位编程老师，用两三句话解释代码思路。"},
                {"role": "user", "content": f"解释下面这段代码的思路：\n```python\n{code}\n```"},
            ]
            expl = strip_think(chat(llm, explain_msgs, sampling, think=False))
            response = f"```python\n{code}\n```\n\n{expl}"
            yield (f"写出满足要求的 Python 代码：\n{task}", response, True, {"source": "seed-task"})
            made += 1
    # Teacher-invented tasks (same execution gate).
    while made < n:
        topic = rng.choice(["字符串处理", "列表操作", "字典统计", "文件读写", "日期计算", "基础算法"])
        msgs = [
            {"role": "system", "content": "你是一位出题老师。"},
            {"role": "user", "content": f"出一道关于{topic}的 Python 编程练习题，并给出配套的单元测试（assert 风格，以 print('ALL TESTS PASSED') 结尾）。格式：\n题目：……\n测试：\n```python\n……\n```"},
        ]
        text = chat(llm, msgs, sampling, think=False)
        m = re.search(r"题目[:：]\s*(.+?)\n测试[:：]\s*```python\n(.+?)```", text, flags=re.DOTALL)
        if not m:
            continue
        task, tests = m.group(1).strip(), m.group(2)
        if "ALL TESTS PASSED" not in tests:
            continue
        msgs2 = [
            {"role": "system", "content": "你是一位 Python 程序员。只输出代码，不要解释。"},
            {"role": "user", "content": f"写出满足要求的 Python 代码，只给代码块：\n{task}"},
        ]
        code = extract_code_block(chat(llm, msgs2, sampling, think=False))
        if code and run_tests(code, tests):
            yield (f"写出满足要求的 Python 代码：\n{task}",
                   f"```python\n{code}\n```", True,
                   {"source": "invented", "topic": topic})
            made += 1


def extract_code_block(text: str):
    m = re.search(r"```python\n(.+?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\n(.+?)```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def run_tests(code: str, tests: str) -> bool:
    """Execute teacher code + tests in a fresh temp dir, no network,
    15 s timeout. Returns True only if the tests print ALL TESTS PASSED."""
    try:
        with tempfile.TemporaryDirectory(prefix="distill-code-") as tmp:
            Path(tmp, "solution.py").write_text(code, encoding="utf-8")
            Path(tmp, "test_solution.py").write_text(tests, encoding="utf-8")
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp}
            res = subprocess.run(
                [sys.executable, "test_solution.py"],
                cwd=tmp, env=env, capture_output=True, text=True, timeout=15,
            )
            return res.returncode == 0 and "ALL TESTS PASSED" in res.stdout
    except Exception:
        return False


def gen_textbook(llm, sampling, n: int, rng: random.Random):
    made = 0
    topics = list(TEXTBOOK_TOPICS)
    rng.shuffle(topics)
    while made < n:
        subject, topic, audience = topics[made % len(topics)]
        msgs = [
            {"role": "system", "content": "你是一位教材作者，写清楚概念、原理和一个生活中的例子。"},
            {"role": "user", "content": f"为{audience}写一段关于「{topic}」（{subject}）的教科书式讲解，400-800 字，先讲是什么，再讲为什么，最后给一个生活中的例子。"},
        ]
        text = strip_think(chat(llm, msgs, sampling, think=False))
        if 300 <= len(text) <= 4000:
            yield (f"请写一段关于「{topic}」（{subject}）的教科书式讲解。",
                   text, True, {"subject": subject, "audience": audience})
            made += 1


def gen_instruct(llm, sampling, n: int, rng: random.Random):
    made = 0
    seeds = list(INSTRUCT_SEEDS)
    rng.shuffle(seeds)
    while made < n:
        instruction, passage = seeds[made % len(seeds)]
        msgs = [
            {"role": "system", "content": "你是一位得力的助手，按要求完成任务，不要多余发挥。"},
            {"role": "user", "content": f"{instruction}：\n{passage}"},
        ]
        text = strip_think(chat(llm, msgs, sampling, think=False))
        if 10 <= len(text) <= 2000:
            yield (f"{instruction}：\n{passage}", text, True, {})
            made += 1


def gen_dialogue(llm, sampling, n: int, rng: random.Random):
    made = 0
    scenarios = list(DIALOGUE_SCENARIOS)
    rng.shuffle(scenarios)
    while made < n:
        scenario, lang = scenarios[made % len(scenarios)]
        if lang == "zh":
            user_msg = f"写一段 3-5 轮的中文日常对话，场景：{scenario}。用「A：」「B：」标出发言人，自然口语，不要书面腔。"
        else:
            user_msg = f"Write a 3-5 turn everyday English dialogue. Scenario: {scenario}. Use 'A:' and 'B:' for speakers, natural spoken language."
        msgs = [
            {"role": "system", "content": "你是一位剧本作者，写自然的生活对话。"},
            {"role": "user", "content": user_msg},
        ]
        text = strip_think(chat(llm, msgs, sampling, think=False))
        if 100 <= len(text) <= 3000 and text.count("：") + text.count(":") >= 4:
            yield (f"写一段日常对话，场景：{scenario}", text, True,
                   {"lang": lang})
            made += 1


def gen_memory(llm, sampling, n: int, rng: random.Random):
    """<memory_slot> QA. The script invents the fact; the teacher
    writes the QA pair. Gate: the fact value must appear inside
    <memory_slot>...</memory_slot> in the answer."""
    made = 0
    values = {
        "API 限流": ["100", "500", "1000", "60"],
        "会议时间": ["三", "五", "一"],
        "项目代号": ["白泽", "青鸾", "玄鸟", "烛龙"],
        "文档位置": ["运维手册", "新人指南", "架构决策", "故障复盘"],
        "联系人": ["林晓", "陈默", "王磊", "苏晴"],
        "版本号": ["v2.4.1", "v3.0.0", "v1.9.3"],
    }
    hours = ["2", "3", "4"]
    while made < n:
        kind, template = rng.choice(MEMORY_FACT_TEMPLATES)
        if kind == "会议时间":
            v = rng.choice(values[kind])
            h = rng.choice(hours)
            fact = template.format(v=v, h=h)
            fact_value = f"周{v}"
        else:
            v = rng.choice(values[kind])
            fact = template.format(v=v)
            fact_value = v
        msgs = [
            {"role": "system", "content": "你是一位助手，回答时必须引用给定的事实，并把事实值放在 <memory_slot> 标签里。"},
            {"role": "user", "content": f"已知事实：{fact}。\n请写一个问答：先提一个需要用到这个事实的问题，再回答。回答中引用事实时，把事实值写在 <memory_slot>事实值</memory_slot> 标签里。格式：\n问题：……\n回答：……"},
        ]
        text = strip_think(chat(llm, msgs, sampling, think=False))
        m = re.search(r"问题[:：]\s*(.+?)\n回答[:：]\s*(.+)", text, flags=re.DOTALL)
        if not m:
            continue
        question, answer = m.group(1).strip(), m.group(2).strip()
        slot = re.search(r"<memory_slot>(.*?)</memory_slot>", answer)
        if not slot or fact_value not in slot.group(1):
            continue
        yield (question, answer, True, {"fact": fact})
        made += 1


# ---------------------------------------------------------------------------
# Tool-use distillation (category: tooluse)
#
# CONTRACT WITH THE RUST HARNESS (crates/cortex-runtime/src/tools.rs):
# the tool set below, the system prompt rendered by
# render_tool_system_prompt(), and the trace format
#   <tool_call>{"name": ..., "arguments": {...}}</tool_call>
#   <tool_result name="...">...</tool_result>
# are shared verbatim between this file and tools.rs. If you change
# the wording on one side, change the other side to match, or the
# student will be trained on a format the harness does not speak.
# ---------------------------------------------------------------------------

# (name, one-line description, parameters schema as shown to the model,
#  required parameter names)
TOOL_SPECS: list[tuple[str, str, str, list[str]]] = [
    ("calculator",
     "Evaluate an arithmetic expression (+, -, *, /, %, parentheses).",
     '{"expression": "string"}', ["expression"]),
    ("read_file",
     "Read a text file inside the workspace.",
     '{"path": "string"}', ["path"]),
    ("write_file",
     "Write text to a file inside the workspace.",
     '{"path": "string", "content": "string"}', ["path", "content"]),
    ("list_dir",
     "List files in a directory inside the workspace.",
     '{"path": "string"}', ["path"]),
    ("get_time",
     "Get the current date and time (UTC).",
     "{}", []),
]

TOOL_NAMES = {t[0] for t in TOOL_SPECS}


def render_tool_system_prompt() -> str:
    """The exact system prompt the Rust harness renders. Keep the two
    in sync (same tool lines, same wording, same order)."""
    lines = [
        "You are Cortex, a helpful assistant with access to tools.",
        "Available tools:",
    ]
    for name, desc, params, _req in TOOL_SPECS:
        lines.append(f"- {name}: {desc} Parameters: {params}")
    lines.append(
        "When you need a tool, emit exactly one block and nothing else after it:"
    )
    lines.append('<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>')
    lines.append(
        'The tool result will appear as <tool_result name="tool_name">...</tool_result>. '
        "Then continue. When you can answer directly, answer without a tool call."
    )
    return "\n".join(lines)


SYSTEM_PROMPT = render_tool_system_prompt()

# Fixture files present in the sandbox for every tool-use example.
TOOLUSE_FIXTURES: dict[str, str] = {
    "notes/todo.txt": "买牛奶\n给妈妈打电话\n周五前交报告\n",
    "data/prices.csv": "item,price\napple,6\nbread,12\nmilk,15\n",
    "notes/ideas.txt": "阳台种薄荷\n学做手冲咖啡\n",
}

# (scenario key, brief for the teacher inventing the task)
TOOLUSE_SCENARIOS: list[tuple[str, str]] = [
    ("calc", "出一道需要用计算器解决的日常算术题（单价×数量、折扣、均摊等），给出具体数字"),
    ("read", "出一道需要读取 notes/todo.txt 或 data/prices.csv 才能回答的问题"),
    ("write", "出一道需要把一段文字写入 notes/xxx.txt 并确认写好的任务"),
    ("list", "出一道需要列出 notes 或 data 目录内容才能回答的问题"),
    ("time", "出一道需要知道当前时间才能回答的任务（比如提醒、计算还有多久）"),
    ("calc_write", "出一道需要两步的任务：先计算，再把结果写入文件"),
    ("read_calc", "出一道需要先读 data/prices.csv 再计算合计的任务"),
]


def fixture_listing() -> str:
    out = []
    for path, content in TOOLUSE_FIXTURES.items():
        out.append(f"{path}:\n{content}")
    return "\n".join(out)


def format_number(v) -> str:
    """Integral results print as integers (42, not 42.0); the Rust
    calculator uses the same rule, so traces stay consistent."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def eval_expr(expr: str):
    """Safe arithmetic only: + - * / % ( ) and unary minus. No names,
    no calls, no power operator -- the Rust calculator accepts the
    same subset, so a verified expression runs on both sides."""
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)
        ):
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
            # Truncated remainder (math.fmod), matching the Rust
            # calculator -- Python's % is floor-modulo and differs
            # for negative operands.
            return math.fmod(l, r)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        raise ValueError(f"unsupported expression: {expr!r}")

    return ev(tree)


def execute_tool(name: str, args: dict, sandbox: Path) -> tuple[bool, str]:
    """Execute one tool call for real, jailed to `sandbox`.
    Returns (ok, result_text). Mirrors the Rust built-in tools."""
    if name == "calculator":
        try:
            return True, format_number(eval_expr(str(args.get("expression", ""))))
        except Exception as e:
            return False, f"calculator error: {e}"
    if name in ("read_file", "write_file", "list_dir"):
        raw = str(args.get("path", ""))
        try:
            target = (sandbox / raw).resolve()
            if not target.is_relative_to(sandbox.resolve()):
                return False, f"path outside workspace: {raw}"
        except Exception as e:
            return False, f"bad path: {e}"
        if name == "read_file":
            try:
                text = target.read_text(encoding="utf-8")
                return True, text[:4000]
            except Exception as e:
                return False, f"read error: {e}"
        if name == "write_file":
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                content = str(args.get("content", ""))
                target.write_text(content, encoding="utf-8")
                return True, f"wrote {raw} ({len(content.encode('utf-8'))} bytes)"
            except Exception as e:
                return False, f"write error: {e}"
        try:
            if not target.is_dir():
                return False, f"not a directory: {raw}"
            names = sorted(p.name for p in target.iterdir())
            return True, "\n".join(names) if names else "(empty)"
        except Exception as e:
            return False, f"list error: {e}"
    if name == "get_time":
        return True, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return False, f"unknown tool: {name}"


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESULT_RE = re.compile(r"<tool_result[^>]*>.*?</tool_result>", re.DOTALL)


@contextlib.contextmanager
def _fresh_sandbox():
    """A temp dir seeded with the tool-use fixture files."""
    with tempfile.TemporaryDirectory(prefix="distill-tooluse-") as tmp:
        sandbox = Path(tmp)
        for path, content in TOOLUSE_FIXTURES.items():
            p = sandbox / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        yield sandbox


def _invent_task(llm, sampling, brief: str):
    """Ask the teacher to invent one concrete task. Returns the task
    text, or None if it is too short/long to use."""
    task = strip_think(chat(llm, [
        {"role": "system", "content": "你是一位出题老师。"},
        {"role": "user", "content": f"{brief}，任务要具体（有具体数字/文件名），只输出任务描述，不要解答。\n可用文件：\n{fixture_listing()}"},
    ], sampling, think=False)).strip()
    if len(task) < 10 or len(task) > 500:
        return None
    return task


def _evidence_ok(outcomes, tail: str) -> bool:
    """The final answer (text after the last tool result) must quote
    the real results of the successful calls."""
    for cname, cargs, ok, res in outcomes:
        if not ok:
            continue
        if cname == "calculator":
            if res not in tail:
                return False
        elif cname == "read_file":
            lines = [ln for ln in res.splitlines() if len(ln.strip()) >= 4]
            if not any(ln.strip() in tail for ln in lines):
                return False
        elif cname == "list_dir":
            names = [ln for ln in res.splitlines() if ln and ln != "(empty)"]
            if names and not any(nm in tail for nm in names):
                return False
        elif cname == "write_file":
            if str(cargs.get("path", "")) not in tail:
                return False
        elif cname == "get_time":
            if not re.search(r"\d{2}:\d{2}", tail):
                return False
    return True


def _run_tool_trace(llm, sampling, task, sandbox, mistake_first=False):
    """Solve `task` with tools, execute every parsed call for real,
    substitute the REAL results for the teacher's hallucinated ones.
    Returns (trace, outcomes) or None. outcomes is a list of
    (name, args, ok, result_text), in call order. When mistake_first
    is set, the first call must fail and a later call must succeed
    (error-recovery traces); otherwise every call must succeed."""
    user = f"任务：{task}\n\n可用文件：\n{fixture_listing()}\n\n请一步一步完成任务。需要工具时输出 <tool_call> 块，工具结果出现后继续，最后直接给出答案。"
    if mistake_first:
        user += "\n注意：请先犯一个合理的错误（比如读一个不存在的文件名、把算式写错、或漏掉一个必填参数），看到工具报错后纠正自己，再完成任务。"
    trace = chat(llm, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ], sampling, think=True)
    trace = strip_think(trace)
    if trace.startswith("Assistant:"):
        trace = trace[len("Assistant:"):].strip()
    # Parse every call; validation failures become failed outcomes
    # (the model can learn to recover from them), while malformed
    # JSON cannot be attributed to a tool and drops the trace.
    outcomes = []
    for m in TOOL_CALL_RE.finditer(trace):
        try:
            raw = json.loads(m.group(1))
            cname = raw["name"]
            cargs = raw.get("arguments", {})
            if not isinstance(cargs, dict):
                raise ValueError("arguments is not an object")
        except Exception:
            return None
        if cname not in TOOL_NAMES:
            outcomes.append((cname, cargs, False, f"unknown tool: {cname}"))
            continue
        spec = next(t for t in TOOL_SPECS if t[0] == cname)
        missing = [r for r in spec[3] if r not in cargs]
        if missing:
            outcomes.append((cname, cargs, False,
                             f"missing required parameter: {missing[0]}"))
            continue
        good, res = execute_tool(cname, cargs, sandbox)
        outcomes.append((cname, cargs, good, res))
    if not outcomes:
        return None
    if mistake_first:
        if outcomes[0][2] or not any(o[2] for o in outcomes[1:]):
            return None
    elif not all(o[2] for o in outcomes):
        return None
    # Substitute the REAL results (failed calls get the same
    # "error: ..." prefix the Rust harness uses). Sanitize temp-dir
    # paths out of error text so no sandbox path leaks into data.
    result_blocks = list(TOOL_RESULT_RE.finditer(trace))
    if len(result_blocks) != len(outcomes):
        return None
    rebuilt = []
    pos = 0
    for (cname, _cargs, ok, res), block in zip(outcomes, result_blocks):
        res = res.replace(str(sandbox) + "/", "").replace(str(sandbox), "")
        content = res if ok else f"error: {res}"
        rebuilt.append(trace[pos:block.start()])
        rebuilt.append(f'<tool_result name="{cname}">{content}</tool_result>')
        pos = block.end()
    rebuilt.append(trace[pos:])
    trace = "".join(rebuilt)
    tail = trace.rsplit("</tool_result>", 1)[-1]
    if not _evidence_ok(outcomes, tail):
        return None
    return trace, outcomes


def gen_tooluse_success(llm, sampling, n: int, rng: random.Random):
    """50%: ordinary tool-use traces (all calls succeed)."""
    made = 0
    attempts = 0
    while made < n and attempts < n * 6:
        attempts += 1
        key, brief = rng.choice(TOOLUSE_SCENARIOS)
        with _fresh_sandbox() as sandbox:
            task = _invent_task(llm, sampling, brief)
            if task is None:
                continue
            got = _run_tool_trace(llm, sampling, task, sandbox)
            if got is None:
                continue
            trace, outcomes = got
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {task}\nAssistant:"
            yield (prompt, trace, True,
                   {"scenario": key, "calls": [o[0] for o in outcomes]})
            made += 1


def gen_tooluse_error_recovery(llm, sampling, n: int, rng: random.Random):
    """20%: the model makes a plausible mistake first (wrong file,
    bad expression, missing parameter), reads the error, corrects
    itself, and completes the task."""
    made = 0
    attempts = 0
    while made < n and attempts < n * 6:
        attempts += 1
        key, brief = rng.choice(TOOLUSE_SCENARIOS)
        with _fresh_sandbox() as sandbox:
            task = _invent_task(llm, sampling, brief)
            if task is None:
                continue
            got = _run_tool_trace(llm, sampling, task, sandbox,
                                  mistake_first=True)
            if got is None:
                continue
            trace, outcomes = got
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {task}\nAssistant:"
            yield (prompt, trace, True,
                   {"scenario": "error_recovery",
                    "calls": [o[0] for o in outcomes]})
            made += 1


CLARIFICATION_SCENARIOS = [
    ("read", "出一道用户想读文件但没说文件名的请求（比如“帮我读一下那个文件”）"),
    ("write", "出一道用户想写文件但没说写什么内容、写到哪里的请求"),
    ("calc", "出一道用户想算账但没给任何数字的请求"),
    ("list", "出一道用户想看目录内容但没说哪个目录的请求"),
]

NO_TOOL_CHAT_SCENARIOS = [
    ("greet", "出一句日常打招呼或闲聊（比如早上好、今天过得怎么样）"),
    ("joke", "出一句想听笑话或段子的请求"),
    ("explain", "出一句想解释一个概念的请求（比如量子纠缠、为什么天空是蓝的）"),
    ("advice", "出一句想要生活建议的请求（比如如何提高睡眠质量）"),
    ("summarize", "出一句让模型复述或改写一段用户直接给出的话的请求"),
]


def _invent_request(llm, sampling, brief: str):
    """Invent one short user request (clarification / chat)."""
    req = strip_think(chat(llm, [
        {"role": "system", "content": "你是一位出题老师。"},
        {"role": "user", "content": f"{brief}，只输出请求本身，不要解答。"},
    ], sampling, think=False)).strip()
    if len(req) < 5 or len(req) > 300:
        return None
    return req


def gen_tooluse_clarification(llm, sampling, n: int, rng: random.Random):
    """15%: the request is missing a required parameter. The correct
    behaviour is NO tool call -- ask the user instead of guessing.
    A trace is kept only if it asks a question and calls nothing."""
    made = 0
    attempts = 0
    while made < n and attempts < n * 6:
        attempts += 1
        key, brief = rng.choice(CLARIFICATION_SCENARIOS)
        req = _invent_request(llm, sampling, brief)
        if req is None:
            continue
        resp = strip_think(chat(llm, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req},
        ], sampling, think=True))
        if resp.startswith("Assistant:"):
            resp = resp[len("Assistant:"):].strip()
        if "<tool_call>" in resp or "<tool_result>" in resp:
            continue
        if "?" not in resp and "？" not in resp:
            continue
        if len(resp) < 10:
            continue
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {req}\nAssistant:"
        yield (prompt, resp, True, {"scenario": "clarification"})
        made += 1


def gen_tooluse_no_tool_chat(llm, sampling, n: int, rng: random.Random):
    """15%: ordinary chat with tools available. The correct behaviour
    is to answer directly and call nothing (anti hammer-syndrome)."""
    made = 0
    attempts = 0
    while made < n and attempts < n * 6:
        attempts += 1
        key, brief = rng.choice(NO_TOOL_CHAT_SCENARIOS)
        req = _invent_request(llm, sampling, brief)
        if req is None:
            continue
        resp = strip_think(chat(llm, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req},
        ], sampling, think=True))
        if resp.startswith("Assistant:"):
            resp = resp[len("Assistant:"):].strip()
        if "<tool_call>" in resp or "<tool_result>" in resp:
            continue
        if len(resp) < 20:
            continue
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {req}\nAssistant:"
        yield (prompt, resp, True, {"scenario": "no_tool_chat"})
        made += 1


def gen_tooluse(llm, sampling, n: int, rng: random.Random):
    """Tool-use traces in the README mix: 50% success, 20% error
    recovery, 15% clarification, 15% no-tool chat."""
    quotas = [
        (gen_tooluse_success, n * 50 // 100),
        (gen_tooluse_error_recovery, n * 20 // 100),
        (gen_tooluse_clarification, n * 15 // 100),
        (gen_tooluse_no_tool_chat,
         n - n * 50 // 100 - n * 20 // 100 - n * 15 // 100),
    ]
    for gen, quota in quotas:
        if quota > 0:
            yield from gen(llm, sampling, quota, rng)


GENERATORS = {
    "math": gen_math,
    "code": gen_code,
    "textbook": gen_textbook,
    "instruct": gen_instruct,
    "dialogue": gen_dialogue,
    "memory": gen_memory,
    "tooluse": gen_tooluse,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_existing(path: Path):
    """Return (records, per-category counts) already in the output file."""
    counts = {c: 0 for c in CATEGORIES}
    seen = set()
    if not path.exists():
        return counts, seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts[rec.get("category", "")] = counts.get(rec.get("category", ""), 0) + 1
            seen.add(dedup_key(rec.get("response", "")))
    return counts, seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate distillation data with a Qwen3 teacher (vLLM)")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--categories", default="all", help="comma list, or 'all'")
    ap.add_argument("--per-category", default=None,
                    help='JSON map like {"math": 3000, "code": 2000}; defaults used otherwise')
    ap.add_argument("--limit", type=int, default=None, help="cap every category at N (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cats = CATEGORIES if args.categories == "all" else [c.strip() for c in args.categories.split(",")]
    for c in cats:
        if c not in GENERATORS:
            print(f"unknown category: {c}", file=sys.stderr)
            return 2
    counts = dict(DEFAULT_COUNTS)
    if args.per_category:
        counts.update(json.loads(args.per_category))
    if args.limit is not None:
        counts = {c: min(counts[c], args.limit) for c in counts}

    out_path = Path(args.out)
    existing, seen = load_existing(out_path)
    rng = random.Random(args.seed)

    # vLLM import is lazy so --help works without a GPU install.
    from vllm import LLM, SamplingParams

    print(f"teacher: {args.teacher}", flush=True)
    llm = LLM(
        model=args.teacher,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(
        n=1, temperature=args.temperature, top_p=0.95,
        max_tokens=args.max_tokens,
    )

    total_new = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for cat in cats:
            target = counts[cat]
            have = existing.get(cat, 0)
            if have >= target:
                print(f"[{cat}] already have {have}/{target}, skip", flush=True)
                continue
            need = target - have
            print(f"[{cat}] generating {need} (have {have}/{target})...", flush=True)
            kept = 0
            for prompt, response, verified, meta in GENERATORS[cat](llm, sampling, need, rng):
                key = dedup_key(response)
                if key in seen:
                    continue
                seen.add(key)
                rec = {
                    "category": cat, "prompt": prompt, "response": response,
                    "verified": verified, "meta": meta, "teacher": args.teacher,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
                total_new += 1
                if kept % 50 == 0:
                    print(f"[{cat}] kept {kept}/{need}", flush=True)
            print(f"[{cat}] done: kept {kept}/{need}", flush=True)
    print(f"wrote {total_new} new records to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
