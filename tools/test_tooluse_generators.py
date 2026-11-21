"""Fake-teacher tests for the tool-use generators in distill_generate.py.

No vLLM / GPU needed: a FakeLLM returns canned teacher replies in
order, and the tests check that the verification logic (real
execution, result substitution, evidence checks, negative-scenario
gates) behaves correctly.

Run:  python3 tools/test_tooluse_generators.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import distill_generate as dg


class _Out:
    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, text):
        self.outputs = [_Out(text)]


class FakeLLM:
    """Returns canned replies in order; repeats the last one when
    the list is exhausted (so attempt-capped generators terminate)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.i = 0

    def chat(self, messages, sampling, use_tqdm=False, chat_template_kwargs=None):
        text = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return [_Resp(text)]


TASK = "计算 12 支笔的总价，每支 3 元，再加 5 元运费"
TRACE_SUCCESS = (
    "我先算一下。\n"
    '<tool_call>{"name": "calculator", "arguments": {"expression": "12*3+5"}}</tool_call>\n'
    '<tool_result name="calculator">999</tool_result>\n'
    "总共是 41 元。"
)
TRACE_NO_EVIDENCE = (
    "我先算一下。\n"
    '<tool_call>{"name": "calculator", "arguments": {"expression": "12*3+5"}}</tool_call>\n'
    '<tool_result name="calculator">41</tool_result>\n'
    "算好了。"
)
TRACE_ERROR_RECOVERY = (
    "我先读一下待办。\n"
    '<tool_call>{"name": "read_file", "arguments": {"path": "notes/tod.txt"}}</tool_call>\n'
    '<tool_result name="read_file">买牛奶</tool_result>\n'
    "读错了，文件名不对，我再读正确的。\n"
    '<tool_call>{"name": "read_file", "arguments": {"path": "notes/todo.txt"}}</tool_call>\n'
    '<tool_result name="read_file">买牛奶</tool_result>\n'
    "今天的任务有：给妈妈打电话，还要买牛奶。"
)
REQ_CLARIFY = "帮我读一下那个文件"
RESP_CLARIFY = "请问你要读的是哪个文件呢？是 notes/todo.txt 还是 data/prices.csv？"
RESP_CLARIFY_BAD = (
    "好的，我来读。\n"
    '<tool_call>{"name": "read_file", "arguments": {"path": "notes/todo.txt"}}</tool_call>'
)
REQ_CHAT = "给我讲个冷笑话"
RESP_CHAT = "哈哈，这个笑话是这样的：有一天，一只鸡走进了图书馆，它问管理员：有虫子吃吗？"
RESP_CHAT_BAD = (
    "让我查一下笑话库。\n"
    '<tool_call>{"name": "list_dir", "arguments": {"path": "notes"}}</tool_call>'
)


def collect(gen):
    return list(gen)


def test_success_substitution():
    recs = collect(dg.gen_tooluse_success(
        FakeLLM([TASK, TRACE_SUCCESS]), None, 1, random.Random(0)))
    assert len(recs) == 1, f"expected 1, got {len(recs)}"
    _prompt, trace, verified, meta = recs[0]
    assert verified
    assert ">41<" in trace, "real result 41 must be in the trace"
    assert "999" not in trace, "hallucinated result 999 must be replaced"
    assert meta["calls"] == ["calculator"]
    print("ok: success substitution (hallucinated 999 -> real 41)")


def test_success_rejects_missing_evidence():
    recs = collect(dg.gen_tooluse_success(
        FakeLLM([TASK, TRACE_NO_EVIDENCE]), None, 1, random.Random(0)))
    assert len(recs) == 0, "final answer without the real result must be dropped"
    print("ok: success rejects missing evidence")


def test_error_recovery():
    recs = collect(dg.gen_tooluse_error_recovery(
        FakeLLM([TASK, TRACE_ERROR_RECOVERY]), None, 1, random.Random(0)))
    assert len(recs) == 1, f"expected 1, got {len(recs)}"
    _prompt, trace, verified, meta = recs[0]
    assert verified
    assert meta["scenario"] == "error_recovery"
    assert "error: read error:" in trace, "failed call must carry the error text"
    assert "/tmp" not in trace, "sandbox temp path must not leak into data"
    assert "给妈妈打电话" in trace, "final answer must quote the real file content"
    assert meta["calls"] == ["read_file", "read_file"]
    print("ok: error recovery (wrong file -> error -> correct file)")


def test_error_recovery_rejects_all_success():
    recs = collect(dg.gen_tooluse_error_recovery(
        FakeLLM([TASK, TRACE_SUCCESS]), None, 1, random.Random(0)))
    assert len(recs) == 0, "no first-call failure means no error-recovery trace"
    print("ok: error recovery rejects all-success traces")


def test_clarification():
    recs = collect(dg.gen_tooluse_clarification(
        FakeLLM([REQ_CLARIFY, RESP_CLARIFY]), None, 1, random.Random(0)))
    assert len(recs) == 1, f"expected 1, got {len(recs)}"
    _prompt, resp, verified, meta = recs[0]
    assert verified and meta["scenario"] == "clarification"
    assert "<tool_call>" not in resp
    print("ok: clarification (asks which file, calls nothing)")


def test_clarification_rejects_tool_call():
    recs = collect(dg.gen_tooluse_clarification(
        FakeLLM([REQ_CLARIFY, RESP_CLARIFY_BAD]), None, 1, random.Random(0)))
    assert len(recs) == 0, "guessing a tool call instead of asking must be dropped"
    print("ok: clarification rejects tool-call guesses")


def test_no_tool_chat():
    recs = collect(dg.gen_tooluse_no_tool_chat(
        FakeLLM([REQ_CHAT, RESP_CHAT]), None, 1, random.Random(0)))
    assert len(recs) == 1, f"expected 1, got {len(recs)}"
    _prompt, resp, verified, meta = recs[0]
    assert verified and meta["scenario"] == "no_tool_chat"
    assert "<tool_call>" not in resp
    print("ok: no-tool chat (answers directly, calls nothing)")


def test_no_tool_chat_rejects_tool_call():
    recs = collect(dg.gen_tooluse_no_tool_chat(
        FakeLLM([REQ_CHAT, RESP_CHAT_BAD]), None, 1, random.Random(0)))
    assert len(recs) == 0, "calling a tool for small talk must be dropped"
    print("ok: no-tool chat rejects tool calls")


def test_dispatcher_mix():
    replies = (
        [TASK, TRACE_SUCCESS] * 10
        + [TASK, TRACE_ERROR_RECOVERY] * 4
        + [REQ_CLARIFY, RESP_CLARIFY] * 3
        + [REQ_CHAT, RESP_CHAT] * 3
    )
    recs = collect(dg.gen_tooluse(
        FakeLLM(replies), None, 20, random.Random(0)))
    assert len(recs) == 20, f"expected 20, got {len(recs)}"
    from collections import Counter
    counts = Counter(r[3]["scenario"] for r in recs)
    # success scenarios keep their invented-task key; count them together
    success = sum(v for k, v in counts.items()
                  if k not in ("error_recovery", "clarification", "no_tool_chat"))
    assert success == 10, f"success: {success}"
    assert counts["error_recovery"] == 4, counts
    assert counts["clarification"] == 3, counts
    assert counts["no_tool_chat"] == 3, counts
    print(f"ok: dispatcher mix 10/4/3/3 -> {dict(counts)}")


def main():
    test_success_substitution()
    test_success_rejects_missing_evidence()
    test_error_recovery()
    test_error_recovery_rejects_all_success()
    test_clarification()
    test_clarification_rejects_tool_call()
    test_no_tool_chat()
    test_no_tool_chat_rejects_tool_call()
    test_dispatcher_mix()
    print("ALL TOOLUSE GENERATOR TESTS PASSED")


if __name__ == "__main__":
    main()
