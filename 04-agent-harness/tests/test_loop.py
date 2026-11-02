import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.loop import AgentLoop  # noqa: E402
from harness.mock_llm import (  # noqa: E402
    MockLLM,
    assistant_text,
    assistant_tool_call,
    assistant_tool_calls,
)
from harness.registry import ToolRegistry  # noqa: E402


def _make(script, **kw):
    reg = ToolRegistry()

    @reg.tool(description="加法")
    def add(a: int, b: int) -> int:
        return a + b

    @reg.tool(description="会崩")
    def boom() -> str:
        raise ValueError("炸了")

    llm = MockLLM(script)
    loop = AgentLoop(llm, reg, **kw)
    return loop, llm, reg


def test_direct_answer_no_tools() -> None:
    loop, llm, _ = _make([assistant_text("你好")])
    tr = loop.run("hi")
    assert tr.stop_reason == "converged"
    assert tr.final_answer == "你好"
    assert len(tr.steps) == 0
    assert llm.call_count == 1


def test_single_tool_then_answer() -> None:
    loop, _, _ = _make([
        assistant_tool_call("add", {"a": 1, "b": 2}, call_id="c1"),
        assistant_text("答案是 3"),
    ])
    tr = loop.run("1+2")
    assert tr.stop_reason == "converged"
    assert tr.final_answer == "答案是 3"
    assert len(tr.steps) == 1
    assert tr.steps[0].results[0].ok is True
    assert tr.steps[0].results[0].output == "3"


def test_multi_step() -> None:
    loop, _, _ = _make([
        assistant_tool_call("add", {"a": 1, "b": 2}, call_id="c1"),
        assistant_tool_call("add", {"a": 3, "b": 4}, call_id="c2"),
        assistant_text("两个都算完了"),
    ])
    tr = loop.run("两题")
    assert len(tr.steps) == 2
    assert tr.steps[0].results[0].output == "3"
    assert tr.steps[1].results[0].output == "7"


def test_parallel_tool_calls() -> None:
    loop, _, _ = _make([
        assistant_tool_calls([("add", {"a": 1, "b": 1}), ("add", {"a": 2, "b": 2})]),
        assistant_text("ok"),
    ])
    tr = loop.run("并行")
    assert len(tr.steps) == 1
    assert len(tr.steps[0].calls) == 2
    assert len(tr.steps[0].results) == 2
    assert all(r.ok for r in tr.steps[0].results)


def test_max_steps_soft_landing() -> None:
    script = [
        assistant_tool_call("add", {"a": 1, "b": 0}, call_id="c1"),
        assistant_tool_call("add", {"a": 2, "b": 0}, call_id="c2"),
        assistant_tool_call("add", {"a": 3, "b": 0}, call_id="c3"),
        assistant_text("步数用尽，我直接总结：1+2+3=6"),
    ]
    loop, llm, _ = _make(script, max_steps=3, max_repeat=10)
    tr = loop.run("累加")
    assert tr.stop_reason == "max_steps"
    assert "6" in tr.final_answer
    assert len(tr.steps) == 3
    # 3 次循环 + 1 次软着陆
    assert llm.call_count == 4


def test_loop_detection() -> None:
    same = assistant_tool_call("add", {"a": 1, "b": 1}, call_id="c1")
    loop, llm, _ = _make(
        [same, same, same, assistant_text("我不重复了，直接答：2")],
        max_steps=10, max_repeat=2,
    )
    tr = loop.run("卡住")
    assert tr.stop_reason == "loop_detected"
    assert "2" in tr.final_answer


def test_tool_error_fed_back_to_model() -> None:
    loop, llm, _ = _make([
        assistant_tool_call("boom", call_id="c1"),
        assistant_text("工具报错了，我换个说法回答"),
    ])
    tr = loop.run("触发错误")
    assert tr.steps[0].results[0].ok is False
    assert "炸了" in (tr.steps[0].results[0].error or "")
    # 错误应当出现在给模型的上下文里
    second_call_msgs = llm.calls[1]["messages"]
    tool_msgs = [m for m in second_call_msgs if m["role"] == "tool"]
    assert tool_msgs and "ERROR" in tool_msgs[0]["content"]


def test_messages_growth() -> None:
    loop, llm, _ = _make([
        assistant_tool_call("add", {"a": 1, "b": 2}, call_id="c1"),
        assistant_text("3"),
    ])
    loop.run("task")
    assert llm.call_count == 2
    first = llm.calls[0]["messages"]
    second = llm.calls[1]["messages"]
    assert len(first) == 2                      # system + user
    assert len(second) == 4                     # + assistant + tool
    assert second[2]["role"] == "assistant"
    assert second[2]["tool_calls"][0]["name"] == "add"
    assert second[3]["role"] == "tool"


def test_tools_schema_passed_to_llm() -> None:
    loop, llm, _ = _make([assistant_text("ok")])
    loop.run("x")
    tools = llm.calls[0]["tools"]
    names = [t["function"]["name"] for t in tools]
    assert "add" in names and "boom" in names

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 通过")
