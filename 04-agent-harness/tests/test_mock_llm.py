"""MockLLM 的测试（5 个）—— 验证我们的测试工具本身是对的。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.mock_llm import (  # noqa: E402
    MockLLM,
    assistant_text,
    assistant_tool_call,
    assistant_tool_calls,
)
from harness.types import Message  # noqa: E402


def test_script_order() -> None:
    llm = MockLLM([assistant_tool_call("calc", {"a": 1}), assistant_text("done")])
    r1 = llm.chat([], [])
    r2 = llm.chat([], [])
    assert r1.wants_tools is True
    assert r2.wants_tools is False
    assert r2.content == "done"


def test_records_calls() -> None:
    llm = MockLLM([assistant_text("ok")])
    msgs = [Message(role="user", content="hi")]
    llm.chat(msgs, [{"name": "t"}])
    assert llm.call_count == 1
    assert llm.calls[0]["messages"][0]["content"] == "hi"
    assert llm.calls[0]["tools"] == [{"name": "t"}]


def test_script_exhausted_raises() -> None:
    llm = MockLLM([assistant_text("only one")])
    llm.chat([], [])
    try:
        llm.chat([], [])
    except RuntimeError as e:
        assert "脚本已用完" in str(e)
    else:
        raise AssertionError("脚本用完后应当抛 RuntimeError")


def test_callable_response() -> None:
    def echo(messages, tools):
        return assistant_text(f"看到 {len(messages)} 条消息")

    llm = MockLLM([echo])
    r = llm.chat([Message(role="user", content="a")], [])
    assert r.content == "看到 1 条消息"


def test_multi_tool_call() -> None:
    llm = MockLLM([assistant_tool_calls([("a", {}), ("b", {"x": 2})])])
    r = llm.chat([], [])
    assert len(r.tool_calls) == 2
    assert r.tool_calls[0].id == "call_1"
    assert r.tool_calls[1].name == "b"
    assert r.tool_calls[1].arguments == {"x": 2}


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
