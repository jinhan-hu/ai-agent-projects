import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.memory import Memory, estimate_tokens  # noqa: E402
from harness.types import Message  # noqa: E402


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    # 中文按 1 字 1 token
    assert estimate_tokens("你好") == 3   # 2 cjk + (0//4) + 1
    # 英文按 4 字符 1 token
    assert estimate_tokens("abcdefgh") == 3   # 0 cjk + 8//4 + 1


def test_add_and_context() -> None:
    mem = Memory(max_messages=10)
    mem.add(Message(role="user", content="a"))
    mem.add(Message(role="assistant", content="b"))
    ctx = mem.get_context()
    assert len(ctx) == 2
    assert ctx[0].content == "a"


def test_context_respects_max_messages() -> None:
    mem = Memory(max_messages=3)
    for i in range(10):
        mem.add(Message(role="user", content=f"m{i}"))
    ctx = mem.get_context()
    assert len(ctx) == 3
    assert ctx[-1].content == "m9"


def test_should_compress_by_count() -> None:
    mem = Memory(max_messages=2)
    mem.add(Message(role="user", content="a"))
    assert mem.should_compress() is False
    mem.add(Message(role="user", content="b"))
    assert mem.should_compress() is False
    mem.add(Message(role="user", content="c"))
    assert mem.should_compress() is True


def test_compress_with_summarizer() -> None:
    mem = Memory(max_messages=10, keep_recent=2, summarizer=lambda msgs: "旧内容摘要")
    for i in range(6):
        mem.add(Message(role="user", content=f"m{i}"))
    assert mem.compress() is True
    assert mem.summary == "旧内容摘要"
    assert len(mem.messages) == 2
    # 上下文里应包含摘要
    ctx = mem.get_context()
    assert "历史摘要" in ctx[0].content


def test_compress_without_summarizer() -> None:
    mem = Memory(max_messages=10, keep_recent=1)
    mem.add(Message(role="user", content="第一句话"))
    mem.add(Message(role="user", content="第二句话"))
    assert mem.compress() is True
    assert "第一句话" in mem.summary


def test_compress_noop_when_short() -> None:
    mem = Memory(max_messages=10, keep_recent=5)
    mem.add(Message(role="user", content="a"))
    assert mem.compress() is False

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
