import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time  # noqa: E402

from harness.registry import ToolRegistry  # noqa: E402
from harness.sandbox import run_in_sandbox  # noqa: E402
from harness.types import ToolCall  # noqa: E402


def _reg() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(description="加法")
    def add(a: int, b: int) -> int:
        return a + b

    @reg.tool(description="会崩")
    def boom() -> str:
        raise ValueError("故意炸了")

    @reg.tool(description="死循环")
    def hang() -> str:
        time.sleep(5)
        return "never"

    @reg.tool(description="长输出")
    def long_out() -> str:
        return "x" * 10000

    return reg


def test_success() -> None:
    r = run_in_sandbox(ToolCall("c1", "add", {"a": 1, "b": 2}), _reg())
    assert r.ok is True
    assert r.output == "3"
    assert r.duration_ms >= 0


def test_tool_not_found() -> None:
    r = run_in_sandbox(ToolCall("c1", "nope", {}), _reg())
    assert r.ok is False
    assert "tool_not_found" in r.error


def test_bad_arguments() -> None:
    r = run_in_sandbox(ToolCall("c1", "add", {"x": 1}), _reg())
    assert r.ok is False
    assert "bad_arguments" in r.error


def test_exception_captured() -> None:
    r = run_in_sandbox(ToolCall("c1", "boom", {}), _reg())
    assert r.ok is False
    assert "ValueError" in r.error and "故意炸了" in r.error


def test_timeout() -> None:
    started = time.perf_counter()
    r = run_in_sandbox(ToolCall("c1", "hang", {}), _reg(), timeout_s=0.3)
    elapsed = time.perf_counter() - started
    assert r.ok is False
    assert "timeout" in r.error
    assert elapsed < 2.0, f"超时后应立即返回，实际 {elapsed:.2f}s"


def test_output_truncated() -> None:
    r = run_in_sandbox(ToolCall("c1", "long_out", {}), _reg(), max_output_chars=100)
    assert r.ok is True
    assert "truncated" in r.output
    assert len(r.output) < 300

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
