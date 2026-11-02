import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402

from harness.tracing import Tracer, trace_summary, trace_to_dict, trace_to_json  # noqa: E402
from harness.types import Step, ToolCall, ToolResult, Trace  # noqa: E402


def _sample_trace() -> Trace:
    step = Step(index=0, thought="我需要算一下")
    step.calls.append(ToolCall("c1", "add", {"a": 1, "b": 2}))
    step.results.append(ToolResult(call_id="c1", ok=True, output="3", duration_ms=1))
    return Trace(task="1+2", steps=[step], final_answer="3", stop_reason="converged")


def test_trace_to_dict() -> None:
    d = trace_to_dict(_sample_trace())
    assert d["task"] == "1+2"
    assert d["stop_reason"] == "converged"
    assert d["step_count"] == 1
    assert d["tool_call_count"] == 1
    assert d["steps"][0]["results"][0]["output"] == "3"


def test_trace_to_json() -> None:
    s = trace_to_json(_sample_trace())
    parsed = json.loads(s)
    assert parsed["final_answer"] == "3"


def test_trace_summary() -> None:
    s = trace_summary(_sample_trace())
    assert "steps=1" in s
    assert "tools=1" in s
    assert "converged" in s


def test_empty_trace() -> None:
    d = trace_to_dict(Trace(task="x"))
    assert d["step_count"] == 0
    assert d["tool_call_count"] == 0


def test_tracer_builds_trace() -> None:
    t = Tracer(task="demo")
    t.start_step(0, thought="想一想")
    t.end_step()
    tr = t.to_trace(final_answer="好了", stop_reason="converged")
    assert tr.task == "demo"
    assert len(tr.steps) == 1
    assert tr.steps[0].thought == "想一想"

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
