"""调用追踪：把一次执行完整记录下来，可序列化、可回放。

没有 tracing 的 Agent 是黑盒 —— 出了问题只能靠猜。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .types import Step, Trace


def _step_to_dict(step: Step) -> dict[str, Any]:
    return {
        "index": step.index,
        "thought": step.thought,
        "calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments}
            for c in step.calls
        ],
        "results": [
            {
                "call_id": r.call_id,
                "ok": r.ok,
                "output": r.output,
                "error": r.error,
                "duration_ms": r.duration_ms,
            }
            for r in step.results
        ],
    }


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    return {
        "task": trace.task,
        "steps": [_step_to_dict(s) for s in trace.steps],
        "final_answer": trace.final_answer,
        "total_tokens": trace.total_tokens,
        "stop_reason": trace.stop_reason,
        "step_count": len(trace.steps),
        "tool_call_count": sum(len(s.calls) for s in trace.steps),
    }


def trace_to_json(trace: Trace, *, indent: int = 2) -> str:
    return json.dumps(trace_to_dict(trace), ensure_ascii=False, indent=indent)


def trace_summary(trace: Trace) -> str:
    """一行人类可读摘要，适合打日志。"""
    parts = [
        f"task={trace.task[:30]!r}",
        f"steps={len(trace.steps)}",
        f"tools={sum(len(s.calls) for s in trace.steps)}",
        f"stop={trace.stop_reason or 'unknown'}",
    ]
    return " | ".join(parts)


@dataclass
class Tracer:
    """执行追踪器：边跑边记，最后可导出 JSON。"""

    task: str
    steps: list[Step] = field(default_factory=list)
    _open: Step | None = None

    def start_step(self, index: int, thought: str = "") -> Step:
        self._open = Step(index=index, thought=thought)
        return self._open

    def end_step(self) -> Step | None:
        if self._open is not None:
            self.steps.append(self._open)
            self._open = None
        return self.steps[-1] if self.steps else None

    def to_trace(self, *, final_answer: str = "", stop_reason: str = "") -> Trace:
        return Trace(
            task=self.task,
            steps=list(self.steps),
            final_answer=final_answer,
            stop_reason=stop_reason,
        )
