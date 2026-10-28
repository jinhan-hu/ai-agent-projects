"""调用追踪：把一次执行完整记录下来，可序列化、可回放。

没有 tracing 的 Agent 是黑盒 —— 出了问题只能靠猜。
"""
from __future__ import annotations

import json
from typing import Any

from .types import Step, Trace


class Tracer:
    """执行追踪器。

    TODO(11-07): 实现：
      - start_step() / end_step()
      - record_call() / record_result()
      - to_json()：完整 trace 序列化
      - summary()：人类可读的单行摘要（用于日志）
    """

    def __init__(self, task: str) -> None:
        self.trace = Trace(task=task)

    def add_step(self, step: Step) -> None:
        self.trace.steps.append(step)

    def to_json(self, *, indent: int = 2) -> str:
        """序列化成 JSON。"""
        return json.dumps(self._to_dict(), ensure_ascii=False, indent=indent)

    def _to_dict(self) -> dict[str, Any]:
        # TODO(11-07): 完整实现 steps / results 的序列化
        return {"task": self.trace.task, "steps": [], "final_answer": self.trace.final_answer}
