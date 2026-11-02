"""评测：对比不同 Agent 策略的成功率与 token 成本。

评测是这个项目最重要的部分 —— 因为它把「我觉得有效」变成「数据上有效」。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """一条评测用例。"""

    task: str
    expected: str
    checker: str = "contains"   # contains / exact / regex


@dataclass
class EvalResult:
    """单条用例的结果。"""

    case: EvalCase
    answer: str
    passed: bool
    stop_reason: str
    steps: int
    duration_ms: int


@dataclass
class EvalReport:
    """评测报告。"""

    results: list[EvalResult] = field(default_factory=list)
    total_tokens: int = 0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_steps(self) -> float:
        return (
            sum(r.steps for r in self.results) / self.total if self.total else 0.0
        )

    @property
    def avg_duration_ms(self) -> float:
        return (
            sum(r.duration_ms for r in self.results) / self.total
            if self.total
            else 0.0
        )

    def summary(self) -> str:
        return (
            f"通过 {self.passed}/{self.total} "
            f"({self.accuracy:.0%}) | "
            f"平均步数 {self.avg_steps:.2f} | "
            f"平均耗时 {self.avg_duration_ms:.1f} ms"
        )


def check(answer: str, case: EvalCase) -> bool:
    """按 checker 判定答案是否正确。"""
    if case.checker == "exact":
        return answer.strip() == case.expected.strip()
    if case.checker == "regex":
        return re.search(case.expected, answer) is not None
    # 默认 contains
    return case.expected in answer


def evaluate(cases: list[EvalCase], loop: Any) -> EvalReport:
    """跑一批用例，输出报告。"""
    report = EvalReport()
    for case in cases:
        started = time.perf_counter()
        trace = loop.run(case.task)
        ms = int((time.perf_counter() - started) * 1000)
        answer = trace.final_answer or ""
        report.results.append(
            EvalResult(
                case=case,
                answer=answer,
                passed=check(answer, case),
                stop_reason=trace.stop_reason,
                steps=len(trace.steps),
                duration_ms=ms,
            )
        )
        report.total_tokens += trace.total_tokens
    return report
