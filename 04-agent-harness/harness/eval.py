"""评测：对比不同 Agent 策略的成功率与 token 成本。

评测是这个项目最重要的部分 —— 因为它把"我觉得有效"变成"数据上有效"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalCase:
    """一条评测用例。"""

    task: str
    expected: str
    checker: str = "contains"   # contains / exact / regex


@dataclass
class EvalReport:
    """评测报告。"""

    total: int = 0
    passed: int = 0
    total_tokens: int = 0
    total_ms: int = 0

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0


def evaluate(cases: list[EvalCase], loop: Any) -> EvalReport:
    """跑一批用例，输出报告。

    TODO(11-07): 实现：
      - 对每个 case 调 loop.run()
      - 用 checker 判定是否通过
      - 累计 token 与耗时
    """
    _ = (cases, loop)
    raise NotImplementedError("评测待实现（阅读周 11-07）")
