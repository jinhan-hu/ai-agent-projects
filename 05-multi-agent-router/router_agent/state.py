"""贯穿整个流程的状态与数据结构。

设计要点：
- State 用 dataclass 而非 dict：字段拼错会立刻报错，而不是静默塞进一个不存在的 key。
- trace 记录每个节点「谁执行、花多久、改了什么」，是排查问题的唯一线索。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


# 路由目标
ROUTES = ("install", "usage", "troubleshoot", "fallback")

ROUTE_LABELS = {
    "install": "安装部署",
    "usage": "功能使用",
    "troubleshoot": "故障排查",
    "fallback": "通用兜底",
}


@dataclass
class Ticket:
    """一张用户工单。"""

    ticket_id: str
    text: str
    customer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepRecord:
    """一个节点的一次执行记录。"""

    node: str
    started_at: float
    duration_ms: int
    changed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TicketState:
    """图的状态。所有节点读它、返回一个增量 dict，由图负责合并。"""

    # 输入
    ticket: Ticket
    history: list[dict] = field(default_factory=list)      # 本会话历史轮次

    # 路由结果
    route: str = ""
    route_confidence: float = 0.0
    route_reason: str = ""
    need_clarify: bool = False

    # 专家产出
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    # 质量校验
    verified: bool = False
    verify_note: str = ""

    # 记忆
    memory_context: str = ""
    memory_written: bool = False

    # 异常
    error: str = ""

    # 轨迹
    trace: list[StepRecord] = field(default_factory=list)

    # ---------- 便捷方法 ----------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["ticket"] = self.ticket.to_dict()
        d["trace"] = [t.to_dict() for t in self.trace]
        d["route_label"] = ROUTE_LABELS.get(self.route, self.route)
        return d

    def summary(self) -> str:
        return (
            f"route={self.route or '-'}({ROUTE_LABELS.get(self.route, '-')}) "
            f"conf={self.route_confidence:.2f} "
            f"citations={len(self.citations)} "
            f"verified={self.verified} "
            f"steps={len(self.trace)}"
        )

    def record(self, node: str, started_at: float, changed: list[str]) -> None:
        self.trace.append(
            StepRecord(
                node=node,
                started_at=started_at,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                changed=changed,
            )
        )

    def apply(self, update: dict[str, Any]) -> None:
        """把节点返回的增量写回状态。未知字段直接报错——防止拼写错误被吞掉。"""
        for k, v in update.items():
            if not hasattr(self, k):
                raise AttributeError(f"状态里没有字段 {k!r}，检查节点返回值")
            setattr(self, k, v)
