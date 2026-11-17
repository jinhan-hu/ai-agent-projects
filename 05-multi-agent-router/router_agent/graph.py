"""轻量状态图引擎（约 100 行）。

为什么不用 LangGraph：
    本项目的目的之一是**看清编排的本质**。LangGraph 的核心就是
    「节点 = 一个 (state) -> update 的函数」+「边 = 固定或条件跳转」，
    外加一个带步数上限的循环。手写一遍之后，再用任何框架都不会被抽象迷惑。

与 LangGraph 的对应关系：
    StateGraph         -> Graph
    add_node           -> add_node
    add_edge           -> add_edge
    add_conditional_edges -> add_conditional
    compile()          -> 无需编译（直接可跑）
    invoke()           -> run()
"""
from __future__ import annotations

import time
from typing import Callable

from .state import TicketState

# 节点签名：接收状态，返回增量字典（空字典表示未改动）
NodeFn = Callable[[TicketState], dict]


class GraphError(RuntimeError):
    """图的执行错误。"""


class Graph:
    """极简状态图：节点 + 固定边 + 条件边 + 步数上限。"""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}               # 固定边：src -> dst
        self._conditionals: dict[str, Callable[[TicketState], str]] = {}
        self._entry: str | None = None
        self._end = "__end__"

    # ---------- 构图 ----------
    def add_node(self, name: str, fn: NodeFn) -> None:
        if name in self._nodes:
            raise GraphError(f"节点重复注册：{name}")
        if name == self._end:
            raise GraphError(f"{self._end} 是保留名")
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self._require_node(src)
        if dst != self._end:
            self._require_node(dst)
        self._edges[src] = dst

    def add_conditional(self, src: str, chooser: Callable[[TicketState], str]) -> None:
        """条件边：chooser 返回下一个节点名（或 __end__）。"""
        self._require_node(src)
        self._conditionals[src] = chooser

    def set_entry(self, name: str) -> None:
        self._require_node(name)
        self._entry = name

    def _require_node(self, name: str) -> None:
        if name not in self._nodes:
            raise GraphError(f"未注册的节点：{name}")

    # ---------- 执行 ----------
    def run(self, state: TicketState, max_steps: int = 20) -> TicketState:
        """执行到 __end__、无后继、或达到步数上限。

        步数上限是必需的：条件边可能形成环，没有上限就会挂死。
        """
        if self._entry is None:
            raise GraphError("没有设置入口节点（set_entry）")

        current = self._entry
        for _ in range(max_steps):
            started = time.perf_counter()
            update = self._nodes[current](state) or {}
            state.apply(update)
            state.record(current, started, sorted(update.keys()))

            nxt = self._next(current, state)
            if nxt is None or nxt == self._end:
                return state
            current = nxt

        state.error = state.error or f"达到步数上限 {max_steps}，可能是条件边成环"
        return state

    def _next(self, current: str, state: TicketState) -> str | None:
        chooser = self._conditionals.get(current)
        if chooser is not None:
            target = chooser(state)
            if target not in self._nodes and target != self._end:
                raise GraphError(
                    f"条件边 {current} 返回了未注册的节点：{target!r}"
                )
            return target
        return self._edges.get(current)


def mermaid(graph: Graph) -> str:
    """把图导出为 Mermaid 文本，方便贴进文档或 README。"""
    lines = ["graph LR"]
    for name in graph._nodes:
        lines.append(f"    {name}[{name}]")
    for src, dst in graph._edges.items():
        lines.append(f"    {src} --> {dst}")
    for src in graph._conditionals:
        lines.append(f"    {src} -.->|条件边| ???")
    return "\n".join(lines)
