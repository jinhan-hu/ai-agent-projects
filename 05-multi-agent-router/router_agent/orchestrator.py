"""编排：把路由、专家、校验、记忆串成一张图。

图结构：

    load_memory ──► route ──(条件边)──┬─► expert_install ─────┐
                                      ├─► expert_usage ───────┤
                                      ├─► expert_troubleshoot ┤
                                      └─► expert_fallback ────┤
                                                              ▼
                                                          verify ──► remember ──► END

三条设计取舍：
1. **只在 route 一处用 LLM 做决策**，其余节点是确定性代码 → 可测、可审。
2. **verify 是独立的节点**，不塞进专家里 —— 生成和检查分离，检查才能发现生成的问题。
3. **remember 在最后**，且失败不影响回答 —— 记忆是增值功能，不是主链路。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from .experts import EXPERTS, ExpertBase
from .graph import Graph
from .knowledge import Doc, KnowledgeBase
from .knowledge_data import DOCS as SAMPLE_DOCS
from .llm import LLM
from .memory import Memory
from .router import next_after_route, route_node
from .state import ROUTE_LABELS, Ticket, TicketState


@dataclass
class RouterConfig:
    max_steps: int = 20
    verify_min_citations: int = 1     # 至少几条引用才算通过
    # ★ 是否让专家走工具调用循环（LLM 不支持时自动回退）
    enable_tools: bool = True


class RouterOrchestrator:
    """多智能体路由系统的对外入口。"""

    def __init__(
        self,
        llm: LLM,
        kb: KnowledgeBase,
        memory: Memory | None = None,
        config: RouterConfig | None = None,
    ) -> None:
        self.llm = llm
        self.kb = kb
        self.memory = memory or Memory()
        self.config = config or RouterConfig()
        self.experts: dict[str, ExpertBase] = {
            route: cls(kb) for route, cls in EXPERTS.items()
        }
        self.graph = self._build()

    # ---------- 构图 ----------
    def _build(self) -> Graph:
        g = Graph()
        g.add_node("load_memory", self._load_memory)
        g.add_node("route", lambda s: route_node(s, self.llm))

        for route in EXPERTS:
            g.add_node(f"expert_{route}", self._make_expert_node(route))

        g.add_node("verify", self._verify)
        g.add_node("remember", self._remember)

        g.set_entry("load_memory")
        g.add_edge("load_memory", "route")
        g.add_conditional("route", next_after_route)
        for route in EXPERTS:
            g.add_edge(f"expert_{route}", "verify")
        g.add_edge("verify", "remember")
        g.add_edge("remember", "__end__")
        return g

    # ---------- 节点 ----------
    def _load_memory(self, state: TicketState) -> dict:
        """把长期档案和会话历史取出来，注入后续节点。"""
        profile = self.memory.profile_context(state.ticket.customer)
        history = self.memory.session_history(state.ticket.ticket_id)
        parts = []
        if profile:
            parts.append(profile)
        return {
            "memory_context": "\n".join(parts),
            "history": history,
        }

    def _make_expert_node(self, route: str):
        expert = self.experts[route]

        def node(state: TicketState) -> dict:
            out = expert.run(
                state, self.llm, enable_tools=self.config.enable_tools
            )
            out.setdefault("error", state.error)
            return out

        return node

    def _verify(self, state: TicketState) -> dict:
        """轻量校验：生成与检查分离。

        不调用 LLM —— 用可判定的规则先筛一遍，成本极低。
        生产中还可再加一类：用第二个模型评审（成本翻倍，按需开启）。
        """
        problems: list[str] = []

        if state.error:
            problems.append(f"上游异常：{state.error}")

        if not state.answer.strip():
            problems.append("专家没有产出回答")

        # 兜底专家允许没有引用（它本来就没检索）
        if state.route != "fallback":
            if len(state.citations) < self.config.verify_min_citations:
                problems.append("回答没有标注引用来源，可能是编造")

        # 需要澄清的工单：回答里必须真的向用户提问，否则等于没处理
        if state.need_clarify:
            asked = any(k in state.answer for k in ("？", "?", "请", "补充", "确认"))
            if not asked:
                problems.append("路由判定信息不足，但回答没有向用户索取补充信息")

        verified = not problems
        return {
            "verified": verified,
            "verify_note": "通过" if verified else "；".join(problems),
        }

    def _remember(self, state: TicketState) -> dict:
        """沉淀到两级记忆。失败不影响主流程。"""
        try:
            entry = {
                "text": state.ticket.text[:300],
                "route": state.route,
                "route_label": ROUTE_LABELS.get(state.route, state.route),
                "summary": state.answer[:200],
                "verified": state.verified,
            }
            self.memory.append_profile(state.ticket.customer, entry)
            self.memory.append_turn(state.ticket.ticket_id, entry)
            return {"memory_written": True}
        except Exception as exc:  # noqa: BLE001
            return {"memory_written": False, "error": f"记忆写入失败：{exc}"}

    # ---------- 执行 ----------
    def handle(self, text: str, customer: str = "", session_id: str = "") -> TicketState:
        """处理一张工单。session_id 相同即视为同一会话（多轮）。"""
        ticket = Ticket(
            ticket_id=session_id or f"t-{uuid.uuid4().hex[:8]}",
            text=text,
            customer=customer,
        )
        state = TicketState(ticket=ticket)
        return self.graph.run(state, max_steps=self.config.max_steps)


# ---------- 示例知识库 ----------



def default_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(SAMPLE_DOCS)
