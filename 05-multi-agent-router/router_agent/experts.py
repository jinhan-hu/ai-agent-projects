"""四个专家 Agent。

两种执行模式（★ 这是本项目一个重要的兼容设计）：

    ① 工具模式（默认）：把 kb_search / ask_user 交给模型，由它自主决定
       调不调、调几次 —— 这是「Agent」该有的样子。
    ② 固定模式（回退）：直接调 kb.search() 再生成 —— 当 LLM 不支持
       工具调用时自动降级，保证在任意模型上都能跑。

为什么保留两种：
    不是"为了兼容而兼容"。很多自建模型/旧端点不支持 function calling，
    一个会直接崩的框架没有生产价值。**优雅降级**本身就是工程要求。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .knowledge import KnowledgeBase
from .llm import LLM
from .state import ROUTE_LABELS, TicketState
from .tool_loop import build_tool_messages, run_tool_loop
from .tools import ToolRegistry, make_ask_user_tool, make_kb_search_tool


@dataclass
class ExpertConfig:
    route: str
    persona: str          # 角色设定
    task: str             # 输出要求
    keywords: tuple[str, ...] = ()   # 用于在回答里锚定关键结论


class ExpertBase:
    """专家 Agent 基类。"""

    config: ExpertConfig

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    # ---------- 子类可覆盖 ----------
    def build_system(self, use_tools: bool = False) -> str:
        tool_rule = (
            "【工具使用】\n"
            "你可以调用 kb_search 检索产品资料。"
            "**在回答任何涉及产品功能、参数、报错、步骤的问题之前，必须先调用它**。\n"
            "如果缺少关键信息无法判断，可以调用 ask_user 向用户提问，"
            "但提问后仍要基于已有信息给出初步建议。\n"
            if use_tools
            else ""
        )
        return (
            f"{self.config.persona}\n\n"
            f"{tool_rule}"
            f"【回答要求】\n"
            f"1. 只根据检索到的资料回答；资料里没有的信息，直接说"
            f"「资料中没有相关内容，建议转人工」，不要编造。\n"
            f"2. 回答要具体可执行：给出步骤、参数或判断依据。\n"
            f"3. 每引用一条资料，在句末用 [来源: xxx] 标注。\n"
            f"4. {self.config.task}\n"
        )

    def build_user(self, state: TicketState, material: str = "") -> str:
        parts = [f"【工单原文】\n{state.ticket.text}"]
        if state.memory_context:
            parts.append(f"【客户历史档案】\n{state.memory_context}")
        if state.history:
            parts.append(f"【本会话前几轮】\n{_render_history(state.history)}")
        if material:
            parts.append(f"【资料】\n{material}")
        if state.need_clarify:
            parts.append(
                "【注意】路由器认为信息不足，请在回答开头先说明需要用户补充什么，"
                "再给出目前能给的通用建议。"
            )
        return "\n\n".join(parts)

    def retrieve(self, state: TicketState) -> list:
        hits = self.kb.search(state.ticket.text, k=3, route=self.config.route)
        if hits:
            return hits
        return self.kb.search(state.ticket.text, k=3, route=None)

    def make_registry(self) -> ToolRegistry:
        """装配本专家的工具集。"""
        reg = ToolRegistry()
        reg.register(make_kb_search_tool(self.kb, default_route=self.config.route))
        reg.register(make_ask_user_tool())
        return reg

    # ---------- 主流程 ----------
    def run(self, state: TicketState, llm: LLM, enable_tools: bool = True) -> dict:
        """执行专家。能走工具就走工具，否则回退固定模式。"""
        can_use_tools = enable_tools and callable(getattr(llm, "chat_with_tools", None))
        if can_use_tools:
            return self._run_with_tools(state, llm)
        return self._run_fixed(state, llm)

    # ---------- 模式 ①：工具调用 ----------
    def _run_with_tools(self, state: TicketState, llm) -> dict:
        registry = self.make_registry()
        messages = build_tool_messages(
            self.build_system(use_tools=True), self.build_user(state)
        )
        result = run_tool_loop(llm, messages, registry, max_rounds=4)
        return {
            "answer": result.answer.strip(),
            "citations": _extract_citations(result.answer),
            "next_steps": _extract_next_steps(result.answer),
            # ★ 新增：工具使用统计（供 trace / 评估使用）
            "tool_stats": result.stats(),
            "asked_user": result.asked_user,
        }

    # ---------- 模式 ②：固定检索（回退） ----------
    def _run_fixed(self, state: TicketState, llm) -> dict:
        hits = self.retrieve(state)
        material = (
            "\n\n".join(
                f"[{doc.source or doc.doc_id}] {doc.title}\n{doc.text}" for doc, _ in hits
            )
            if hits
            else "（未检索到相关资料）"
        )
        messages = [
            {"role": "system", "content": self.build_system(use_tools=False)},
            {"role": "user", "content": self.build_user(state, material)},
        ]
        answer = llm.chat(messages, temperature=0.2).strip()
        return {
            "answer": answer,
            "citations": _extract_citations(answer),
            "next_steps": _extract_next_steps(answer),
            "tool_stats": {"rounds": 0, "tool_calls": 0, "used_tools": False, "mode": "fixed"},
            "asked_user": [],
        }


# ---------- 四个专家 ----------

class InstallExpert(ExpertBase):
    config = ExpertConfig(
        route="install",
        persona="你是产品安装部署专家，熟悉环境准备、依赖安装、许可证激活。",
        task="给出可直接执行的安装步骤，标明前置条件和常见坑。",
        keywords=("安装", "部署", "环境", "激活", "许可证"),
    )


class UsageExpert(ExpertBase):
    config = ExpertConfig(
        route="usage",
        persona="你是产品功能使用专家，熟悉各模块的操作路径与参数含义。",
        task="给出操作路径（从哪个入口点到哪一步）和关键参数说明。",
        keywords=("使用", "功能", "参数", "入口", "设置"),
    )


class TroubleshootExpert(ExpertBase):
    config = ExpertConfig(
        route="troubleshoot",
        persona="你是故障排查专家，擅长从报错信息定位根因。",
        task="先给出最可能的原因（按概率排序），再给每一条对应的验证方法。",
        keywords=("报错", "失败", "异常", "错误", "超时", "打不开"),
    )


class FallbackExpert(ExpertBase):
    config = ExpertConfig(
        route="fallback",
        persona="你是通用客服。当前问题无法确定具体类型，或信息不足。",
        task="先用一句话请用户补充关键信息（是什么问题、在哪一步、有没有报错），"
             "再给出通用的自助排查建议。不要假装知道具体答案。",
        keywords=("补充", "确认", "信息"),
    )

    def retrieve(self, state: TicketState) -> list:
        # 兜底专家不做定向检索：它的问题就是"不知道属于哪类"
        return []

    def make_registry(self) -> ToolRegistry:
        """兜底专家不用 kb_search（不知道查哪个类），只保留 ask_user。"""
        reg = ToolRegistry()
        reg.register(make_ask_user_tool())
        return reg


EXPERTS: dict[str, type[ExpertBase]] = {
    "install": InstallExpert,
    "usage": UsageExpert,
    "troubleshoot": TroubleshootExpert,
    "fallback": FallbackExpert,
}


# ---------- 辅助 ----------

def _render_history(history: list[dict]) -> str:
    lines = []
    for i, turn in enumerate(history[-3:], 1):
        lines.append(f"[第{i}轮] 用户：{turn.get('text', '')[:80]}")
        if turn.get("answer"):
            lines.append(f"　　　回答摘要：{turn['answer'][:80]}")
    return "\n".join(lines)


def _extract_citations(answer: str) -> list[str]:
    """只从**回答正文**里抽 [来源: xxx]。

    ★ 刻意不回落到「检索到的文档」：
      模型检索到了却没在回答里标注出处，本身就是缺陷——
      如果这里帮它兜底，「没有引用」这个信号就会被掩盖，校验也就失效了。
    """
    found = re.findall(r"\[来源[:：]\s*([^\]]+)\]", answer)
    return sorted({f.strip() for f in found if f.strip()})


def _extract_next_steps(answer: str) -> list[str]:
    """抓「下一步」类条目，供工单系统展示。"""
    lines = [ln.strip() for ln in answer.splitlines()]
    steps = []
    for ln in lines:
        m = re.match(r"^(?:\d+[.、)]|[-*•])\s*(.{4,})$", ln)
        if m:
            steps.append(m.group(1)[:100])
    return steps[:5]
