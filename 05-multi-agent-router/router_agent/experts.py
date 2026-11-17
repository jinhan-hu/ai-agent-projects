"""四个专家 Agent。

共同骨架（ExpertBase）体现了一件事：**路由之后，专家的上下文是干净的**——
安装专家只拿到安装类的知识库片段，不会混进故障排查的内容。
这是路由相比"单体 Agent 全包"的核心收益。

新增一个类型只需要：继承 ExpertBase → 写 persona 和 keywords → 注册到 EXPERTS。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .knowledge import KnowledgeBase
from .llm import LLM
from .state import ROUTE_LABELS, TicketState


@dataclass
class ExpertConfig:
    route: str
    persona: str          # 角色设定
    task: str             # 输出要求
    keywords: tuple[str, ...] = ()   # 用于在回答里锚定关键结论


class ExpertBase:
    """专家 Agent 基类：检索 → 拼上下文 → 生成 → 抽引用。"""

    config: ExpertConfig

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    # ---------- 子类可覆盖 ----------
    def build_system(self) -> str:
        return (
            f"{self.config.persona}\n\n"
            f"【要求】\n"
            f"1. 只根据下面提供的【资料】回答；资料里没有的信息，直接说"
            f"「资料中没有相关内容，建议转人工」，不要编造。\n"
            f"2. 回答要具体可执行：给出步骤、参数或判断依据。\n"
            f"3. 每引用一条资料，在句末用 [来源: xxx] 标注。\n"
            f"4. {self.config.task}\n"
        )

    def retrieve(self, state: TicketState) -> list:
        # ① 先按本类型检索（强约束）
        hits = self.kb.search(state.ticket.text, k=3, route=self.config.route)
        if hits:
            return hits
        # ② 同类没命中，放宽到全库——但要标记出来，让校验节点知道
        return self.kb.search(state.ticket.text, k=3, route=None)

    # ---------- 主流程 ----------
    def run(self, state: TicketState, llm: LLM) -> dict:
        hits = self.retrieve(state)

        if hits:
            blocks = [
                f"[{doc.source or doc.doc_id}] {doc.title}\n{doc.text}"
                for doc, _ in hits
            ]
            material = "\n\n".join(blocks)
        else:
            material = "（未检索到相关资料）"

        context_parts = []
        if state.memory_context:
            context_parts.append(f"【客户历史档案】\n{state.memory_context}")
        if state.history:
            context_parts.append(f"【本会话前几轮】\n{_render_history(state.history)}")

        user_parts = [f"【工单原文】\n{state.ticket.text}"]
        if context_parts:
            user_parts.append("\n".join(context_parts))
        user_parts.append(f"【资料】\n{material}")
        if state.need_clarify:
            user_parts.append(
                "【注意】路由器认为信息不足，请在回答开头先说明需要用户补充什么，"
                "再给出目前能给的通用建议。"
            )

        messages = [
            {"role": "system", "content": self.build_system()},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        answer = llm.chat(messages, temperature=0.2).strip()

        citations = _extract_citations(answer, hits)
        next_steps = _extract_next_steps(answer)

        return {"answer": answer, "citations": citations, "next_steps": next_steps}


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


def _extract_citations(answer: str, hits: list) -> list[str]:
    """只从**回答正文**里抽 [来源: xxx]。

    ★ 刻意不回落到「检索到的文档」：
      模型检索到了却没在回答里标注出处，本身就是缺陷——
      如果这里帮它兜底，「没有引用」这个信号就会被掩盖，校验也就失效了。
      检索到的出处单独通过 `retrieved` 传递（见 orchestrator 的 trace）。
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
