"""路由节点：判定工单属于哪一类。

这是整个系统里**唯一一个"模型做决策"的地方**——分派给哪个专家由 LLM 判断。
其余流程（检索 → 生成 → 校验 → 沉淀）都是代码写死的。

★ 这个划分很重要：
  - 把「判断」交给模型：灵活，能处理未见过的说法
  - 把「流程」写死：可控、可测、可审计
  纯自主 Agent 不可控，纯规则系统不灵活——两者之间的这个点才是生产可用的。
"""
from __future__ import annotations

from .knowledge import KnowledgeBase
from .llm import LLM, extract_json
from .state import ROUTE_LABELS, TicketState

ROUTE_INSTALL = "install"
ROUTE_USAGE = "usage"
ROUTE_TROUBLE = "troubleshoot"
ROUTE_FALLBACK = "fallback"

# 低于该置信度就不硬分派，走兜底——分错的代价比"承认不确定"更大
CONFIDENCE_THRESHOLD = 0.7

SYSTEM = f"""你是工单分派器。把用户工单归到下面四类之一：
- install：安装、部署、环境配置、许可证激活
- usage：功能使用、操作步骤、参数含义
- troubleshoot：报错、异常、故障、性能问题
- fallback：信息不足、与产品无关、或属于其他类型

只输出一个 JSON 对象，不要任何解释：
{{"route": "install|usage|troubleshoot|fallback",
  "confidence": 0.0 到 1.0,
  "reason": "一句话说明依据",
  "need_clarify": true 或 false}}

判断规则：
1. 工单里出现具体报错信息、错误码、"打不开"、"失败" → troubleshoot
2. 问"怎么…"、"在哪里"、"是什么意思" → usage
3. 问"如何装"、"要什么环境"、"激活" → install
4. 信息不足以判断（比如只有"你好"、"有个问题"）→ fallback 且 need_clarify=true
5. 拿不准就给低 confidence，不要硬猜
"""


def route_node(state: TicketState, llm: LLM) -> dict:
    """返回路由判定。LLM 输出非法时安全降级到 fallback。"""
    user = state.ticket.text.strip()
    if not user:
        return {
            "route": ROUTE_FALLBACK,
            "route_confidence": 0.0,
            "route_reason": "工单内容为空",
            "need_clarify": True,
        }

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        raw = llm.chat(messages, temperature=0.0)
        data = extract_json(raw)
        route = str(data.get("route", "")).strip().lower()
        confidence = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", "")).strip()
        need_clarify = bool(data.get("need_clarify", False))
    except Exception as exc:  # noqa: BLE001  模型输出不可信时也不能崩
        return {
            "route": ROUTE_FALLBACK,
            "route_confidence": 0.0,
            "route_reason": f"路由输出无法解析（{type(exc).__name__}），安全降级",
            "need_clarify": True,
        }

    if route not in ROUTE_LABELS:
        route = ROUTE_FALLBACK
        confidence = 0.0
        reason = f"模型返回了未知类别，安全降级：{reason}"

    confidence = max(0.0, min(1.0, confidence))

    # 低置信度不硬分派
    if confidence < CONFIDENCE_THRESHOLD and route != ROUTE_FALLBACK:
        return {
            "route": ROUTE_FALLBACK,
            "route_confidence": confidence,
            "route_reason": f"置信度 {confidence:.2f} 低于阈值 {CONFIDENCE_THRESHOLD}，转兜底。原判：{reason}",
            "need_clarify": True,
        }

    return {
        "route": route,
        "route_confidence": confidence,
        "route_reason": reason,
        "need_clarify": need_clarify,
    }


def next_after_route(state: TicketState) -> str:
    """条件边：把 route 映射到具体的专家节点。"""
    return {
        ROUTE_INSTALL: "expert_install",
        ROUTE_USAGE: "expert_usage",
        ROUTE_TROUBLE: "expert_troubleshoot",
        ROUTE_FALLBACK: "expert_fallback",
    }.get(state.route, "expert_fallback")
