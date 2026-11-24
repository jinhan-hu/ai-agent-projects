"""业务工具：把「读 / 写 / 高风险」三类工具一次讲清。

为什么要有这一层：
    前两批只有 kb_search（读）和 ask_user（提问）。
    ★ 这两种都是"无副作用"的，试不出工具设计里最要命的问题：
       - 写操作怎么防重复（幂等）
       - 高风险操作怎么防误触（确认）
       - 工具从 2 个变 7 个以后，模型还选得对吗

本模块提供 5 个业务工具：

    读操作（天然幂等）
      get_customer_history    查客户历史工单
      get_error_code          查错误码字典
      check_service_status    查服务健康状态

    ★ 写操作（必须幂等）
      create_ticket           创建工单 —— 带 idempotency_key

    ★ 高风险操作（需确认）
      escalate_to_human       转人工 —— 不可撤销

★ 幂等的实现方式（看 IdempotencyStore）：
    服务端记住「这个 key 处理过、结果是什么」。
    重复请求直接返回第一次的结果，不再执行副作用。
    这就是 Stripe / 支付宝那种 "idempotency key" 的做法。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .tools import ToolSpec


# ══════════════════════ 幂等存储 ══════════════════════

class IdempotencyStore:
    """记住「某个幂等键处理过、结果是什么」。

    ★ 三个必须说清的点：
      1. 键由**调用方**生成（Agent），服务端只负责去重
      2. 键必须在**重试之间保持不变**，否则去重失效
      3. 命中缓存时返回**第一次的结果**，不是重新执行
    """

    def __init__(self) -> None:
        self._results: dict[str, dict] = {}
        self.hits = 0          # 命中次数（用于测试和埋点）
        self.misses = 0        # 首次执行次数

    def run(self, key: str, fn):
        if key in self._results:
            self.hits += 1
            return self._results[key], True
        self.misses += 1
        result = fn()
        self._results[key] = result
        return result, False

    def clear(self) -> None:
        self._results.clear()
        self.hits = self.misses = 0


# ══════════════════════ 模拟数据 ══════════════════════

@dataclass
class TicketRecord:
    ticket_id: str
    customer: str
    title: str
    route: str
    status: str = "open"


@dataclass
class BizData:
    """业务数据的模拟实现（内存）。

    ★ 用内存而不是数据库，是为了让「工具调用链路」可测、可复现。
      换真库只需要替换这里的实现，工具层接口不变。
    """

    # 错误码字典
    error_codes: dict[str, str] = field(default_factory=lambda: {
        "E1024": "许可证校验失败：系统时间不准 / 设备指纹变化 / 许可证过期",
        "E2001": "数据库连接池耗尽：检查慢查询与连接泄漏",
        "E2002": "磁盘写入失败：检查剩余空间与挂载权限",
        "E2003": "许可证模块缺失：该版本未包含此模块",
        "E2004": "服务健康检查超时：确认下游依赖是否可用",
    })
    # 服务状态
    service_status: dict[str, dict] = field(default_factory=lambda: {
        "api": {"status": "healthy", "latency_ms": 42, "detail": "所有端点正常"},
        "worker": {"status": "degraded", "latency_ms": 380, "detail": "队列积压 1.2 万条"},
        "db": {"status": "healthy", "latency_ms": 8, "detail": "连接数 12/100"},
        "redis": {"status": "healthy", "latency_ms": 2, "detail": "内存使用 31%"},
    })
    # 客户历史工单
    history: dict[str, list[dict]] = field(default_factory=lambda: {
        "华信科技": [
            {"date": "2026-09-12", "title": "安装时 E1024", "result": "对齐系统时间后解决"},
            {"date": "2026-10-03", "title": "报表导出慢", "result": "清理查询缓存后恢复"},
        ],
        "恒远数据": [
            {"date": "2026-10-21", "title": "worker 队列积压", "result": "扩容 worker 至 4 实例"},
        ],
    })
    # 已创建的工单
    tickets: list[TicketRecord] = field(default_factory=list)
    # 幂等存储
    idem: IdempotencyStore = field(default_factory=IdempotencyStore)
    _seq: itertools.count = field(default_factory=lambda: itertools.count(1001))

    def reset(self) -> None:
        self.tickets.clear()
        self.idem.clear()
        self._seq = itertools.count(1001)


# ══════════════════════ 工具工厂 ══════════════════════

def make_get_customer_history_tool(data: BizData) -> ToolSpec:
    def _run(args: dict) -> str:
        customer = str(args.get("customer", "")).strip()
        rows = data.history.get(customer)
        if rows is None:
            return f"未找到客户「{customer}」的历史记录。可用客户：{', '.join(data.history)}"
        if not rows:
            return f"客户「{customer}」暂无历史工单。"
        lines = [f"客户「{customer}」最近 {len(rows)} 条历史："]
        for r in rows:
            lines.append(f"- {r['date']} {r['title']} → {r['result']}")
        return "\n".join(lines)

    return ToolSpec(
        name="get_customer_history",
        description=(
            "查询某个客户过去提交过的工单及处理结果。\n"
            "① 在处理老客户的问题前应先调用，避免重复给同样的方案；\n"
            "② 不要用它查询客户联系方式或合同信息——那属于 CRM 系统，不在本工具范围；\n"
            "③ customer 传客户全称，不要传拼音或简称。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "customer": {"type": "string", "description": "客户全称，如「华信科技」"},
            },
            "required": ["customer"],
        },
        func=_run,
    )


def make_get_error_code_tool(data: BizData) -> ToolSpec:
    def _run(args: dict) -> str:
        code = str(args.get("code", "")).strip().upper()
        desc = data.error_codes.get(code)
        if not desc:
            return f"错误码 {code} 不在字典中。可用：{', '.join(sorted(data.error_codes))}"
        return f"{code}：{desc}"

    return ToolSpec(
        name="get_error_code",
        description=(
            "查询产品错误码的含义与常见原因。\n"
            "① 用户报错信息里出现 E 开头的四位码时，先用它查清含义再回答；\n"
            "② 不要用它查询 HTTP 状态码（404/500）或数据库错误码——那些不在此字典；\n"
            "③ code 传大写形式，如 E1024。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "错误码，如 E1024"},
            },
            "required": ["code"],
        },
        func=_run,
    )


def make_check_service_status_tool(data: BizData) -> ToolSpec:
    def _run(args: dict) -> str:
        svc = str(args.get("service", "")).strip().lower()
        st = data.service_status.get(svc)
        if not st:
            return f"未知服务「{svc}」。可用：{', '.join(data.service_status)}"
        return (
            f"{svc}: {st['status']} | 延迟 {st['latency_ms']}ms | {st['detail']}"
        )

    return ToolSpec(
        name="check_service_status",
        description=(
            "查询后端某个服务当前的健康状态、延迟与细节。\n"
            "① 用户反馈「打不开/很慢/超时」时，先查服务状态再下结论；\n"
            "② 只覆盖本平台的内部服务，不要用它判断用户自己的网络或第三方服务；\n"
            "③ service 传英文服务名（api / worker / db / redis）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "服务名",
                    "enum": ["api", "worker", "db", "redis"],
                },
            },
            "required": ["service"],
        },
        func=_run,
    )


def make_create_ticket_tool(data: BizData, default_route: str = "fallback") -> ToolSpec:
    """★ 写操作：必须幂等。

    幂等键由调用方（Agent）生成，重试时复用同一个键。
    """

    def _run(args: dict) -> str:
        key = str(args.get("idempotency_key", "")).strip()
        if not key:
            # 理论上 registry 会先拦必填，这里是双保险
            raise ValueError("idempotency_key 不能为空")

        customer = str(args.get("customer", "")).strip()
        title = str(args.get("title", "")).strip()

        def _create() -> dict:
            tid = f"T{next(data._seq)}"
            rec = TicketRecord(
                ticket_id=tid, customer=customer, title=title, route=default_route
            )
            data.tickets.append(rec)
            return {"ticket_id": tid, "customer": customer, "title": title}

        result, cached = data.idem.run(key, _create)
        if cached:
            return (
                f"（幂等命中：该请求此前已处理，未重复创建）"
                f"工单号 {result['ticket_id']}，客户 {result['customer']}，标题「{result['title']}」"
            )
        return (
            f"已创建工单 {result['ticket_id']}：客户 {result['customer']}，"
            f"标题「{result['title']}」"
        )

    return ToolSpec(
        name="create_ticket",
        description=(
            "在工单系统中创建一条新工单。\n"
            "★ 这是写操作，不幂等，重复调用会产生多条工单。\n"
            "① 必须传 idempotency_key（UUID 字符串）；相同的 key 重复调用只会创建一条；\n"
            "② 如果上一次调用超时，重试时请**复用同一个 key**，不要重新生成；\n"
            "③ 只在确认需要人工跟进时才创建，不要为每个问题都建工单。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "customer": {"type": "string", "description": "客户全称"},
                "title": {"type": "string", "description": "工单标题，一句话概括问题"},
                "idempotency_key": {
                    "type": "string",
                    "description": "幂等键（UUID），重试时必须复用同一个",
                },
            },
            "required": ["customer", "title", "idempotency_key"],
        },
        func=_run,
    )


def make_escalate_to_human_tool(data: BizData) -> ToolSpec:
    """★ 高风险操作：不可撤销，需要显式理由。"""

    def _run(args: dict) -> str:
        ticket_id = str(args.get("ticket_id", "")).strip()
        reason = str(args.get("reason", "")).strip()
        exists = any(t.ticket_id == ticket_id for t in data.tickets)
        if not exists:
            return (
                f"工单 {ticket_id} 不存在，无法转人工。"
                f"当前工单：{', '.join(t.ticket_id for t in data.tickets) or '（空）'}"
            )
        for t in data.tickets:
            if t.ticket_id == ticket_id:
                t.status = "escalated"
        return f"工单 {ticket_id} 已转人工，理由：{reason}（此操作不可撤销）"

    return ToolSpec(
        name="escalate_to_human",
        description=(
            "把工单升级给人工客服。\n"
            "★ 这是高风险操作，不可撤销，升级后该工单将被人工接管。\n"
            "① 只在以下情况使用：自己无法解决、涉及退款/合规/情绪激烈的投诉；\n"
            "② 不要一遇到不确定就转人工——先尝试 kb_search 和 ask_user；\n"
            "③ reason 必须具体说明为什么必须转人工，不要写「不清楚」。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "要升级的工单号，如 T1001"},
                "reason": {"type": "string", "description": "转人工的具体理由"},
            },
            "required": ["ticket_id", "reason"],
        },
        func=_run,
    )


def register_biz_tools(registry, data: BizData, default_route: str = "fallback") -> None:
    """把所有业务工具注册进一个 registry。"""
    registry.register(make_get_customer_history_tool(data))
    registry.register(make_get_error_code_tool(data))
    registry.register(make_check_service_status_tool(data))
    registry.register(make_create_ticket_tool(data, default_route=default_route))
    registry.register(make_escalate_to_human_tool(data))
