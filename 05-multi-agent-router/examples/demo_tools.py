"""演示：7 个工具 + 幂等写操作 + 高风险操作。

    python examples/demo_tools.py

★ 这个 demo 想展示的不是"工具能调通"，而是**三种工具性质的差别**：
    读操作     —— 随便调，不产生副作用
    写操作     —— 必须带幂等键，重复调用不会重复创建
    高风险操作 —— 不可撤销，必须先校验前置条件

看点：
    第 3 段 —— 同一个 idempotency_key 调两次，只创建 1 条工单
    第 5 段 —— 缺幂等键会被注册表拦住（不需要工具自己判断）
    第 6 段 —— 高风险操作在工单不存在时直接拒绝
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_agent import (  # noqa: E402
    BizData,
    ToolRegistry,
    default_knowledge_base,
    make_ask_user_tool,
    make_kb_search_tool,
    register_biz_tools,
)


def hr(title: str) -> None:
    print(f"\n{'─' * 74}\n{title}\n{'─' * 74}")


def show(reg: ToolRegistry, name: str, args: dict) -> None:
    r = reg.execute(name, args)
    tag = "OK " if r.ok else "ERR"
    body = r.content if r.ok else r.error
    for i, line in enumerate(body.splitlines()):
        print(f"  [{tag}] {line}" if i == 0 else f"        {line}")


def main() -> int:
    data = BizData()
    reg = ToolRegistry()
    reg.register(make_kb_search_tool(default_knowledge_base(), default_route="troubleshoot"))
    reg.register(make_ask_user_tool())
    register_biz_tools(reg, data)

    print("=" * 74)
    print("工具集演示：读 / 写 / 高风险，三种性质")
    print("=" * 74)
    print(f"已加载 {len(reg.names())} 个工具：")
    kinds = {
        "kb_search": "读",
        "ask_user": "读（交互）",
        "get_customer_history": "读",
        "get_error_code": "读",
        "check_service_status": "读",
        "create_ticket": "★ 写（需幂等键）",
        "escalate_to_human": "★★ 高风险（不可撤销）",
    }
    for n in reg.names():
        print(f"  - {n:24} {kinds.get(n, '')}")

    # ---------- 1. 读操作 ----------
    hr("【1】读操作 —— 随便调，没有副作用")
    show(reg, "get_error_code", {"code": "E1024"})
    show(reg, "check_service_status", {"service": "worker"})
    show(reg, "get_customer_history", {"customer": "华信科技"})
    print(f"\n  → 调了 3 次读操作，工单数仍然是 {len(data.tickets)}")

    # ---------- 2. 错误信息要模型可读 ----------
    hr("【2】错误信息必须「模型可读」—— 告诉它下一步怎么办")
    show(reg, "get_error_code", {"code": "E9999"})       # 不在字典 → 给出可用列表
    show(reg, "check_service_status", {"service": "nope"})
    show(reg, "no_such_tool", {})

    # ---------- 3. ★ 写操作：幂等 ----------
    hr("【3】★ 写操作：同一个幂等键，调两次只创建一条")
    args = {
        "customer": "华信科技",
        "title": "E1024 复发，需人工确认",
        "idempotency_key": "req-2026-11-24-001",
    }
    print("  第 1 次调用：")
    show(reg, "create_ticket", args)
    print(f"  当前工单数 = {len(data.tickets)}")

    print("\n  第 2 次调用（完全相同的重试，同一个 idempotency_key）：")
    show(reg, "create_ticket", args)
    print(f"  当前工单数 = {len(data.tickets)}   ← ★ 没有变成 2")

    print("\n  第 3 次调用（换一个新的 idempotency_key）：")
    show(reg, "create_ticket", {**args, "idempotency_key": "req-2026-11-24-002"})
    print(f"  当前工单数 = {len(data.tickets)}   ← 这才应该 +1")

    print(f"\n  幂等统计：命中 {data.idem.hits} 次 / 首次执行 {data.idem.misses} 次")

    # ---------- 4. 幂等命中返回原结果 ----------
    hr("【4】幂等命中时，返回的是**第一次的结果**，不是重新执行")
    first = reg.execute("create_ticket", {**args, "idempotency_key": "req-x-001"})
    again = reg.execute("create_ticket", {**args, "idempotency_key": "req-x-001"})
    import re
    id1 = re.search(r"T\d+", first.content)
    id2 = re.search(r"T\d+", again.content)
    print(f"  第一次返回工单号：{id1.group() if id1 else '?'}")
    print(f"  重试返回工单号　：{id2.group() if id2 else '?'}")
    print(f"  → {'★ 同一个工单号，正确' if id1 and id2 and id1.group() == id2.group() else '✗ 异常'}")

    # ---------- 5. 缺幂等键被拦住 ----------
    hr("【5】缺幂等键 —— 注册表在调用前就拦住了")
    show(reg, "create_ticket", {"customer": "恒远数据", "title": "没有幂等键"})
    print("  → 工具根本没被执行，工单数不变")

    # ---------- 6. 高风险操作 ----------
    hr("【6】★ 高风险操作：先校验前置条件，再执行不可撤销动作")
    print("  情况 A：工单不存在")
    show(reg, "escalate_to_human", {"ticket_id": "T9999", "reason": "试试"})

    tid = data.tickets[0].ticket_id
    print(f"\n  情况 B：工单 {tid} 存在")
    show(reg, "escalate_to_human", {"ticket_id": tid, "reason": "涉及退款合规，AI 无权处理"})
    print(f"  该工单状态已变为：{data.tickets[0].status}")

    print("\n  情况 C：缺 reason")
    show(reg, "escalate_to_human", {"ticket_id": tid})

    # ---------- 7. 汇总 ----------
    hr("【7】最终状态")
    print(f"  工具总数：{len(reg.names())}")
    print(f"  工单总数：{len(data.tickets)}")
    for t in data.tickets:
        print(f"    {t.ticket_id}  {t.status:10}  {t.customer}  「{t.title}」")
    print()
    print("=" * 74)
    print("三个可带走的结论：")
    print("  ① 读操作天然幂等 → 随便重试")
    print("  ② 写操作必须带幂等键，否则重试 = 重复下单")
    print("  ③ 高风险操作要先校验前置条件，再执行不可撤销动作")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
