"""测试：业务工具集（读 / 写 / 高风险）与幂等语义。

★ 这批工具测的不是"能不能跑"，是三件工程上最容易出事的事：
    1. 写操作重复调用会不会重复创建（幂等）
    2. 高风险操作会不会被误触发（确认与前置校验）
    3. 工具变多以后，schema 还是不是模型能读懂的样子
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_agent import default_knowledge_base  # noqa: E402
from router_agent.biz_tools import (  # noqa: E402
    BizData,
    IdempotencyStore,
    make_check_service_status_tool,
    make_create_ticket_tool,
    make_escalate_to_human_tool,
    make_get_customer_history_tool,
    make_get_error_code_tool,
    register_biz_tools,
)
from router_agent.tools import (  # noqa: E402
    ToolRegistry,
    make_ask_user_tool,
    make_kb_search_tool,
)

PASS = 0
FAIL = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {name}")
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  FAIL {name}  {detail}")


def fresh() -> tuple[BizData, ToolRegistry]:
    data = BizData()
    reg = ToolRegistry()
    reg.register(make_kb_search_tool(default_knowledge_base(), default_route="troubleshoot"))
    reg.register(make_ask_user_tool())
    register_biz_tools(reg, data)
    return data, reg


# ══════════════════ 1. 幂等存储 ══════════════════
def test_idempotency_store() -> None:
    print("\n[1] 幂等存储")
    store = IdempotencyStore()
    counter = {"n": 0}

    def side_effect():
        counter["n"] += 1
        return {"v": counter["n"]}

    r1, cached1 = store.run("k1", side_effect)
    r2, cached2 = store.run("k1", side_effect)
    check("首次未命中缓存", cached1 is False)
    check("★ 第二次命中缓存", cached2 is True)
    check("★ 副作用只执行一次", counter["n"] == 1, str(counter))
    check("返回同一份结果", r1 == r2, f"{r1} vs {r2}")

    r3, cached3 = store.run("k2", side_effect)
    check("不同 key 重新执行", counter["n"] == 2 and cached3 is False)
    check("不同 key 结果不同", r3 != r1)

    store.clear()
    check("clear 后计数归零", store.hits == 0 and store.misses == 0)
    r4, cached4 = store.run("k1", side_effect)
    check("clear 后同 key 重新执行", cached4 is False and counter["n"] == 3)


# ══════════════════ 2. 读操作 ══════════════════
def test_read_tools() -> None:
    print("\n[2] 读操作")
    data, reg = fresh()

    r = reg.execute("get_error_code", {"code": "E1024"})
    check("错误码可查", r.ok and "许可证" in r.content, r.content[:60])

    r = reg.execute("get_error_code", {"code": "e1024"})
    check("★ 错误码大小写不敏感", r.ok and "许可证" in r.content, r.content[:60])

    r = reg.execute("get_error_code", {"code": "E9999"})
    check("未知错误码给出可用列表", "不在字典中" in r.content and "E1024" in r.content)

    r = reg.execute("check_service_status", {"service": "worker"})
    check("服务状态可查", r.ok and "degraded" in r.content, r.content)

    r = reg.execute("check_service_status", {"service": "nope"})
    check("未知服务给出可用列表", "未知服务" in r.content)

    r = reg.execute("get_customer_history", {"customer": "华信科技"})
    check("客户历史可查", r.ok and "E1024" in r.content, r.content[:80])

    r = reg.execute("get_customer_history", {"customer": "某不存在客户"})
    check("未知客户给出可用列表", "未找到" in r.content)

    check("读操作不产生工单", len(data.tickets) == 0)


# ══════════════════ 3. ★ 写操作：幂等 ══════════════════
def test_create_ticket_idempotent() -> None:
    print("\n[3] ★ 写操作：创建工单的幂等性")
    data, reg = fresh()

    args = {"customer": "华信科技", "title": "E1024 复发", "idempotency_key": "uuid-aaa"}

    r1 = reg.execute("create_ticket", args)
    check("首次创建成功", r1.ok and "已创建工单" in r1.content, r1.content)
    check("工单数为 1", len(data.tickets) == 1, str(len(data.tickets)))

    r2 = reg.execute("create_ticket", args)          # 完全相同的重试
    check("★ 重试不产生新工单", len(data.tickets) == 1, str(len(data.tickets)))
    check("★ 重试命中幂等并说明", "幂等命中" in r2.content, r2.content[:80])

    tid1 = data.tickets[0].ticket_id
    check("★ 重试返回同一个工单号", tid1 in r2.content, r2.content)

    # 换 key → 应当新建
    r3 = reg.execute(
        "create_ticket",
        {"customer": "华信科技", "title": "E1024 复发", "idempotency_key": "uuid-bbb"},
    )
    check("★ 不同 key 产生新工单", len(data.tickets) == 2, str(len(data.tickets)))
    check("新工单号不同", data.tickets[1].ticket_id != tid1)

    # 缺 key → 被注册表拦住
    r4 = reg.execute("create_ticket", {"customer": "X", "title": "Y"})
    check("★ 缺幂等键被拦", not r4.ok and "idempotency_key" in r4.error, r4.error)
    check("缺 key 不产生工单", len(data.tickets) == 2, str(len(data.tickets)))

    # 空 key → 工具内部再拦一次
    r5 = reg.execute(
        "create_ticket", {"customer": "X", "title": "Y", "idempotency_key": "  "}
    )
    check("★ 空幂等键也被拦（双保险）", not r5.ok, r5.error)
    check("空 key 不产生工单", len(data.tickets) == 2, str(len(data.tickets)))

    # 统计
    check("幂等命中计数正确", data.idem.hits == 1, str(data.idem.hits))
    check("首次执行计数正确", data.idem.misses == 2, str(data.idem.misses))


# ══════════════════ 4. ★ 高风险操作 ══════════════════
def test_escalate() -> None:
    print("\n[4] ★ 高风险操作：转人工")
    data, reg = fresh()

    r = reg.execute("escalate_to_human", {"ticket_id": "T1001", "reason": "x"})
    check("★ 工单不存在时拒绝", "不存在" in r.content, r.content[:80])

    reg.execute(
        "create_ticket",
        {"customer": "华信科技", "title": "退款投诉", "idempotency_key": "k1"},
    )
    tid = data.tickets[0].ticket_id

    r = reg.execute("escalate_to_human", {"ticket_id": tid, "reason": "涉及退款合规"})
    check("转人工成功", "已转人工" in r.content, r.content)
    check("★ 明确标注不可撤销", "不可撤销" in r.content)
    check("工单状态已变更", data.tickets[0].status == "escalated")

    r = reg.execute("escalate_to_human", {"ticket_id": tid, "reason": "再转一次"})
    check("重复转人工不崩", r.ok)

    # 缺 reason
    r = reg.execute("escalate_to_human", {"ticket_id": tid})
    check("缺 reason 被拦", not r.ok and "reason" in r.error, r.error)


# ══════════════════ 5. 工具数量增长的影响 ══════════════════
def test_tool_count() -> None:
    print("\n[5] 工具数量与 schema 质量")
    data, reg = fresh()

    names = reg.names()
    check("工具总数为 7", len(names) == 7, str(len(names)))
    check("包含读操作 3 个", sum(1 for n in names if n.startswith("get_") or n.startswith("check_")) == 3)
    check("包含写操作 1 个", "create_ticket" in names)
    check("包含高风险 1 个", "escalate_to_human" in names)
    check("包含基础 2 个", {"kb_search", "ask_user"} <= set(names))

    schemas = reg.to_schema()
    check("schema 数量等于工具数", len(schemas) == 7)
    check("每个 schema 都有 description", all(s["function"]["description"] for s in schemas))
    check("每个 schema 都有 parameters", all(s["function"]["parameters"] for s in schemas))

    # ★ 写操作和高风险工具必须在描述里写明性质
    by_name = {s["function"]["name"]: s["function"] for s in schemas}
    check(
        "★ 写操作描述里写明「写操作」",
        "写操作" in by_name["create_ticket"]["description"],
    )
    check(
        "★ 写操作描述里写明幂等键",
        "idempotency_key" in by_name["create_ticket"]["description"],
    )
    check(
        "★ 高风险描述里写明「高风险」",
        "高风险" in by_name["escalate_to_human"]["description"],
    )
    check(
        "★ 高风险描述里写明「不可撤销」",
        "不可撤销" in by_name["escalate_to_human"]["description"],
    )
    # 每个工具都应写「不要用它做什么」
    check(
        "★ 每个工具都写了反向边界",
        all("不要" in s["function"]["description"] for s in schemas),
        str([s["function"]["name"] for s in schemas if "不要" not in s["function"]["description"]]),
    )

    # enum 约束
    svc = by_name["check_service_status"]["parameters"]["properties"]["service"]
    check("★ 服务名用 enum 约束", "enum" in svc and len(svc["enum"]) == 4, str(svc))

    # 必填项
    req = by_name["create_ticket"]["parameters"]["required"]
    check("★ create_ticket 必填包含幂等键", "idempotency_key" in req, str(req))


def main() -> int:
    print("=" * 62)
    print("业务工具集 · 测试")
    print("=" * 62)
    test_idempotency_store()
    test_read_tools()
    test_create_ticket_idempotent()
    test_escalate()
    test_tool_count()

    print("\n" + "=" * 62)
    print(f"通过 {PASS} / {PASS + FAIL}")
    if FAILED:
        print("失败项：")
        for f in FAILED:
            print("  -", f)
    print("=" * 62)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
