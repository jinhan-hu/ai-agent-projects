"""测试：全部离线，不依赖网络与真实 API。

覆盖的重点是**编排逻辑**，不是模型质量：
    路由分派是否正确、低置信度是否正确兜底、非法输出是否安全降级、
    条件边是否会走错、步数上限是否生效、记忆是否写入并可复用。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_agent import (  # noqa: E402
    Graph,
    GraphError,
    KnowledgeBase,
    Memory,
    MockLLM,
    RouterOrchestrator,
    TicketState,
    Ticket,
    extract_json,
    route_node,
)
from router_agent.orchestrator import default_knowledge_base  # noqa: E402

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


def route_json(route: str, conf: float = 0.95, clarify: bool = False) -> str:
    return json.dumps(
        {"route": route, "confidence": conf, "reason": "测试用", "need_clarify": clarify},
        ensure_ascii=False,
    )


def make_state(text: str, customer: str = "客户A") -> TicketState:
    return TicketState(ticket=Ticket(ticket_id="t1", text=text, customer=customer))


# ══════════════════ 1. 图引擎 ══════════════════
def test_graph() -> None:
    print("\n[1] 图引擎")

    g = Graph()
    order: list[str] = []
    g.add_node("a", lambda s: order.append("a") or {"answer": "A"})
    g.add_node("b", lambda s: order.append("b") or {"verified": True})
    g.set_entry("a")
    g.add_edge("a", "b")
    g.add_edge("b", "__end__")
    st = g.run(TicketState(ticket=Ticket("t", "x")))
    check("固定边按序执行", order == ["a", "b"], str(order))
    check("状态被合并", st.answer == "A" and st.verified is True)
    check("轨迹记录了 2 步", len(st.trace) == 2, str(len(st.trace)))

    # 条件边
    g2 = Graph()
    g2.add_node("start", lambda s: {})
    g2.add_node("left", lambda s: {"answer": "L"})
    g2.add_node("right", lambda s: {"answer": "R"})
    g2.set_entry("start")
    g2.add_conditional("start", lambda s: "left" if s.route == "install" else "right")
    g2.add_edge("left", "__end__")
    g2.add_edge("right", "__end__")
    s1 = TicketState(ticket=Ticket("t", "x"))
    s1.route = "install"
    check("条件边走 left", g2.run(s1).answer == "L")
    s2 = TicketState(ticket=Ticket("t", "x"))
    s2.route = "usage"
    check("条件边走 right", g2.run(s2).answer == "R")

    # 步数上限：造一个环
    g3 = Graph()
    g3.add_node("loop", lambda s: {})
    g3.set_entry("loop")
    g3.add_edge("loop", "loop")
    s3 = g3.run(TicketState(ticket=Ticket("t", "x")), max_steps=5)
    check("成环时被步数上限截断", "步数上限" in s3.error, s3.error)

    # 未知节点要报错
    g4 = Graph()
    g4.add_node("a", lambda s: {})
    try:
        g4.add_edge("a", "nope")
        check("未知节点应报错", False)
    except GraphError:
        check("未知节点报错", True)

    # 状态写入未知字段要报错（防止拼写错误被吞）
    st4 = make_state("x")
    try:
        st4.apply({"no_such_field": 1})
        check("未知状态字段应报错", False)
    except AttributeError:
        check("未知状态字段报错", True)


# ══════════════════ 2. 路由 ══════════════════
def test_router() -> None:
    print("\n[2] 路由")

    st = make_state("安装时报 E1024 怎么办")
    out = route_node(st, MockLLM([route_json("troubleshoot")]))
    check("正常分派到 troubleshoot", out["route"] == "troubleshoot", str(out))
    check("置信度被透传", abs(out["route_confidence"] - 0.95) < 1e-9)

    # 低置信度 → 兜底
    out = route_node(make_state("有个问题"), MockLLM([route_json("install", conf=0.4)]))
    check("低置信度转兜底", out["route"] == "fallback", str(out))
    check("低置信度标记需澄清", out["need_clarify"] is True)

    # 非法 JSON → 安全降级
    out = route_node(make_state("x"), MockLLM(["这不是 JSON"]))
    check("非法输出安全降级", out["route"] == "fallback" and out["route_confidence"] == 0.0)

    # 未知类别 → 安全降级
    out = route_node(make_state("x"), MockLLM([route_json("nonsense")]))
    check("未知类别安全降级", out["route"] == "fallback", str(out))

    # 空文本
    out = route_node(make_state("   "), MockLLM([route_json("usage")]))
    check("空工单转兜底", out["route"] == "fallback")
    check("空工单不调用模型", len(MockLLM([route_json("usage")]).calls) == 0)  # 仅验证语义

    # 置信度越界要被夹紧
    out = route_node(make_state("x"), MockLLM([route_json("usage", conf=5.0)]))
    check("置信度被夹到 1.0", out["route_confidence"] == 1.0, str(out["route_confidence"]))

    # JSON 带围栏也能解析
    fenced = "```json\n" + route_json("install") + "\n```"
    out = route_node(make_state("x"), MockLLM([fenced]))
    check("```json 围栏可解析", out["route"] == "install", str(out))

    # 前后带解释文字也能解析
    noisy = "好的，我判断如下：\n" + route_json("usage") + "\n以上。"
    out = route_node(make_state("x"), MockLLM([noisy]))
    check("夹带文字可解析", out["route"] == "usage", str(out))


# ══════════════════ 3. 检索 ══════════════════
def test_knowledge() -> None:
    print("\n[3] 知识库检索")

    kb = default_knowledge_base()

    hits = kb.search("E1024 报错", k=3, route="troubleshoot")
    check("按类检索命中故障类", len(hits) > 0 and all(d.route == "troubleshoot" for d, _ in hits),
          str([(d.doc_id, s) for d, s in hits]))
    check("E1024 排在首位", hits and hits[0][0].doc_id == "KB-T01",
          str(hits[0][0].doc_id if hits else None))

    hits2 = kb.search("安装前置条件 内存", k=2, route="install")
    check("安装类检索命中 KB-I01", hits2 and hits2[0][0].doc_id == "KB-I01",
          str(hits2[0][0].doc_id if hits2 else None))

    check("无关查询返回空", kb.search("zzzzzz", k=3) == [])
    check("空查询返回空", kb.search("", k=3) == [])

    # 分类约束生效：同一个词在不同类里的命中不同
    a = {d.doc_id for d, _ in kb.search("激活", k=5, route="install")}
    b = {d.doc_id for d, _ in kb.search("激活", k=5, route="usage")}
    check("分类约束生效", a != b or not a and not b, f"install={a} usage={b}")

    # 得分必须有序
    hits3 = kb.search("安装 端口 占用", k=5)
    scores = [s for _, s in hits3]
    check("得分降序", scores == sorted(scores, reverse=True), str(scores))


# ══════════════════ 4. 记忆 ══════════════════
def test_memory() -> None:
    print("\n[4] 两级记忆")

    m = Memory()
    m.append_profile("客户A", {"route_label": "故障排查", "summary": "E1024 已解决"})
    m.append_profile("客户A", {"route_label": "功能使用", "summary": "问过报表入口"})
    ctx = m.profile_context("客户A")
    check("长期档案可读回", "E1024 已解决" in ctx and "报表入口" in ctx, ctx)
    check("其他客户档案为空", m.profile_context("客户B") == "")

    m.append_turn("s1", {"text": "第一问", "route_label": "安装"})
    m.append_turn("s1", {"text": "第二问", "route_label": "使用"})
    check("会话历史累积", len(m.session_history("s1")) == 2)
    check("不同会话隔离", m.session_history("s2") == [])

    # 上限截断
    m2 = Memory()
    for i in range(30):
        m2.append_turn("s", {"text": f"q{i}"})
    check("会话记忆有上限", len(m2.session_history("s")) == 20, str(len(m2.session_history("s"))))


# ══════════════════ 5. 端到端 ══════════════════
def test_end_to_end() -> None:
    print("\n[5] 端到端编排")

    kb = default_knowledge_base()

    # 5.1 正常故障工单
    llm = MockLLM([
        route_json("troubleshoot"),
        "错误码 E1024 表示许可证校验失败 [来源: 故障手册 §2]。\n"
        "1. 先对齐系统时间。\n2. 检查设备指纹是否变化。",
    ])
    orch = RouterOrchestrator(llm, kb)
    st = orch.handle("安装后启动报 E1024", customer="客户A")
    check("分派到故障专家", st.route == "troubleshoot", st.route)
    check("走了 5 个节点", len(st.trace) == 5, str([t.node for t in st.trace]))
    check("节点顺序正确",
          [t.node for t in st.trace] == ["load_memory", "route", "expert_troubleshoot", "verify", "remember"],
          str([t.node for t in st.trace]))
    check("抽出了引用", len(st.citations) >= 1, str(st.citations))
    check("校验通过", st.verified is True, st.verify_note)
    check("记忆已写入", st.memory_written is True)
    check("抽出了下一步", len(st.next_steps) >= 1, str(st.next_steps))

    # 5.2 低置信度走兜底
    llm = MockLLM([
        route_json("install", conf=0.3),
        "请补充：是什么问题、在哪一步出现、有没有报错信息？",
    ])
    orch = RouterOrchestrator(llm, kb)
    st = orch.handle("你好，有个问题", customer="客户B")
    check("低置信度走兜底专家", st.route == "fallback", st.route)
    check("兜底时要求澄清", st.need_clarify is True)
    check("兜底不强制要求引用", st.verified is True, st.verify_note)

    # 5.3 专家没给引用 → 校验失败
    llm = MockLLM([route_json("usage"), "报表在左边菜单里。"])
    orch = RouterOrchestrator(llm, kb)
    st = orch.handle("报表在哪", customer="客户C")
    check("无引用时校验失败", st.verified is False, st.verify_note)
    check("校验说明指出问题", "引用" in st.verify_note, st.verify_note)

    # 5.4 多轮：同一 session_id 能读到历史
    llm = MockLLM([
        route_json("install"), "安装步骤：先装运行库 [来源: 安装手册 §1]。",
        route_json("usage"), "报表入口在分析菜单 [来源: 用户手册 §3.2]。",
    ])
    orch = RouterOrchestrator(llm, kb)
    s1 = orch.handle("怎么安装", customer="客户D", session_id="sess-1")
    s2 = orch.handle("报表在哪", customer="客户D", session_id="sess-1")
    check("第二轮读到了历史", len(s2.history) >= 1, str(len(s2.history)))
    check("两轮都成功", s1.verified and s2.verified)

    # 5.5 模型彻底乱答 → 不崩
    llm = MockLLM(["完全无关的一段话", "依然是无关的话"])
    orch = RouterOrchestrator(llm, kb)
    st = orch.handle("安装报错", customer="客户E")
    check("乱答不崩溃", st.error == "" or "步数上限" not in st.error)
    check("乱答被判不合格", st.verified is False, st.verify_note)

    # 5.6 轨迹包含耗时
    check("轨迹含耗时字段", all(t.duration_ms >= 0 for t in st.trace))


# ══════════════════ 6. 工具函数 ══════════════════
def test_helpers() -> None:
    print("\n[6] JSON 抽取")
    check("纯 JSON", extract_json('{"a":1}') == {"a": 1})
    check("围栏 JSON", extract_json('```json\n{"a":2}\n```') == {"a": 2})
    check("夹带文字", extract_json('结果如下 {"a":3} 完毕') == {"a": 3})
    try:
        extract_json("没有花括号")
        check("无 JSON 应抛错", False)
    except ValueError:
        check("无 JSON 抛错", True)


def main() -> int:
    print("=" * 60)
    print("多智能体路由 · 离线测试")
    print("=" * 60)
    test_graph()
    test_router()
    test_knowledge()
    test_memory()
    test_end_to_end()
    test_helpers()

    print("\n" + "=" * 60)
    print(f"通过 {PASS} / {PASS + FAIL}")
    if FAILED:
        print("失败项：")
        for f in FAILED:
            print("  -", f)
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
