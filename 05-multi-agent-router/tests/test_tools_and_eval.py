"""测试：工具层 + 工具调用循环 + 检索评估。全部离线。

重点是「模型自主性」和「可靠性」这两件事：
    - 模型决定调 0 次 / 1 次 / 多次工具，都要能正确收敛
    - 工具报错要回填给模型，而不是中断
    - 死循环要被截断
    - 不支持工具调用的 LLM 要优雅降级
    - 评估指标口径要正确
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_agent import (  # noqa: E402
    EvalCase,
    KnowledgeBase,
    MockLLM,
    MockToolLLM,
    RouterConfig,
    RouterOrchestrator,
    ToolRegistry,
    ToolSpec,
    compare_routing,
    default_knowledge_base,
    evaluate,
    make_ask_user_tool,
    make_kb_search_tool,
    run_tool_loop,
)
from router_agent.knowledge_data import DOCS, EVAL_CASES  # noqa: E402
from router_agent.tool_loop import build_tool_messages  # noqa: E402

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


def tc(name: str, arguments: dict | None = None, call_id: str = "") -> dict:
    """构造一个 tool_call。"""
    return {"id": call_id or f"call-{name}", "name": name, "arguments": arguments or {}}


# ══════════════════ 1. 工具注册表 ══════════════════
def test_registry() -> None:
    print("\n[1] 工具注册表")

    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="echo",
            description="原样返回",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            func=lambda a: f"echo:{a['x']}",
        )
    )
    check("注册后能列出", reg.names() == ["echo"])
    check("schema 结构正确", reg.to_schema()[0]["function"]["name"] == "echo")

    r = reg.execute("echo", {"x": "hi"})
    check("正常执行", r.ok and r.content == "echo:hi")

    r = reg.execute("nope", {})
    check("未知工具返回错误而非抛异常", not r.ok and "tool_not_found" in r.error, r.error)
    check("★ 错误信息列出可用工具", "echo" in r.error, r.error)

    r = reg.execute("echo", {})
    check("缺必填参数被拦住", not r.ok and "bad_arguments" in r.error and "x" in r.error, r.error)

    # 工具内部抛异常 → 转成错误
    reg.register(
        ToolSpec(
            name="boom",
            description="抛异常",
            parameters={"type": "object", "properties": {}},
            func=lambda a: (_ for _ in ()).throw(ValueError("炸了")),
        )
    )
    r = reg.execute("boom", {})
    check("工具异常被兜住", not r.ok and "ValueError" in r.error, r.error)

    # 重复注册
    try:
        reg.register(ToolSpec(name="echo", description="", parameters={}, func=lambda a: ""))
        check("重复注册应报错", False)
    except ValueError:
        check("重复注册报错", True)


# ══════════════════ 2. ask_user 工具 ══════════════════
def test_ask_user() -> None:
    print("\n[2] ask_user 工具")

    reg = ToolRegistry()
    reg.register(make_ask_user_tool())
    r = reg.execute("ask_user", {"question": "是哪一步出现的问题？"})
    check("ask_user 执行成功", r.ok, r.error)
    check("★ 标记 needs_user", r.needs_user is True)
    check("回显了问题", "是哪一步出现的问题" in r.content, r.content)

    # kb_search 不应带 needs_user
    kb = default_knowledge_base()
    reg2 = ToolRegistry()
    reg2.register(make_kb_search_tool(kb, default_route="install"))
    r2 = reg2.execute("kb_search", {"query": "安装前置条件"})
    check("kb_search 不标记 needs_user", r2.ok and not r2.needs_user)
    check("kb_search 返回出处", "KB-I01" in r2.content or "安装手册" in r2.content, r2.content[:80])


# ══════════════════ 3. 工具调用循环 ══════════════════
def test_loop() -> None:
    print("\n[3] 工具调用循环")

    kb = default_knowledge_base()
    reg = ToolRegistry()
    reg.register(make_kb_search_tool(kb, default_route="troubleshoot"))
    msgs = build_tool_messages("你是助手", "E1024 怎么处理")

    # 3.1 模型不调工具 → 一次收敛
    llm = MockToolLLM(["直接回答：我不确定 [来源: 无]"])
    res = run_tool_loop(llm, msgs, reg)
    check("不调工具时一轮收敛", res.rounds == 1 and res.stop_reason == "converged")
    check("未使用工具标记正确", res.used_tools is False)
    check("工具 schema 被传进去了", llm.chat_tools_seen() == [1], str(llm.chat_tools_seen()))

    # 3.2 调一次工具再回答
    llm = MockToolLLM([
        {"content": "先查知识库", "tool_calls": [tc("kb_search", {"query": "E1024"})]},
        "E1024 是许可证校验失败 [来源: 故障手册 §2]",
    ])
    res = run_tool_loop(llm, msgs, reg)
    check("调一次工具后收敛", res.stop_reason == "converged")
    check("轮数为 2", res.rounds == 2, str(res.rounds))
    check("★ used_tools=True", res.used_tools is True)
    check("记录了 1 次调用", len(res.tool_calls) == 1, str(len(res.tool_calls)))
    check("调用成功", res.tool_calls[0]["ok"] is True)
    check("答案是模型最终输出", "许可证校验失败" in res.answer, res.answer[:60])
    check("★ 消息历史含 tool 回填", any(m.get("role") == "tool" for m in res.messages))

    # 3.3 调多次工具（模型自主决定）
    llm = MockToolLLM([
        {"content": "先查 E1024", "tool_calls": [tc("kb_search", {"query": "E1024"})]},
        {"content": "再确认一下激活步骤", "tool_calls": [tc("kb_search", {"query": "激活码 步骤"})]},
        "综合两条资料：……[来源: 故障手册 §2][来源: 安装手册 §4]",
    ])
    res = run_tool_loop(llm, msgs, reg)
    check("★ 支持多次自主调用", len(res.tool_calls) == 2, str(len(res.tool_calls)))
    check("多轮轮数正确", res.rounds == 3, str(res.rounds))

    # 3.4 工具报错要回填，循环不中断
    llm = MockToolLLM([
        {"content": "调用一个不存在的工具", "tool_calls": [tc("no_such_tool", {})]},
        "工具不可用，我基于已有信息回答 [来源: 无]",
    ])
    res = run_tool_loop(llm, msgs, reg)
    check("★ 工具报错不中断循环", res.stop_reason == "converged", res.stop_reason)
    check("报错被记录", res.tool_calls[0]["ok"] is False)
    check("错误信息含 tool_not_found", "tool_not_found" in res.tool_calls[0]["error"])
    tool_msgs = [m for m in res.messages if m.get("role") == "tool"]
    check("★ 错误以 ERROR 回填给模型", tool_msgs and "ERROR" in tool_msgs[0]["content"], str(tool_msgs))

    # 3.5 死循环被截断
    same = {"content": "再查一次", "tool_calls": [tc("kb_search", {"query": "E1024"})]}
    llm = MockToolLLM([same] * 10)
    res = run_tool_loop(llm, msgs, reg, max_rounds=6, max_repeat=2)
    check("★ 重复调用被检测并截断", res.stop_reason == "loop_detected", res.stop_reason)
    check("截断时给出可读说明", "重复" in res.answer or "无法" in res.answer, res.answer[:60])

    # 3.6 轮数上限（每轮调用不同工具，规避重复检测）
    replies = [
        {"content": f"第{i}次", "tool_calls": [tc("kb_search", {"query": f"查询{i}"})]}
        for i in range(10)
    ]
    llm = MockToolLLM(replies)
    res = run_tool_loop(llm, msgs, reg, max_rounds=3, max_repeat=99)
    check("★ 达到轮数上限后软着陆", res.stop_reason == "max_rounds", res.stop_reason)
    check("软着陆仍产出回答", bool(res.answer.strip()))

    # 3.7 ask_user 记录
    reg2 = ToolRegistry()
    reg2.register(make_ask_user_tool())
    llm = MockToolLLM([
        {"content": "信息不足，先问用户", "tool_calls": [tc("ask_user", {"question": "在哪一步报错？"})]},
        "请补充报错位置。同时可以先检查系统时间 [来源: 无]",
    ])
    res = run_tool_loop(llm, msgs, reg2)
    check("★ ask_user 的问题被收集", res.asked_user == ["在哪一步报错？"], str(res.asked_user))


# ══════════════════ 4. 专家两种模式 ══════════════════
def test_expert_modes() -> None:
    print("\n[4] 专家：工具模式 vs 固定模式")

    kb = default_knowledge_base()

    # 4.1 工具模式（LLM 支持 chat_with_tools）
    # MockToolLLM 同时具备 chat() 与 chat_with_tools()，共享同一回复队列：
    # 第 1 条给路由器（纯文本），第 2 条给专家（要工具），第 3 条是最终回答。
    llm = MockToolLLM([
        '{"route":"troubleshoot","confidence":0.95,"reason":"含错误码","need_clarify":false}',
        {"content": "查一下", "tool_calls": [tc("kb_search", {"query": "E1024"})]},
        "E1024 表示许可证校验失败 [来源: 故障手册 §2]",
    ])
    orch = RouterOrchestrator(llm, kb)
    st = orch.handle("启动报 E1024", customer="客户A")
    check("★ 工具模式下 used_tools=True", st.tool_stats.get("used_tools") is True, str(st.tool_stats))
    check("工具调用被记录", st.tool_stats.get("tool_calls") == 1, str(st.tool_stats))
    check("校验通过（有引用）", st.verified is True, st.verify_note)

    # 4.2 固定模式（LLM 只支持 chat）→ 优雅降级
    llm2 = MockLLM([
        '{"route":"install","confidence":0.95,"reason":"问安装","need_clarify":false}',
        "安装需要 64 位系统 [来源: 安装手册 §1]",
    ])
    orch2 = RouterOrchestrator(llm2, kb)
    st2 = orch2.handle("怎么安装", customer="客户B")
    check("★ 不支持工具时自动降级", st2.tool_stats.get("mode") == "fixed", str(st2.tool_stats))
    check("降级后仍能产出回答", bool(st2.answer.strip()))
    check("降级后校验仍通过", st2.verified is True, st2.verify_note)

    # 4.3 显式关闭工具
    llm3 = MockToolLLM([
        '{"route":"usage","confidence":0.95,"reason":"问功能","need_clarify":false}',
        "报表在分析菜单 [来源: 用户手册 §3.2]",
    ])
    orch3 = RouterOrchestrator(llm3, kb, config=RouterConfig(enable_tools=False))
    st3 = orch3.handle("报表在哪", customer="客户C")
    check("★ enable_tools=False 时走固定模式",
          st3.tool_stats.get("mode") == "fixed", str(st3.tool_stats))


# ══════════════════ 5. 检索评估 ══════════════════
def test_eval() -> None:
    print("\n[5] 检索评估")

    kb = KnowledgeBase(DOCS)
    check("知识库规模", len(DOCS) == 24, str(len(DOCS)))
    check("标注集规模", len(EVAL_CASES) == 24, str(len(EVAL_CASES)))

    rep = evaluate(kb, EVAL_CASES, k=3, name="按类")
    check("样本数正确", rep.n == 24, str(rep.n))
    check("召回率在 [0,1]", 0 <= rep.recall <= 1, str(rep.recall))
    check("MRR 在 [0,1]", 0 <= rep.mrr <= 1, str(rep.mrr))
    check("★ 按类检索无跨类污染", rep.off_route_ratio == 0.0, str(rep.off_route_ratio))

    # 指标口径自测：构造一个必然命中的样本
    tiny = KnowledgeBase(DOCS)
    one = evaluate(tiny, [EvalCase("启动报 E1024", {"KB-T01"}, "troubleshoot")], k=3)
    check("已知命中样本 Recall=1", one.recall == 1.0, str(one.recall))
    check("已知命中样本 MRR=1（排第一）", one.mrr == 1.0, str(one.mrr))

    # 构造一个必然不命中的样本
    miss = evaluate(tiny, [EvalCase("完全不相关的查询", {"KB-I01"}, "install")], k=3)
    check("已知未命中样本 Recall=0", miss.recall == 0.0, str(miss.recall))
    check("已知未命中样本 MRR=0", miss.mrr == 0.0, str(miss.mrr))

    # 路由收益对比
    by_route, glob = compare_routing(kb, EVAL_CASES, k=3)
    check("★ 全库检索有跨类污染", glob.off_route_ratio > 0.2, str(glob.off_route_ratio))
    check("★ 按类检索污染率为 0", by_route.off_route_ratio == 0.0)
    check("对比结果可导出", len(by_route.summary()) >= 4)

    # 不同 k 不应报错
    for k in (1, 5):
        r = evaluate(kb, EVAL_CASES, k=k)
        check(f"k={k} 可运行", r.n == 24)


def main() -> int:
    print("=" * 60)
    print("多智能体路由 · 工具层与评估测试")
    print("=" * 60)
    test_registry()
    test_ask_user()
    test_loop()
    test_expert_modes()
    test_eval()

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
