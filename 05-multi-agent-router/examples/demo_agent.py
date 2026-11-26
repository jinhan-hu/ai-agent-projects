"""演示：专家 Agent 自主决定「调几次工具」（第 1 批的核心能力）。

    python examples/demo_agent.py

★ 为什么需要这个 demo：
    examples/demo.py 走的是**固定检索流水线**（第 0 版能力）——
    专家内部直接调 kb.search()，模型不参与"要不要检索"的决定。
    第 1 批加了 tool calling 之后，旧 demo 展示不到这个能力。

★ 这个 demo 展示 4 种"模型自主性"：
    ① 不调工具，直接回答          （简单问题不该浪费检索）
    ② 调 1 次工具                 （常规问题）
    ③ 调 2 次工具                 （先查错误码，再查知识库）
    ④ 反复调同一个工具 → 被重复检测截断（★ 死循环防护）

★ 全部离线：MockToolLLM 把模型的回复脚本化，不需要 API Key。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_agent import MockToolLLM, RouterOrchestrator, default_knowledge_base  # noqa: E402
from router_agent.state import ROUTE_LABELS  # noqa: E402


def route_json(route: str, conf: float = 0.93) -> str:
    return json.dumps(
        {"route": route, "confidence": conf, "reason": "demo", "need_clarify": False},
        ensure_ascii=False,
    )


def call(name: str, args: dict, cid: str = "") -> dict:
    return {"id": cid or f"call-{name}", "name": name, "arguments": args}


def hr(title: str, sub: str = "") -> None:
    print(f"\n{'═' * 78}")
    print(f"  {title}")
    if sub:
        print(f"  {sub}")
    print(f"{'═' * 78}")


def report(st, note: str = "") -> None:
    """打印一张工单的完整处理结果。"""
    print(f"  路由　：{ROUTE_LABELS.get(st.route, st.route)}（置信度 {st.route_confidence:.2f}）")
    ts = st.tool_stats or {}
    print(f"  模式　：{'工具调用' if ts.get('used_tools') else ('固定检索' if ts.get('mode') == 'fixed' else '工具模式·未调用')}")
    print(f"  轮数　：{ts.get('rounds', 0)}　工具调用：{ts.get('tool_calls', 0)} 次"
          f"　停止原因：{ts.get('stop_reason', '-')}")
    if st.asked_user:
        print(f"  追问　：{st.asked_user}")
    print(f"  校验　：{'✔ 通过' if st.verified else '✘ ' + st.verify_note}")
    print("  回答　：")
    for line in st.answer.splitlines()[:6]:
        print(f"    {line}")
    if note:
        print(f"  ★ {note}")


def case1_no_tool() -> None:
    hr("① 不调工具，直接回答", "简单问题不该浪费一次检索")
    llm = MockToolLLM([
        route_json("usage"),
        # ★ 直接给答案，tool_calls 为空 → 循环一轮就收敛
        "报表在「分析 → 报表中心」[来源: 用户手册 §3.2]。",
    ])
    orch = RouterOrchestrator(llm, default_knowledge_base())
    st = orch.handle("报表在哪个菜单下面？", customer="演示客户")
    report(st, "模型判定「这题我直接会」，0 次工具调用")
    print(f"  模型被调用次数：{len(llm.calls)}（路由 1 次 + 专家 1 轮）")


def case2_one_tool() -> None:
    hr("② 调 1 次工具", "常规问题：先检索，再回答")
    llm = MockToolLLM([
        route_json("troubleshoot"),
        {"content": "先查知识库", "tool_calls": [call("kb_search", {"query": "E1024 许可证"})]},
        "E1024 表示许可证校验失败 [来源: 故障手册 §2]。\n"
        "1. 对齐系统时间。\n2. 核对设备指纹。\n3. 确认许可证未过期。",
    ])
    orch = RouterOrchestrator(llm, default_knowledge_base())
    st = orch.handle("启动报 E1024", customer="演示客户")
    report(st, "模型自主决定「需要检索」，调 1 次 kb_search")


def case3_two_tools() -> None:
    hr("③ 调 2 次工具", "复杂问题：先查错误码字典，再查知识库")
    llm = MockToolLLM([
        route_json("troubleshoot"),
        {"content": "先查错误码含义",
         "tool_calls": [call("get_error_code", {"code": "E2001"}, "c1")]},
        {"content": "再查处理流程",
         "tool_calls": [call("kb_search", {"query": "连接池耗尽 排查"}, "c2")]},
        "E2001 是数据库连接池耗尽 [来源: 错误码字典]。\n"
        "1. 查慢查询日志，定位长事务。\n"
        "2. 确认连接是否泄漏（未 close）。\n"
        "3. 临时调大连接池上限。",
    ])
    orch = RouterOrchestrator(llm, default_knowledge_base())
    st = orch.handle("系统报 E2001，很慢", customer="演示客户")
    report(st, "★ 模型自主调 2 次，且第二次是基于第一次的结果决定要查什么")


def case4_loop_guard() -> None:
    hr("④ 模型卡住 → 重复检测截断", "★ 死循环防护：同一组调用重复超过阈值就停")
    same = {"content": "再查一次同样的东西",
            "tool_calls": [call("kb_search", {"query": "E1024"}, "dup")]}
    llm = MockToolLLM([route_json("troubleshoot")] + [same] * 8)
    orch = RouterOrchestrator(llm, default_knowledge_base())
    st = orch.handle("启动报 E1024", customer="演示客户")
    report(st, "★ 检测到重复调用，主动停止并给出可读说明")
    print("  ★ 注意：校验显示 ✘ 是**预期行为**——")
    print("     回答里确实没有引用，verify 正确拦住了「无依据的回答」。")

    ts = st.tool_stats or {}
    print(f"\n  ★ 关键：stop_reason = {ts.get('stop_reason')!r}")
    print("     → 模型没有自己停，是**系统侧**截断的")
    print("     → 而且答案是「可读的失败说明」，不是「再查一次」这种无意义输出")


def case5_max_rounds() -> None:
    hr("⑤ 轮数用尽 → 软着陆", "★ 每轮换不同 query，规避重复检测，靠轮数上限停")
    # 索引说明：call#0 = 路由；call#1..#4 = 4 轮工具；call#5 = 软着陆（不带工具那次）
    replies = [
        {"content": f"第 {i} 轮检索",
         "tool_calls": [call("kb_search", {"query": f"查询变体 {i}"}, f"r{i}")]}
        for i in range(1, 5)
    ]
    replies.append(
        "（软着陆）4 轮检索已到上限，基于已有信息：安装需要 64 位系统、内存 ≥ 8 GB "
        "[来源: 安装手册 §1]。如需更细的步骤请提供产品版本。"
    )
    llm = MockToolLLM([route_json("install")] + replies)
    # max_rounds 默认 4（专家内部传给 run_tool_loop）
    orch = RouterOrchestrator(llm, default_knowledge_base())
    st = orch.handle("安装需要什么环境", customer="演示客户")
    report(st, "★ 达到轮数上限后「不带工具再问一次」，让模型用已有信息总结")


def main() -> int:
    print("=" * 78)
    print("专家 Agent · 自主决定调几次工具")
    print("=" * 78)
    print("对照：examples/demo.py 是「固定检索流水线」——专家内部直接调 kb.search()，")
    print("      模型不参与「要不要检索」的决定。")
    print("      本 demo 展示第 1 批加入的 tool calling 能力。")

    case1_no_tool()
    case2_one_tool()
    case3_two_tools()
    case4_loop_guard()
    case5_max_rounds()

    print(f"\n{'=' * 78}")
    print("三个可带走的结论：")
    print("  ① 简单问题不该调工具 —— 让模型自己判断「不用查」，省一次调用")
    print("  ② 复杂问题模型会多调 —— 且第二次调用由第一次的结果决定")
    print("  ③ ★ 模型可能卡住 —— 必须有「重复检测」+「轮数上限」两道闸，")
    print("     而且停下时要给出**人类可读**的说明，不是把模型的最后一句原样返回")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
