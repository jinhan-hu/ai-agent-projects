"""演示：四种典型工单各走一遍。

默认离线运行（MockLLM），不需要 API Key：
    python examples/demo.py

用真实模型跑（读环境变量）：
    set DEEPSEEK_API_KEY=sk-...
    python examples/demo.py --real
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_agent import (  # noqa: E402
    MockLLM,
    OpenAICompatibleClient,
    RouterOrchestrator,
    default_knowledge_base,
)
from router_agent.state import ROUTE_LABELS  # noqa: E402


def route_json(route: str, conf: float = 0.92, clarify: bool = False) -> str:
    return json.dumps(
        {"route": route, "confidence": conf, "reason": "demo", "need_clarify": clarify},
        ensure_ascii=False,
    )


# 离线脚本：每张工单两轮调用（路由 + 专家回答）
OFFLINE_CASES = [
    (
        "安装时提示 E1024，一直进不去",
        route_json("troubleshoot"),
        "E1024 表示许可证校验失败 [来源: 故障手册 §2]。\n"
        "1. 检查系统时间是否准确。\n"
        "2. 到授权页确认设备指纹有没有变化。\n"
        "3. 若近期换过硬件，需要重新申请许可证。",
    ),
    (
        "报表模块在哪个菜单下面？",
        route_json("usage"),
        "报表在「分析 → 报表中心」[来源: 用户手册 §3.2]。\n"
        "1. 选择日期区间。\n"
        "2. 点右上角「生成」。",
    ),
    (
        "新的机器怎么安装？要什么环境",
        route_json("install"),
        "安装前置条件：64 位系统、内存 ≥ 8 GB、磁盘剩余 ≥ 20 GB [来源: 安装手册 §1]。\n"
        "1. 先安装 VC++ 运行库。\n"
        "2. 确认 8080 端口空闲 [来源: 安装手册 §2]。\n"
        "3. 在「设置 → 授权」输入 32 位激活码 [来源: 安装手册 §4]。",
    ),
    (
        "你好，有个问题想问一下",
        route_json("install", conf=0.35, clarify=True),
        "需要先确认几个信息：\n"
        "1. 具体是什么问题（安装/使用/报错）？\n"
        "2. 在哪一步出现？\n"
        "3. 有没有报错信息或错误码？",
    ),
]


def run_offline() -> None:
    kb = default_knowledge_base()
    replies: list[str] = []
    for _, r, a in OFFLINE_CASES:
        replies.extend([r, a])

    llm = MockLLM(replies)
    orch = RouterOrchestrator(llm, kb)

    print("=" * 74)
    print("多智能体路由 · 离线演示（MockLLM）")
    print("=" * 74)

    for text, _, _ in OFFLINE_CASES:
        st = orch.handle(text, customer="演示客户", session_id="demo-session")
        print(f"\n【工单】{text}")
        print(f"  路由　：{ROUTE_LABELS.get(st.route, st.route)}"
              f"（置信度 {st.route_confidence:.2f}）")
        if st.route_reason:
            print(f"  依据　：{st.route_reason}")
        if st.need_clarify:
            print("  ★ 判定为信息不足，已要求用户澄清")
        print(f"  回答　：{st.answer.splitlines()[0]}")
        if st.citations:
            print(f"  引用　：{', '.join(st.citations)}")
        print(f"  校验　：{'✓ 通过' if st.verified else '✗ ' + st.verify_note}")
        print(f"  轨迹　：{' → '.join(t.node for t in st.trace)}")

    print("\n" + "-" * 74)
    print("客户档案（长期记忆）：")
    print(orch.memory.profile_context("演示客户") or "（空）")
    print("-" * 74)


def run_real() -> None:
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("未找到 DEEPSEEK_API_KEY / OPENAI_API_KEY，无法运行真实模式。")
        print("改用离线模式。\n")
        return run_offline()

    base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    llm = OpenAICompatibleClient(base_url=base, api_key=key, model=model)
    orch = RouterOrchestrator(llm, default_knowledge_base())

    print("=" * 74)
    print(f"多智能体路由 · 真实模型演示（{model} @ {base}）")
    print("=" * 74)

    for text, _, _ in OFFLINE_CASES:
        st = orch.handle(text, customer="演示客户", session_id="real-session")
        print(f"\n【工单】{text}")
        print(f"  路由　：{ROUTE_LABELS.get(st.route, st.route)}"
              f"（置信度 {st.route_confidence:.2f}）")
        print(f"  依据　：{st.route_reason}")
        print(f"  回答　：\n" + "\n".join("    " + ln for ln in st.answer.splitlines()))
        print(f"  引用　：{', '.join(st.citations) or '（无）'}")
        print(f"  校验　：{'✓ 通过' if st.verified else '✗ ' + st.verify_note}")


if __name__ == "__main__":
    if "--real" in sys.argv:
        run_real()
    else:
        run_offline()
