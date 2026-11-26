"""演示：把工具来源从「本地函数」换成「MCP server」——上层一行不改。

    python examples/demo_mcp.py

★ 这个 demo 要证明的那句话（写在 README「扩展点」里的）：

    「接入 MCP 工具 —— 只改 `ToolRegistry.to_schema()` 和 `execute()` 两个方法」

★ 本 demo 的验证方式：
    同一个 `run_tool_loop`、同一个 `ExpertBase`、同一段"模型回复脚本"，
    只把 registry 里的工具换成来自 MCP server 的 —— 看行为是否完全一致。

对照：
    - 本地工具：router_agent/tools.py 里的 make_kb_search_tool()
    - MCP 工具 ：mcp_demo/kb_server.py 通过 stdio 暴露的 kb_search
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from router_agent.mcp import MCPClient, register_mcp_tools  # noqa: E402
from router_agent.tool_loop import build_tool_messages, run_tool_loop  # noqa: E402
from router_agent.tools import (  # noqa: E402
    ToolRegistry,
    make_ask_user_tool,
    make_kb_search_tool,
)
from router_agent import MockToolLLM, default_knowledge_base  # noqa: E402

SERVER = os.path.join(_ROOT, "mcp_demo", "kb_server.py")


def hr(title: str) -> None:
    print(f"\n{'═' * 76}\n  {title}\n{'═' * 76}")


def scripted_llm() -> MockToolLLM:
    """同一段"模型行为脚本"——两个实验都用它。"""
    return MockToolLLM([
        {
            "content": "先查错误码的含义",
            "tool_calls": [
                {"id": "c1", "name": "get_error_code", "arguments": {"code": "E1024"}}
            ],
        },
        {
            "content": "再检索处理步骤",
            "tool_calls": [
                {"id": "c2", "name": "kb_search",
                 "arguments": {"query": "E1024 许可证 校验失败"}}
            ],
        },
        "E1024 表示许可证校验失败 [来源: 故障手册 §2]。\n"
        "1. 对齐系统时间。\n2. 核对设备指纹。\n3. 确认许可证未过期。",
    ])


def run_with_registry(reg: ToolRegistry, label: str) -> dict:
    llm = scripted_llm()
    msgs = build_tool_messages("你是技术支持专家。", "启动报 E1024，怎么处理？")
    res = run_tool_loop(llm, msgs, reg, max_rounds=5)
    print(f"  [{label}] 轮数={res.rounds}　工具调用={len(res.tool_calls)} 次"
          f"　停止原因={res.stop_reason}　used_tools={res.used_tools}")
    print(f"  [{label}] 答案首行：{res.answer.splitlines()[0][:60]}")
    return {
        "rounds": res.rounds,
        "calls": [(c["name"], c["ok"]) for c in res.tool_calls],
        "stop": res.stop_reason,
        "used_tools": res.used_tools,
    }


def main() -> int:
    print("=" * 76)
    print("MCP 接入演示：工具来源换成 MCP server，上层不改")
    print("=" * 76)

    # ---------- ① 本地工具 ----------
    hr("① 本地工具（make_kb_search_tool / get_error_code 直接是 Python 函数）")
    local = ToolRegistry()
    local.register(make_kb_search_tool(default_knowledge_base(), default_route="troubleshoot"))
    local.register(make_ask_user_tool())
    # 本地没有 get_error_code？补一个最小实现，保证两边工具集一致
    from router_agent.tools import ToolSpec
    local.register(ToolSpec(
        name="get_error_code",
        description="查询错误码含义（本地实现）",
        parameters={"type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"]},
        func=lambda a: ("E1024：许可证校验失败（本地字典）"
                        if a.get("code", "").upper() == "E1024"
                        else "不在本地字典中"),
    ))
    print(f"  工具：{local.names()}")
    r_local = run_with_registry(local, "本地")

    # ---------- ② MCP 工具 ----------
    hr("② MCP 工具（同样两个名字，实现跑在独立进程里）")
    with MCPClient([sys.executable, SERVER]) as client:
        print(f"  MCP server：{client.server_info}")
        caps = client.capabilities
        print(f"  能力声明：{caps}　（★ 只有 tools —— 本项目不实现 Resources/Prompts）")

        remote = ToolRegistry()
        names = register_mcp_tools(remote, client)
        print(f"  注册进 ToolRegistry：{names}")
        r_mcp = run_with_registry(remote, "MCP ")

        hr("③ 对比：两者的循环行为应完全一致")
        print(f"  本地：{r_local}")
        print(f"  MCP ：{r_mcp}")

        same = (
            r_local["rounds"] == r_mcp["rounds"]
            and r_local["calls"] == r_mcp["calls"]
            and r_local["stop"] == r_mcp["stop"]
            and r_local["used_tools"] == r_mcp["used_tools"]
        )
        print(f"\n  → {'★ 完全一致 —— 上层没有任何改动' if same else '✗ 不一致（需要排查）'}")
        print(f"\n  MCP 客户端统计：{client.stats}")

    # ---------- ④ 协议层可以直接调 ----------
    hr("④ 不只经过 registry —— 也能直接用 MCP 协议调")
    with MCPClient([sys.executable, SERVER]) as client:
        print("  原始 tools/call（绕过 ToolRegistry）：")
        ok, text = client.call_tool("get_error_code", {"code": "E2001"})
        print(f"    ok={ok}  {text}")

        print("\n  缺必填参数时服务端返回 isError：")
        ok, text = client.call_tool("kb_search", {})
        print(f"    ok={ok}  {text}")

    print(f"\n{'=' * 76}")
    print("三个可带走的结论：")
    print("  ① MCP 的最小闭环只有三段：initialize → tools/list → tools/call")
    print("  ② ★ 接 MCP 只需改 ToolRegistry 的「schema 从哪来」和「execute 走哪去」，")
    print("     上层的 tool_loop / experts 一行都不用改")
    print("  ③ 但要诚实：本项目只实现 Tools 原语，")
    print("     Resources / Prompts 没有消费接口 —— 别把它说成「完整支持 MCP」")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
