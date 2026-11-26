"""测试：MCP 客户端 / 工具适配器 / 协议层。

★ 重点不是"能不能连上"，而是：
    1. 协议三段闭环是否走通（initialize → tools/list → tools/call）
    2. 错误路径：未知工具 / 缺必填参数 / server 提前退出
    3. ★ 适配器的核心主张：**接 MCP 后上层零改动**（行为与本地工具完全一致）
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from router_agent import MockToolLLM  # noqa: E402
from router_agent.mcp import (  # noqa: E402
    MCPClient,
    MCPError,
    register_mcp_tools,
    register_mcp_tools_silently,
)
from router_agent.tool_loop import build_tool_messages, run_tool_loop  # noqa: E402
from router_agent.tools import ToolRegistry, ToolSpec  # noqa: E402

SERVER = os.path.join(_ROOT, "mcp_demo", "kb_server.py")
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


def _client() -> MCPClient:
    return MCPClient([sys.executable, SERVER], timeout=30)


# ══════════════════ 1. 协议闭环 ══════════════════
def test_protocol() -> None:
    print("\n[1] 协议三段闭环")
    with _client() as c:
        check("initialize 拿到 serverInfo", c.server_info.get("name") == "kb-server",
              str(c.server_info))
        check("能力声明只有 tools", set(c.capabilities) == {"tools"}, str(c.capabilities))

        tools = c.list_tools()
        names = [t["name"] for t in tools]
        check("tools/list 返回 2 个工具", len(tools) == 2, str(names))
        check("工具名正确", set(names) == {"kb_search", "get_error_code"}, str(names))
        check("每个工具都有 inputSchema",
              all(t.get("inputSchema", {}).get("type") == "object" for t in tools))
        check("每个工具都有 description",
              all(t.get("description") for t in tools))

        ok, text = c.call_tool("get_error_code", {"code": "E1024"})
        check("tools/call 成功", ok and "许可证" in text, text[:60])

        ok, text = c.call_tool("kb_search", {"query": "许可证", "k": 1})
        check("kb_search 返回带出处", ok and "§" in text, text[:60])


# ══════════════════ 2. 错误路径 ══════════════════
def test_errors() -> None:
    print("\n[2] 错误路径")
    with _client() as c:
        ok, text = c.call_tool("no_such_tool", {})
        check("未知工具 → isError", not ok, f"ok={ok}")
        check("未知工具信息可读", "未知工具" in text, text[:60])

        ok, text = c.call_tool("kb_search", {})
        check("缺必填 → isError", not ok, f"ok={ok}")
        check("缺必填信息含字段名", "query" in text, text[:60])

        ok, text = c.call_tool("get_error_code", {"code": "E9999"})
        check("未知错误码不报错但给可用列表", ok and "不在字典中" in text, text[:60])

    # server 不存在 → 应该抛 MCPError 而不是挂死
    bad = MCPClient([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=10)
    try:
        bad.start()
        check("server 提前退出应抛 MCPError", False, "没抛异常")
    except Exception as exc:  # noqa: BLE001
        check("server 提前退出抛 MCPError", isinstance(exc, MCPError), type(exc).__name__)
    finally:
        bad.close()


# ══════════════════ 3. 适配器 ══════════════════
def test_adapter() -> None:
    print("\n[3] 工具适配器（MCP → ToolRegistry）")
    with _client() as c:
        reg = ToolRegistry()
        names = register_mcp_tools(reg, c)
        check("注册了 2 个工具", len(names) == 2, str(names))

        schema = reg.to_schema()
        check("schema 结构与本地工具一致",
              all({"type", "function"} <= set(s) for s in schema))
        check("参数从 inputSchema 搬过来",
              "query" in schema[0]["function"]["parameters"]["properties"])

        r = reg.execute("kb_search", {"query": "安装", "k": 1})
        check("通过 registry 执行成功", r.ok, r.error)
        check("内容真的是 MCP 返回的", "安装" in r.content or "64 位" in r.content,
              r.content[:60])

        # 前缀
        reg2 = ToolRegistry()
        names2 = register_mcp_tools(reg2, c, prefix="mcp__kb__")
        check("支持加前缀", all(n.startswith("mcp__kb__") for n in names2), str(names2))

        # only 过滤
        reg3 = ToolRegistry()
        names3 = register_mcp_tools(reg3, c, only=["kb_search"])
        check("only 只注册指定工具", names3 == ["kb_search"], str(names3))

    # 静默版
    dead = MCPClient([sys.executable, "-c", "import sys; sys.exit(1)"], timeout=5)
    names4 = register_mcp_tools_silently(ToolRegistry(), dead)
    check("静默版失败返回空列表", names4 == [], str(names4))
    dead.close()


# ══════════════════ 4. ★ 核心主张：上层零改动 ══════════════════
def test_upper_layer_unchanged() -> None:
    print("\n[4] ★ 核心主张：换成 MCP 后，tool_loop 行为完全一致")

    def build_local() -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(ToolSpec(
            name="kb_search",
            description="检索知识库（本地实现）",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]},
            func=lambda a: f"[来源: 安装手册 §1] 安装需要 64 位系统（本地）",
        ))
        reg.register(ToolSpec(
            name="get_error_code",
            description="查错误码（本地实现）",
            parameters={"type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"]},
            func=lambda a: "E1024：许可证校验失败（本地）",
        ))
        return reg

    def script() -> MockToolLLM:
        return MockToolLLM([
            {"content": "查错误码",
             "tool_calls": [{"id": "c1", "name": "get_error_code",
                             "arguments": {"code": "E1024"}}]},
            {"content": "再检索",
             "tool_calls": [{"id": "c2", "name": "kb_search",
                             "arguments": {"query": "E1024"}}]},
            "E1024 表示许可证校验失败 [来源: 故障手册 §2]。",
        ])

    msgs = build_tool_messages("你是专家。", "启动报 E1024")

    res_local = run_tool_loop(script(), msgs, build_local(), max_rounds=5)

    with _client() as c:
        reg_mcp = ToolRegistry()
        register_mcp_tools(reg_mcp, c)
        res_mcp = run_tool_loop(script(), msgs, reg_mcp, max_rounds=5)

    check("轮数一致", res_local.rounds == res_mcp.rounds,
          f"{res_local.rounds} vs {res_mcp.rounds}")
    check("调用序列一致",
          [x["name"] for x in res_local.tool_calls] == [x["name"] for x in res_mcp.tool_calls],
          f"{[x['name'] for x in res_local.tool_calls]} vs {[x['name'] for x in res_mcp.tool_calls]}")
    check("停止原因一致", res_local.stop_reason == res_mcp.stop_reason,
          f"{res_local.stop_reason} vs {res_mcp.stop_reason}")
    check("used_tools 一致", res_local.used_tools == res_mcp.used_tools)
    check("答案一致（模型脚本决定，与工具来源无关）",
          res_local.answer == res_mcp.answer)
    # 取第一条 role=tool 的消息（工具回填），对比两边内容确实不同
    def first_tool_msg(res):
        for m in res.messages:
            if m.get("role") == "tool":
                return m.get("content") or ""
        return ""

    local_tool = first_tool_msg(res_local)
    mcp_tool = first_tool_msg(res_mcp)
    check("★ 本地工具回填含「本地」", "本地" in local_tool, local_tool[:80])
    check("★ MCP 工具回填不含「本地」（确实是另一个实现）",
          "本地" not in mcp_tool, mcp_tool[:80])
    check("★ 两边工具返回内容不同", local_tool != mcp_tool,
          f"{local_tool[:40]} vs {mcp_tool[:40]}")


def main() -> int:
    print("=" * 62)
    print("MCP · 测试")
    print("=" * 62)
    test_protocol()
    test_errors()
    test_adapter()
    test_upper_layer_unchanged()

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
