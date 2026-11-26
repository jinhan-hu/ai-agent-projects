"""MCP（Model Context Protocol）客户端 + 工具适配器。

★ 为什么要有这一层：
    README 的「扩展点」里写着「接 MCP 只改 ToolRegistry 的两个方法」——
    这个文件就是**验证那句话**：把工具来源从「本地函数」换成「MCP server」，
    上层的 tool_loop / experts 一行都不用改。

★ 协议要点（最小闭环，够用且够讲清）：
    transport   —— stdio，**每行一条 JSON-RPC 2.0**（不是 Content-Length 头）
    握手        —— initialize → notifications/initialized
    列工具      —— tools/list  → {"tools":[{name, description, inputSchema}]}
    调工具      —— tools/call  → {"content":[{type:"text", text:"..."}], "isError":bool}

★ 刻意不实现 Resources / Prompts：
    那是「客户端主动读资源」和「提示模板」两类能力，本项目的工具调用只需要 Tools。
    ★ 但要在 README 里说清这个边界——否则「支持 MCP」就成了夸大。

★ 零依赖：只用 subprocess + json，不引入 mcp SDK。
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from .tools import ToolRegistry, ToolSpec

PROTOCOL_VERSION = "2024-11-05"


# ══════════════════════ JSON-RPC ══════════════════════

class MCPError(RuntimeError):
    """MCP 协议层的错误。"""


@dataclass
class _Pending:
    id: int
    result: dict | None = None
    error: dict | None = None


class MCPClient:
    """一个最小的 MCP 客户端（stdio 传输）。

    用法：
        with MCPClient([sys.executable, "mcp_demo/kb_server.py"]) as c:
            tools = c.list_tools()
            out = c.call_tool("kb_search", {"query": "E1024"})
    """

    def __init__(
        self,
        command: list[str],
        *,
        timeout: float = 30.0,
        env: dict | None = None,
    ) -> None:
        self.command = command
        self.timeout = timeout
        self._env = env
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self.server_info: dict = {}
        self.capabilities: dict = {}
        # 统计（测试与演示都用得上）
        self.stats = {"requests": 0, "errors": 0, "tools_called": 0}

    # ---------- 生命周期 ----------
    def start(self) -> "MCPClient":
        if self._proc is not None:
            return self
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,     # ★ stderr 单独收，不污染协议
            text=True,
            encoding="utf-8",
            bufsize=1,                  # 行缓冲
            env=self._env,
        )
        self._handshake()
        return self

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._proc.kill()
        finally:
            self._proc = None

    def __enter__(self) -> "MCPClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- 底层收发 ----------
    def _write(self, obj: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _read(self) -> dict:
        assert self._proc and self._proc.stdout
        line = self._proc.stdout.readline()
        if not line:
            err = ""
            if self._proc.stderr:
                err = (self._proc.stderr.read() or "")[:400]
            raise MCPError(f"MCP server 提前退出（stderr: {err}）")
        return json.loads(line)

    def _request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        rid = self._next_id
        self.stats["requests"] += 1
        self._write(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        )
        # ★ 服务端可能先发通知（notification 没有 id），要跳过
        while True:
            msg = self._read()
            if msg.get("id") != rid:
                continue
            if "error" in msg:
                self.stats["errors"] += 1
                raise MCPError(f"{method} 失败：{msg['error']}")
            return msg.get("result") or {}

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ---------- 协议层 ----------
    def _handshake(self) -> None:
        res = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "router-agent", "version": "0.5.0"},
            },
        )
        self.server_info = res.get("serverInfo") or {}
        self.capabilities = res.get("capabilities") or {}
        self._notify("notifications/initialized")

    # ---------- 能力层 ----------
    def list_tools(self) -> list[dict]:
        res = self._request("tools/list")
        return res.get("tools") or []

    def call_tool(self, name: str, arguments: dict | None = None) -> tuple[bool, str]:
        """返回 (ok, text)。ok=False 表示工具报错或 isError=True。"""
        self.stats["tools_called"] += 1
        res = self._request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        is_error = bool(res.get("isError"))
        parts: list[str] = []
        for block in res.get("content") or []:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(f"[{block.get('type')}]")
        text = "\n".join(p for p in parts if p)
        return (not is_error, text)


# ══════════════════════ 工具适配器 ══════════════════════

def register_mcp_tools(
    registry: ToolRegistry,
    client: MCPClient,
    *,
    prefix: str = "",
    only: list[str] | None = None,
) -> list[str]:
    """把 MCP server 暴露的工具**注册进本地 ToolRegistry**。

    ★ 这就是「接 MCP 只改 ToolRegistry 两个方法」的实证：
      - `to_schema()` 的数据来自 `client.list_tools()`（走 tools/list）
      - `execute()` 的实现走 `client.call_tool()`（走 tools/call）
      - **上层的 tool_loop / experts 完全不知道工具来自本地还是 MCP**

    返回被注册的工具名列表。
    """
    registered: list[str] = []
    for t in client.list_tools():
        name = t.get("name")
        if not name:
            continue
        if only and name not in only:
            continue
        local_name = f"{prefix}{name}"
        schema = t.get("inputSchema") or {"type": "object", "properties": {}}
        # 只保留我们需要的字段（MCP 的 inputSchema 就是 JSON Schema）
        params = {
            "type": schema.get("type", "object"),
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

        def _make_runner(tool_name: str):
            def _run(args: dict) -> str:
                ok, text = client.call_tool(tool_name, args)
                if not ok:
                    raise RuntimeError(text or "MCP 工具返回错误")
                return text
            return _run

        registry.register(
            ToolSpec(
                name=local_name,
                description=t.get("description") or f"(来自 MCP server 的工具 {name})",
                parameters=params,
                func=_make_runner(name),
            )
        )
        registered.append(local_name)
    return registered


def register_mcp_tools_silently(
    registry: ToolRegistry, client: MCPClient, **kw
) -> list[str]:
    """失败时返回空列表而不是抛异常（便于「MCP 可选」的场景）。"""
    try:
        return register_mcp_tools(registry, client, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"[mcp] 注册失败，跳过：{type(exc).__name__}: {exc}", file=sys.stderr)
        return []
