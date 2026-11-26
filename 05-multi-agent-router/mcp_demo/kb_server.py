"""一个最小的 MCP Server（stdio）：把知识库和错误码字典暴露成 MCP 工具。

    python mcp_demo/kb_server.py

★ 这个 server 只实现 Tools 原语（tools/list + tools/call），
  不实现 Resources / Prompts —— 因为本项目只需要工具调用。
  ★ 但这个边界要在 README 里说清，不能含糊成"支持 MCP"。

★ 协议细节：
    - 传输：stdin/stdout，**每行一条 JSON-RPC 2.0**
    - ★ 日志必须走 stderr：往 stdout 打任何非协议内容都会让客户端解析失败
"""
from __future__ import annotations

import json
import os
import sys

# 让 server 能 import 到 router_agent（单独进程启动时路径不同）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from router_agent.knowledge import Doc  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"

# ══════════════════ 数据（与主项目的知识库同构，但独立进程内自持） ══════════════════

_DOCS = [
    Doc("KB-T01", "错误码 E1024",
        "E1024 表示许可证校验失败。常见原因：系统时间不准、激活码与设备指纹不匹配、许可证过期。",
        "troubleshoot", "故障手册 §2"),
    Doc("KB-T02", "错误码 E2001",
        "E2001 表示数据库连接池耗尽。检查慢查询与连接泄漏。",
        "troubleshoot", "故障手册 §5"),
    Doc("KB-I01", "安装前置条件",
        "需要 64 位系统；内存至少 8 GB；磁盘剩余 20 GB；Windows 需 VC++ 运行库。",
        "install", "安装手册 §1"),
    Doc("KB-U01", "报表模块入口",
        "报表在左侧导航「分析 → 报表中心」。选择日期区间后点右上角「生成」。",
        "usage", "用户手册 §3.2"),
]

_ERROR_CODES = {
    "E1024": "许可证校验失败：系统时间 / 设备指纹 / 许可证过期",
    "E2001": "数据库连接池耗尽：查慢查询与连接泄漏",
    "E2002": "磁盘写入失败：检查剩余空间与挂载权限",
}


def _kb_search(args: dict) -> str:
    q = str(args.get("query", "")).strip()
    k = max(1, min(int(args.get("k", 3)), 10))
    if not q:
        return "（空查询）"
    # 极简打分：字符 bigram 命中数（server 端只求"能返回点东西"，不追求检索质量）
    def score(doc: Doc) -> int:
        text = doc.title + doc.text
        return sum(1 for i in range(len(q) - 1) if q[i:i+2] in text)

    ranked = sorted(_DOCS, key=score, reverse=True)
    hits = [d for d in ranked[:k] if score(d) > 0]
    if not hits:
        return "（未检索到相关资料）"
    return "\n\n".join(
        f"{d.source} | {d.title}\n{d.text}" for d in hits
    )


def _get_error_code(args: dict) -> str:
    code = str(args.get("code", "")).strip().upper()
    desc = _ERROR_CODES.get(code)
    if not desc:
        return f"错误码 {code} 不在字典中。可用：{', '.join(sorted(_ERROR_CODES))}"
    return f"{code}：{desc}"


TOOLS = {
    "kb_search": {
        "description": (
            "检索产品知识库，返回与 query 相关的文档片段（带出处）。\n"
            "① 回答产品功能/参数/报错前必须先调用；\n"
            "② 不要用它查客户个人信息；\n"
            "③ query 用中文自然语言。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题"},
                "k": {"type": "integer", "description": "返回条数，1-10", "default": 3},
            },
            "required": ["query"],
        },
        "fn": _kb_search,
    },
    "get_error_code": {
        "description": (
            "查询产品错误码的含义与常见原因。\n"
            "① 用户报 E 开头四位码时先用它查清；\n"
            "② 不查 HTTP 状态码；\n"
            "③ code 传大写。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "错误码，如 E1024"}},
            "required": ["code"],
        },
        "fn": _get_error_code,
    },
}


# ══════════════════ JSON-RPC 处理 ══════════════════

def _ok(rid, result):
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def handle(msg: dict) -> dict | None:
    """处理一条 JSON-RPC 消息。返回响应；通知（无 id）返回 None。"""
    method = msg.get("method")
    rid = msg.get("id")

    if method == "initialize":
        return _ok(rid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "kb-server", "version": "0.1.0"},
        })

    if method == "notifications/initialized":
        return None                      # 通知不响应

    if method == "tools/list":
        return _ok(rid, {
            "tools": [
                {
                    "name": name,
                    "description": spec["description"],
                    "inputSchema": spec["inputSchema"],
                }
                for name, spec in TOOLS.items()
            ]
        })

    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        spec = TOOLS.get(name)
        if spec is None:
            return _ok(rid, {
                "content": [{"type": "text", "text": f"未知工具：{name}"}],
                "isError": True,
            })
        # 必填校验（错误信息要让模型能读懂）
        required = spec["inputSchema"].get("required", [])
        missing = [k for k in required if k not in args]
        if missing:
            return _ok(rid, {
                "content": [{"type": "text",
                             "text": f"bad_arguments: {name} 缺少必填参数 {missing}"}],
                "isError": True,
            })
        try:
            text = spec["fn"](args)
        except Exception as exc:  # noqa: BLE001
            return _ok(rid, {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            })
        return _ok(rid, {"content": [{"type": "text", "text": text}], "isError": False})

    if rid is None:
        return None
    return _err(rid, -32601, f"Method not found: {method}")


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[kb-server] 非法 JSON: {exc}", file=sys.stderr)
            continue
        try:
            resp = handle(msg)
        except Exception as exc:  # noqa: BLE001
            print(f"[kb-server] 处理异常: {exc}", file=sys.stderr)
            resp = _err(msg.get("id"), -32603, str(exc))
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
