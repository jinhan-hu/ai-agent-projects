"""LLM 接口与两个实现。

为什么要有 MockLLM：
    路由阈值、兜底逻辑、记忆拼接这些**编排逻辑**必须能离线测。
    如果测试依赖真实 API，就会变成「网络抖一下测试就红」。
    Mock 让编排的每一条分支都能被确定性覆盖。
"""
from __future__ import annotations

import json
import re
from typing import Protocol


class LLM(Protocol):
    """最小接口：给消息列表，返回文本。"""

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str: ...


class MockLLM:
    """按顺序返回预设回复；用完后重复最后一条。

    用法：
        llm = MockLLM(["安装类的回答", "确认类的回答"])
        llm.calls   # 调用记录，用于断言「LLM 被调了几次、带了什么」
    """

    def __init__(self, replies: list[str]) -> None:
        if not replies:
            raise ValueError("MockLLM 至少需要一条回复")
        self._replies = replies
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        self.calls.append({"messages": messages, "temperature": temperature})
        idx = min(len(self.calls) - 1, len(self._replies) - 1)
        return self._replies[idx]


class OpenAICompatibleClient:
    """真实 LLM：任何 OpenAI 兼容端点（DeepSeek / OpenAI / 通义 / vLLM …）。

    刻意用标准库 urllib 实现，不加额外依赖——
    这份代码放进任何环境都能跑，不会因为装不上 SDK 而演示失败。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        import urllib.request

        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": temperature},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    # ---------- 工具调用（可选能力） ----------
    def chat_with_tools(
        self, messages: list[dict], tools: list[dict], temperature: float = 0.0
    ) -> dict:
        """带工具调用的一次请求。

        特意做成「可选能力」而不是改 `chat()` 的返回值类型：
        这样纯文本调用方的代码不需要改，工具调用方按需使用。
        """
        import urllib.request

        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        return {
            "content": msg.get("content") or "",
            "tool_calls": _normalize_tool_calls(msg.get("tool_calls")),
        }


def extract_json(text: str) -> dict:
    """从模型输出里抠出 JSON 对象。

    模型经常在 JSON 外面包一层 ```json fence 或解释性文字，
    所以不能直接 json.loads。先找最外层花括号，再解析。
    """
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退而求其次：找最外层 { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法从输出中解析 JSON：{text[:200]!r}")


# ══════════════════════ 工具调用能力 ══════════════════════

def _normalize_tool_calls(raw: list) -> list[dict]:
    """把 OpenAI 风格的 tool_calls 归一成内部格式。

    内部格式：{"id": str, "name": str, "arguments": dict}
    """
    out: list[dict] = []
    for i, tc in enumerate(raw or []):
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                # 模型给了非法 JSON：把原文塞进一个特殊字段，让模型自己看到
                args = {"__raw_arguments__": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        out.append(
            {
                "id": tc.get("id") or f"call-{i}",
                "name": fn.get("name", ""),
                "arguments": args,
            }
        )
    return out


class MockToolLLM:
    """脚本化的工具调用 LLM，用于离线测试。

    每条 reply 可以是：
      - 字符串          → 视为「不调工具，直接回答」
      - dict            → {"content": str, "tool_calls": [{"name","arguments","id"?}]}

    用法：
        llm = MockToolLLM([
            {"content": "调一下检索", "tool_calls": [{"name": "kb_search", "arguments": {"query": "E1024"}}]},
            "根据检索结果：……",
        ])
    """

    def __init__(self, replies: list) -> None:
        if not replies:
            raise ValueError("MockToolLLM 至少需要一条回复")
        self._replies = replies
        self.calls: list[dict] = []

    def chat_tools_seen(self) -> list[int]:
        """每一轮看到的工具数量，用于断言「工具确实传进去了」。"""
        return [len(c["tools"]) for c in self.calls]

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        """纯文本模式（路由器用）。

        ★ 有意让它和 chat_with_tools 共享同一个回复队列：
        这样一个 MockToolLLM 能同时服务「路由（纯文本）」和「专家（工具）」两类调用，
        端到端测试只需要准备一份脚本。
        """
        return self.chat_with_tools(messages, [], temperature).get("content", "")

    def chat_with_tools(
        self, messages: list[dict], tools: list[dict], temperature: float = 0.0
    ) -> dict:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": list(tools),
                "temperature": temperature,
            }
        )
        idx = min(len(self.calls) - 1, len(self._replies) - 1)
        reply = self._replies[idx]
        if isinstance(reply, str):
            return {"content": reply, "tool_calls": []}
        if not isinstance(reply, dict):
            raise TypeError(f"MockToolLLM 收到不支持的回复类型：{type(reply)}")
        return {
            "content": reply.get("content", ""),
            "tool_calls": [dict(c) for c in (reply.get("tool_calls") or [])],
        }
