"""真实 LLM 客户端（OpenAI 兼容协议，零第三方依赖）。

支持：DeepSeek / 阿里百炼 / OpenAI / 任何兼容 /v1/chat/completions 的服务。

为什么不用 openai SDK：
  本项目宣称「零第三方依赖」，用标准库 urllib 就能完成同样的事，
  而且能看清 HTTP 层的错误处理（这对工程能力是加分项）。

用法：
    client = OpenAICompatibleClient(
        base_url="https://api.deepseek.com",
        api_key="sk-...",
        model="deepseek-chat",
    )
    loop = AgentLoop(client, registry)
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .types import Message, ToolCall


def _message_to_api(m: Message) -> dict[str, Any]:
    """把内部 Message 转成 OpenAI 协议格式。"""
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                },
            }
            for c in m.tool_calls
        ]
    if m.role == "tool" and m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d


class OpenAICompatibleClient:
    """OpenAI 兼容的 Chat Completions 客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.last_usage: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_api(m) for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = self._post("/v1/chat/completions", payload)
        self.last_usage = data.get("usage") or {}

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}

        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"__raw__": raw_args}
            calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        return Message(
            role="assistant",
            content=msg.get("content") or "",
            tool_calls=calls,
        )

    # ------------------------------------------------------------------
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:500]
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e
