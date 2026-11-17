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
