"""会话记忆：滑动窗口 + 摘要压缩。

两层记忆：
  - 短期：最近 N 轮对话原文（保真，但占 token）
  - 长期：更早内容的摘要（省 token，但会失真）

权衡点：什么时候触发压缩？压缩到什么颗粒度？
  本项目取「按消息条数触发、按 LLM 摘要压缩」，够用且可解释。
"""
from __future__ import annotations

from typing import Callable

from .types import Message


def estimate_tokens(text: str) -> int:
    """粗估 token 数：中日韩字符按 1 token，其余按 4 字符 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk + (len(text) - cjk) // 4 + 1


class Memory:
    """会话记忆管理器。

    用法：
        mem = Memory(max_messages=10, summarizer=llm)
        mem.add(Message(role="user", content="..."))
        ctx = mem.get_context()      # 塞进 prompt 的消息列表
    """

    def __init__(
        self,
        *,
        max_messages: int = 20,
        max_tokens: int = 3000,
        summarizer: Callable[[list[Message]], str] | None = None,
        keep_recent: int = 6,
    ) -> None:
        self.messages: list[Message] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarizer = summarizer
        self.keep_recent = keep_recent
        self.summary: str = ""

    # ---------- 写入 ----------
    def add(self, msg: Message) -> None:
        self.messages.append(msg)

    # ---------- 读取 ----------
    def get_context(self) -> list[Message]:
        """返回当前可用上下文（摘要 + 最近若干条）。"""
        ctx: list[Message] = []
        if self.summary:
            ctx.append(Message(role="system", content=f"[历史摘要] {self.summary}"))
        ctx.extend(self.messages[-self.max_messages:])
        return ctx

    def total_tokens(self) -> int:
        return estimate_tokens(self.summary) + sum(
            estimate_tokens(m.content) for m in self.messages
        )

    def should_compress(self) -> bool:
        return (
            len(self.messages) > self.max_messages
            or self.total_tokens() > self.max_tokens
        )

    # ---------- 压缩 ----------
    def compress(self) -> bool:
        """把较早的若干条压成一句话摘要。返回是否真的压了。"""
        if len(self.messages) <= self.keep_recent:
            return False
        old = self.messages[: -self.keep_recent]
        self.messages = self.messages[-self.keep_recent :]

        if self.summarizer is not None:
            try:
                text = self.summarizer(old)
            except Exception:  # noqa: BLE001  摘要失败不能影响主流程
                text = "（摘要生成失败）"
        else:
            # 没有 summarizer 时退化为「截断拼接」
            text = " / ".join(m.content[:40] for m in old if m.content)[:300]

        self.summary = (self.summary + " " + text).strip() if self.summary else text
        return True
