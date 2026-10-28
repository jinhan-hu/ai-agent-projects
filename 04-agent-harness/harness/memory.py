"""会话记忆：滑动窗口 + 摘要压缩。

两层记忆：
  - 短期：最近 N 轮对话原文（保真，但占 token）
  - 长期：更早内容的摘要（省 token，但会失真）

权衡点：什么时候触发压缩？压缩到什么颗粒度？
"""
from __future__ import annotations

from .types import Message


class Memory:
    """会话记忆管理器。

    TODO(11-05): 实现：
      - add(msg)：追加消息
      - get_context()：返回可塞进 prompt 的消息列表
      - 当总 token 超过阈值时，把最早的若干轮压成一条摘要
      - 摘要由 LLM 生成（需传入 llm client）
    """

    def __init__(self, *, max_messages: int = 20, max_tokens: int = 3000) -> None:
        self.messages: list[Message] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens

    def add(self, msg: Message) -> None:
        """追加一条消息。"""
        self.messages.append(msg)

    def get_context(self) -> list[Message]:
        """返回当前可用的上下文。"""
        return list(self.messages[-self.max_messages :])

    def should_compress(self) -> bool:
        """是否需要触发压缩。"""
        return len(self.messages) > self.max_messages
