"""Agent Loop 主循环：思考 → 调用 → 观察 → 收敛。

这是整个框架的心脏。其余模块都是它的「零件」。
"""
from __future__ import annotations

from typing import Any, Protocol

from .registry import ToolRegistry
from .types import Message, Trace


class LLMClient(Protocol):
    """LLM 客户端接口 —— 抽象出来，方便接任意模型。"""

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        """给定对话历史和工具 schema，返回下一条 assistant 消息。"""
        ...


class AgentLoop:
    """Agent 主循环。

    循环终止条件（三选一）：
      1. 模型不再请求工具调用，直接给出最终回答
      2. 达到最大步数 max_steps
      3. 检测到循环（相同的调用重复出现）

    TODO(11-03): 实现 run()：
      - 组装 messages（system + user + 历史）
      - 调 llm.chat() 拿到 assistant 消息
      - 若有 tool_calls → 交 sandbox 执行 → 结果塞回 messages → 继续
      - 若无 tool_calls → 收敛，返回最终答案
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        max_steps: int = 10,
        system_prompt: str = "你是一个能调用工具的助手。",
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def run(self, task: str) -> Trace:
        """执行一个任务，返回完整 trace。"""
        trace = Trace(task=task)
        # TODO(11-03): 实现主循环
        raise NotImplementedError("Agent Loop 主循环待实现（阅读周 11-03）")
