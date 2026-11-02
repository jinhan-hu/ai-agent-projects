"""离线测试用的假 LLM 客户端。

为什么需要它：
  真实 LLM 要 API key、要联网、要钱、输出不确定 —— 没法写稳定的单元测试。
  MockLLM 把「模型会怎么回」变成一段可预期的脚本，于是 loop / sandbox /
  retry / tracing 的全部逻辑都能离线验证。

用法：
    from harness.mock_llm import MockLLM, assistant_tool_call, assistant_text

    script = [
        assistant_tool_call("calculator", {"expression": "1+1"}),  # 第 1 轮：要调工具
        assistant_text("答案是 2"),                                  # 第 2 轮：收敛
    ]
    llm = MockLLM(script)
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from .types import Message, ToolCall

# 脚本元素可以是 Message，也可以是 (messages, tools) -> Message 的函数
ScriptItem = Message | Callable[[list[Message], list[dict[str, Any]]], Message]


def assistant_tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str = "call_1",
    thought: str = "",
) -> Message:
    """构造「模型请求调用工具」的响应。"""
    return Message(
        role="assistant",
        content=thought,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments or {})],
    )


def assistant_tool_calls(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    thought: str = "",
) -> Message:
    """构造「模型一次请求多个工具」的响应（并行调用）。"""
    return Message(
        role="assistant",
        content=thought,
        tool_calls=[
            ToolCall(id=f"call_{i + 1}", name=n, arguments=a)
            for i, (n, a) in enumerate(calls)
        ],
    )


def assistant_text(text: str) -> Message:
    """构造「模型给出最终答案」的响应（无工具调用）。"""
    return Message(role="assistant", content=text)


class MockLLM:
    """按脚本返回固定响应的假 LLM。

    参数
    ----
    script : 响应序列。元素是 Message，或 (messages, tools) -> Message 的函数。
    repeat_last : 脚本用完后是否重复最后一条。False 则抛异常（防止测试意外多调）。
    记录
    ----
    calls : 每次 chat() 的入参快照，便于断言「loop 有没有正确组装上下文」。
    """

    def __init__(self, script: Iterable[ScriptItem], *, repeat_last: bool = False) -> None:
        self.script: list[ScriptItem] = list(script)
        if not self.script:
            raise ValueError("script 不能为空")
        self.repeat_last = repeat_last
        self.i = 0
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        """模拟一次模型调用。"""
        # 记录入参快照（deep copy 一层，避免调用方后续修改影响断言）
        self.calls.append(
            {
                "messages": [m.to_dict() for m in messages],
                "tools": list(tools),
            }
        )

        if self.i >= len(self.script):
            if not self.repeat_last:
                raise RuntimeError(
                    f"脚本已用完：已调用 {self.i} 次，但 loop 还在请求第 {self.i + 1} 次"
                )
            item = self.script[-1]
        else:
            item = self.script[self.i]
            self.i += 1

        if callable(item):
            return item(messages, tools)
        return item

    @property
    def call_count(self) -> int:
        """已被调用次数。"""
        return len(self.calls)
