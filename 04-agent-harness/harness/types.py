"""核心数据结构：Agent 执行过程中流转的所有对象。

设计原则：
  1. 全部用 dataclass —— 便于序列化、便于 tracing 时 dump 成 JSON
  2. 消息要能承载「工具调用」和「工具结果」两类特殊内容
  3. 不做校验 —— 校验放在 registry / sandbox 里，数据类保持纯粹
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass
class ToolCall:
    """模型发起的一次工具调用请求。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """工具执行结果。失败时 ok=False 且 error 非空。"""

    call_id: str
    ok: bool
    output: str = ""
    error: str | None = None
    duration_ms: int = 0

    def to_message(self) -> "Message":
        """把结果包装成一条 role='tool' 的消息，供塞回 messages。"""
        return Message(
            role="tool",
            content=self.output if self.ok else f"ERROR: {self.error}",
            name=self.call_id,
            tool_call_id=self.call_id,
        )


@dataclass
class Message:
    """一条对话消息。

    role 的四种取值：
      - system    : 系统提示
      - user      : 用户输入
      - assistant : 模型输出（可能带 tool_calls）
      - tool      : 工具执行结果（必须带 tool_call_id）
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    # ★ 模型请求调用工具时填这里；工具结果消息不填
    tool_calls: list[ToolCall] = field(default_factory=list)
    # ★ role="tool" 时，指向它回应的是哪一次 call
    tool_call_id: str | None = None

    @property
    def wants_tools(self) -> bool:
        """模型是否在请求调用工具。"""
        return self.role == "assistant" and bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class ToolSpec:
    """工具的声明式描述：名字、说明、参数 schema、可调用体。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema 风格
    func: Callable[..., Any] | None = None


@dataclass
class Step:
    """Agent Loop 的一步：思考 → 调用 → 观察。"""

    index: int
    thought: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)


@dataclass
class Trace:
    """一次完整执行的追踪记录，可序列化成 JSON 回放。"""

    task: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    total_tokens: int = 0
    stop_reason: str = ""   # converged / max_steps / loop_detected / error
