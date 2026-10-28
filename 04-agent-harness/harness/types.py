"""核心数据结构：Agent 执行过程中流转的所有对象。

设计原则：全部用 dataclass，不可变字段尽量少，方便 tracing 时序列化。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass
class Message:
    """一条对话消息。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolSpec:
    """工具的声明式描述：名字、说明、参数 schema、可调用体。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema 风格
    func: Callable[..., Any] | None = None


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
