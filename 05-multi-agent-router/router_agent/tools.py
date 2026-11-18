"""工具层：注册、schema 生成、执行。

设计要点：
1. 工具用**显式 Schema 声明**，而不是从函数签名反射。
   理由：参数约束（枚举、范围、默认值）必须写进 JSON Schema，
   靠反射拿不到这些语义；而且显式声明一眼能看出「模型看到的是什么」。
2. **执行永不抛异常**，所有失败转成 ToolResult(ok=False)。
   工具失败是正常业务分支，不是程序错误——抛异常会中断整个循环。
3. 错误信息必须**模型可读**：写成"缺哪个参数"而不是 "failed"。

为什么 ask_user 是一个工具而不是 prompt 里的一句话：
    prompt 里写「信息不足就提问」，模型经常忘了；
    做成工具后，「要不要问」变成模型**主动选择的一个动作**，可靠得多。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# 工具执行函数：收一个参数字典，返回字符串内容
ToolFn = Callable[[dict], str]


@dataclass
class ToolSpec:
    """一个工具的完整定义。"""

    name: str
    description: str
    parameters: dict[str, Any]        # JSON Schema（type/properties/required）
    func: ToolFn
    # 是否需要用户交互（ask_user 用）
    needs_user: bool = False

    def to_openai_schema(self) -> dict:
        """转成 OpenAI / MCP 通用的 tools 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error: str = ""
    # ask_user 专用：模型请求用户补充信息
    needs_user: bool = False

    def to_message(self, call_id: str) -> dict:
        """转成回填进对话的 tool 消息。"""
        if self.ok:
            return {"role": "tool", "tool_call_id": call_id, "content": self.content}
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"ERROR: {self.error}",
        }


class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册：{spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def to_schema(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    # ---------- 执行 ----------
    def execute(self, name: str, arguments: dict) -> ToolResult:
        """执行一个工具。永不抛异常。"""
        spec = self._tools.get(name)
        if spec is None:
            # ★ 模型可读的错误：告诉它有哪些工具可用
            available = ", ".join(self.names()) or "（无）"
            return ToolResult(ok=False, error=f"tool_not_found: {name}. 可用工具：{available}")

        # 必填参数校验（在调用前拦，错误信息更精确）
        required = spec.parameters.get("required", [])
        missing = [k for k in required if k not in arguments]
        if missing:
            return ToolResult(
                ok=False,
                error=f"bad_arguments: {name} 缺少必填参数 {missing}",
            )

        try:
            content = spec.func(arguments)
        except TypeError as exc:
            return ToolResult(ok=False, error=f"bad_arguments: {name} 参数不匹配（{exc}）")
        except Exception as exc:  # noqa: BLE001  工具内部异常一律兜住
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        return ToolResult(ok=True, content=str(content), needs_user=spec.needs_user)


# ══════════════════════ 内置工具工厂 ══════════════════════

def make_kb_search_tool(kb, default_route: str | None = None) -> ToolSpec:
    """知识库检索工具。

    ★ description 里写清了三件事：什么时候用、什么时候不要用、参数含义。
    这是「工具设计」的核心——模型只能靠这段文字理解工具。
    """

    def _run(args: dict) -> str:
        query = str(args.get("query", "")).strip()
        k = int(args.get("k", 3))
        k = max(1, min(k, 10))
        # 默认先在本类里检索（路由先于检索）
        hits = kb.search(query, k=k, route=default_route) if default_route else []
        scope = "本类"
        if not hits:
            hits = kb.search(query, k=k, route=None)
            scope = "全库"
        if not hits:
            return "（未检索到相关资料。可以换一个更具体的 query，或直接说明资料不足。）"
        blocks = [f"[检索范围：{scope}]"]
        for doc, score in hits:
            blocks.append(
                f"{doc.source or doc.doc_id} | {doc.title}（相关度 {score}）\n{doc.text}"
            )
        return "\n\n".join(blocks)

    return ToolSpec(
        name="kb_search",
        description=(
            "检索产品知识库，返回与 query 相关的文档片段（带出处）。\n"
            "① 在回答任何涉及产品功能、参数、报错、步骤的问题之前，必须先调用本工具；\n"
            "② 不要用它查询客户个人信息或订单数据——那些不在知识库里；\n"
            "③ query 用中文自然语言描述问题，不要传字段名或代码。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题，中文自然语言"},
                "k": {
                    "type": "integer",
                    "description": "返回条数，默认 3，范围 1-10",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        func=_run,
    )


def make_ask_user_tool() -> ToolSpec:
    """向用户提问的工具。

    ★ 这是「模型自主性」的一个体现点：
    把「该不该问」从硬编码逻辑变成模型可选择的一个动作。
    """

    def _run(args: dict) -> str:
        question = str(args.get("question", "")).strip()
        # 无人工交互时（测试 / 演示），返回提示让模型自行继续
        return (
            f"已向用户提问：{question}\n"
            "（当前为无人值守模式，用户未即时响应。"
            "请基于已知信息给出最可能的建议，并说明还缺哪些信息。）"
        )

    return ToolSpec(
        name="ask_user",
        description=(
            "当信息不足以判断、或需要用户确认关键前提时，用它向用户提问。\n"
            "① 只在真正缺少关键信息时使用，不要用它代替思考；\n"
            "② question 要具体，最好一次问清 1-2 个要点，不要笼统地问「请详细说明」；\n"
            "③ 提问后仍需基于已有信息给出可行的初步建议。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "要问用户的具体问题（1-2 个要点）",
                }
            },
            "required": ["question"],
        },
        func=_run,
        needs_user=True,
    )
