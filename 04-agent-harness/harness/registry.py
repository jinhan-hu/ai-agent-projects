"""工具注册中心：统一 schema、装饰器注册、按名查找。

这是全框架唯一个「先实现」的模块 —— 因为其它六个模块都要依赖它。
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

from .types import ToolSpec


class ToolRegistry:
    """工具注册表。

    用法：
        reg = ToolRegistry()

        @reg.tool(description="两数相加")
        def add(a: int, b: int) -> int:
            return a + b

        reg.get("add").func(1, 2)   # 3
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        description: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """装饰器：把一个普通函数注册成工具。"""

        def deco(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            if tool_name in self._tools:
                raise ValueError(f"工具名重复: {tool_name}")
            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=description or (inspect.getdoc(func) or ""),
                parameters=self._schema_from_func(func),
                func=func,
            )
            return func

        return deco

    @staticmethod
    def _schema_from_func(func: Callable[..., Any]) -> dict[str, Any]:
        """从函数签名+类型注解推导 JSON Schema（简单版，够用即可）。

        注意：若模块用了 `from __future__ import annotations`，
        inspect 拿到的是字符串注解（如 "int"），必须按名字映射，
        否则 int 会被误判成 string。
        """
        py2json = {int: "integer", float: "number", str: "string", bool: "boolean"}
        str2json = {
            "int": "integer",
            "float": "number",
            "str": "string",
            "bool": "boolean",
        }
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, p in inspect.signature(func).parameters.items():
            ann = p.annotation
            if isinstance(ann, str):
                jtype = str2json.get(ann.strip("'\" "), "string")
            else:
                jtype = py2json.get(ann, "string")
            props[pname] = {"type": jtype}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": props, "required": required}

    def get(self, name: str) -> ToolSpec:
        """按名字取工具；不存在则抛 KeyError。"""
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        return self._tools[name]

    def list_tools(self) -> list[ToolSpec]:
        """返回全部工具（供塞进 prompt 用）。"""
        return list(self._tools.values())

    def to_openai_schema(self) -> list[dict[str, Any]]:
        """导出成 OpenAI function-calling 格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)
