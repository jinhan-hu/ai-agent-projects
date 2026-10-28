"""两个示例工具：计算器 + 读文件。

演示「一个工具应该长什么样」：函数 + 类型注解 + 一句清晰说明。
"""
from __future__ import annotations


def calculator(expression: str) -> str:
    """计算一个数学表达式，例如 '2 * (3 + 4)'。"""
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "错误：表达式含非法字符"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:  # noqa: BLE001
        return f"错误：{e}"


def read_file(path: str, max_chars: int = 2000) -> str:
    """读取文本文件内容（最多 max_chars 个字符）。"""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception as e:  # noqa: BLE001
        return f"错误：{e}"
