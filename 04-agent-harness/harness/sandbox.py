"""沙箱执行：隔离工具调用，控制超时、异常与输出长度。

为什么要沙箱：
  1. 工具可能死循环 → 必须能超时
  2. 工具可能崩溃 → 不能拖垮主进程
  3. 工具可能输出超长 → 必须截断，否则撑爆上下文
"""
from __future__ import annotations

import time
from typing import Any

from .types import ToolCall, ToolResult

MAX_OUTPUT_CHARS = 4000


def run_in_sandbox(
    call: ToolCall,
    registry: Any,
    *,
    timeout_s: float = 10.0,
) -> ToolResult:
    """在受控环境下执行一次工具调用。

    TODO(11-04): 实现：
      - 用 subprocess 或 multiprocessing 隔离执行
      - 超时则 kill 并返回 ok=False, error="timeout"
      - 捕获异常转成 error 字符串
      - 输出超过 MAX_OUTPUT_CHARS 则截断并加标记
      - 记录 duration_ms
    """
    started = time.perf_counter()
    _ = (call, registry, timeout_s, started)
    raise NotImplementedError("沙箱执行待实现（阅读周 11-04）")
