"""沙箱执行：隔离工具调用，控制超时、异常与输出长度。

为什么要沙箱：
  1. 工具可能死循环 -> 必须能超时
  2. 工具可能崩溃   -> 不能拖垮主进程
  3. 工具可能输出超长 -> 必须截断，否则撑爆上下文

实现取舍（重要）：
  本模块用「线程 + 软超时」：超时后立即返回 error="timeout"，但线程本身
  无法被强制杀死（Python 限制）。生产环境应改用 subprocess 做硬隔离。
  这里选线程是为了：零依赖、跨平台、工具函数可以是闭包（便于测试）。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from .registry import ToolRegistry
from .types import ToolCall, ToolResult

MAX_OUTPUT_CHARS = 4000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def run_in_sandbox(
    call: ToolCall,
    registry: ToolRegistry,
    *,
    timeout_s: float = 10.0,
    max_output_chars: int = MAX_OUTPUT_CHARS,
) -> ToolResult:
    """在受控环境下执行一次工具调用。

    永不抛异常，所有失败都转成 ToolResult(ok=False)。
    """
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    # ① 工具不存在
    try:
        spec = registry.get(call.name)
    except KeyError:
        return ToolResult(
            call_id=call.id,
            ok=False,
            error=f"tool_not_found: {call.name}",
            duration_ms=elapsed_ms(),
        )

    if spec.func is None:
        return ToolResult(
            call_id=call.id, ok=False, error=f"tool_has_no_func: {call.name}",
            duration_ms=elapsed_ms(),
        )

    # ② 执行（线程 + 软超时）
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        future = ex.submit(spec.func, **call.arguments)
        try:
            raw: Any = future.result(timeout=timeout_s)
        except FuturesTimeout:
            return ToolResult(
                call_id=call.id, ok=False,
                error=f"timeout after {timeout_s}s", duration_ms=elapsed_ms(),
            )
    except TypeError as e:
        # 参数不匹配 -> 属于「模型可改正」的错误
        return ToolResult(
            call_id=call.id, ok=False, error=f"bad_arguments: {e}",
            duration_ms=elapsed_ms(),
        )
    except Exception as e:  # noqa: BLE001  工具内部异常一律兜住
        return ToolResult(
            call_id=call.id, ok=False,
            error=f"{type(e).__name__}: {e}", duration_ms=elapsed_ms(),
        )
    finally:
        # 不等待线程结束：超时场景下等待会挂死
        ex.shutdown(wait=False)

    # ③ 成功：输出转字符串并截断
    return ToolResult(
        call_id=call.id,
        ok=True,
        output=_truncate(str(raw), max_output_chars),
        duration_ms=elapsed_ms(),
    )
