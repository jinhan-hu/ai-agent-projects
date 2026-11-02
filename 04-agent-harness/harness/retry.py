"""错误分类、重试与降级兜底。

Agent 的失败不是「成功/失败」二元，要分类型：
  - 可重试（网络抖动、限流）  -> 指数退避重试
  - 可改正（参数错误、工具不存在）-> 直接把错误反馈给模型，让它改
  - 致命（鉴权失败、上下文超长）-> 立刻终止
"""
from __future__ import annotations

import enum
import time
from typing import Any, Callable


class ErrorKind(enum.Enum):
    """错误类型。"""

    RETRYABLE = "retryable"   # 可重试：网络、限流、超时
    FIXABLE = "fixable"       # 可改正：参数不对，让模型改
    FATAL = "fatal"           # 致命：终止


_RETRYABLE = (TimeoutError, ConnectionError, OSError)


def classify_error(exc: BaseException) -> ErrorKind:
    """把异常归类。

    规则：
      TimeoutError / ConnectionError / OSError -> RETRYABLE
      ValueError / TypeError / KeyError        -> FIXABLE（多半是参数问题）
      其它                                      -> FATAL
    """
    if isinstance(exc, _RETRYABLE):
        return ErrorKind.RETRYABLE
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return ErrorKind.FIXABLE
    return ErrorKind.FATAL


def retry_call(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    base_delay_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> Any:
    """带指数退避的重试。

    延迟 = base_delay_s * 2 ** attempt（0, 0.5, 1.0, ...）
    非 RETRYABLE 的错误立即抛出，不浪费时间。
    """
    if max_attempts < 1:
        raise ValueError("max_attempts 必须 >= 1")

    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            last = e
            kind = classify_error(e)
            is_last = attempt == max_attempts - 1
            if kind is not ErrorKind.RETRYABLE or is_last:
                raise
            sleep(base_delay_s * (2 ** attempt))

    raise last  # pragma: no cover  理论上到不了
