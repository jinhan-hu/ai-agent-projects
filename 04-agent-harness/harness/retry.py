"""错误分类、重试与降级兜底。

Agent 的失败不是"成功/失败"二元，要分类型：
  - 可重试（网络抖动、限流）→ 指数退避重试
  - 不可重试（参数错误、工具不存在）→ 直接反馈给模型让它改正
  - 致命（上下文超长、鉴权失败）→ 立刻终止
"""
from __future__ import annotations

import enum
from typing import Any, Callable


class ErrorKind(enum.Enum):
    """错误类型。"""

    RETRYABLE = "retryable"      # 可重试
    FIXABLE = "fixable"          # 可让模型改正参数
    FATAL = "fatal"              # 致命，终止


def classify_error(exc: Exception) -> ErrorKind:
    """把异常归类。

    TODO(11-06): 实现分类规则：
      - TimeoutError / ConnectionError → RETRYABLE
      - ValueError / KeyError（参数问题）→ FIXABLE
      - 其它 → FATAL
    """
    raise NotImplementedError("错误分类待实现（阅读周 11-06）")


def retry_call(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    base_delay_s: float = 0.5,
    **kwargs: Any,
) -> Any:
    """带指数退避的重试。

    TODO(11-06): 实现：
      - 按 classify_error 决定是否重试
      - 延迟 = base_delay_s * 2 ** attempt
      - 超过 max_attempts 抛最后一次异常
    """
    _ = (fn, args, max_attempts, base_delay_s, kwargs)
    raise NotImplementedError("重试逻辑待实现（阅读周 11-06）")
