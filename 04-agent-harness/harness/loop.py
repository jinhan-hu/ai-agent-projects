"""Agent Loop 主循环：思考 -> 调用 -> 观察 -> 收敛。

这是整个框架的心脏。其余模块都是它的「零件」。

三个出口（循环必须能停）：
  ① converged       模型不再请求工具调用 -> 直接返回答案
  ② max_steps       步数用尽 -> 软着陆（不带 tools 再问一次，让模型总结）
  ③ loop_detected   相同工具调用重复超过阈值 -> 软着陆并标记原因
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Protocol

from .registry import ToolRegistry
from .sandbox import run_in_sandbox
from .types import Message, Step, ToolCall, ToolResult, Trace


class LLMClient(Protocol):
    """LLM 客户端接口 —— 抽象出来，方便接任意模型。"""

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        """给定对话历史和工具 schema，返回下一条 assistant 消息。"""
        ...


def _signature(calls: list[ToolCall]) -> tuple:
    """把一组工具调用压成可比较的签名，用于循环检测。"""
    return tuple(sorted((c.name, tuple(sorted(c.arguments.items()))) for c in calls))


class AgentLoop:
    """Agent 主循环。"""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        max_steps: int = 10,
        system_prompt: str = "你是一个能调用工具的助手。需要时调用工具，不要凭空猜测。",
        timeout_s: float = 10.0,
        max_repeat: int = 2,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s
        self.max_repeat = max_repeat
        self.verbose = verbose
        # 统计：一共调了几次模型
        self.llm_calls = 0

    # ------------------------------------------------------------------
    def run(self, task: str) -> Trace:
        """执行一个任务，返回完整 trace。"""
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=task),
        ]
        trace = Trace(task=task)
        seen: Counter = Counter()
        self.llm_calls = 0

        for step_idx in range(self.max_steps):
            # ① 调模型
            resp = self.llm.chat(messages, self.registry.to_openai_schema())
            self.llm_calls += 1
            messages.append(resp)

            # ② 正常收敛
            if not resp.wants_tools:
                trace.final_answer = resp.content
                trace.stop_reason = "converged"
                return trace

            # ③ 循环检测（同一组调用重复太多次）
            sig = _signature(resp.tool_calls)
            seen[sig] += 1
            if seen[sig] > self.max_repeat:
                trace.stop_reason = "loop_detected"
                trace.final_answer = self._soft_landing(
                    messages,
                    hint="你已多次重复相同的工具调用，请不要再调用工具，直接基于现有信息给出最终答案。",
                )
                return trace

            # ④ 执行工具
            step = Step(index=step_idx, thought=resp.content, calls=list(resp.tool_calls))
            for call in resp.tool_calls:
                result = run_in_sandbox(
                    call, self.registry, timeout_s=self.timeout_s
                )
                step.results.append(result)
                messages.append(result.to_message())
            trace.steps.append(step)

            if self.verbose:
                self._log_step(step)

        # ⑤ 步数用尽 -> 软着陆
        trace.stop_reason = "max_steps"
        trace.final_answer = self._soft_landing(
            messages,
            hint="已达到工具调用步数上限，请不要再调用工具，直接基于现有信息给出最终答案。",
        )
        return trace

    # ------------------------------------------------------------------
    def _soft_landing(self, messages: list[Message], *, hint: str) -> str:
        """软着陆：不带 tools 再问一次，让模型基于已有信息总结。"""
        try:
            resp = self.llm.chat(
                list(messages) + [Message(role="user", content=hint)], []
            )
            self.llm_calls += 1
            return resp.content
        except Exception as e:  # noqa: BLE001  软着陆失败也不能崩
            return f"（未能生成最终答案：{type(e).__name__}: {e}）"

    def _log_step(self, step: Step) -> None:
        names = ", ".join(c.name for c in step.calls)
        oks = sum(1 for r in step.results if r.ok)
        print(f"    step {step.index}: 调用 [{names}] -> {oks}/{len(step.results)} 成功")

