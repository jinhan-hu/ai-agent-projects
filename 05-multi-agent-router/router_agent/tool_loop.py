"""ReAct 工具调用循环。

这是「模型自主使用工具」的最小实现。三个出口，缺一不可：

  ① 收敛      —— 模型不再请求工具调用 → 返回最终回答
  ② 轮数上限  —— 模型一直在调工具 → 强制停止
  ③ 工具报错  —— 把错误回填给模型，让它自己改正（不是中断循环）

★ 为什么必须有轮数上限：
    模型可能陷入「调工具 → 结果不满意 → 再调同样的工具」的死循环。
    没有上限的 Agent 在生产环境等于一颗定时炸弹。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .tools import ToolRegistry, ToolResult


class ToolLLM(Protocol):
    """支持工具调用的 LLM 接口。"""

    def chat_with_tools(
        self, messages: list[dict], tools: list[dict], temperature: float = 0.0
    ) -> dict:
        """返回 {"content": str, "tool_calls": [{"id","name","arguments"}, ...]}"""
        ...


@dataclass
class LoopResult:
    """循环的执行结果。"""

    answer: str
    rounds: int                              # 实际循环了几轮
    tool_calls: list[dict] = field(default_factory=list)   # 每一次调用记录
    messages: list[dict] = field(default_factory=list)     # 完整对话
    stop_reason: str = "converged"           # converged / max_rounds
    used_tools: bool = False                 # ★ 模型是否真的调过工具
    asked_user: list[str] = field(default_factory=list)    # ★ ask_user 提出的问题

    def stats(self) -> dict:
        return {
            "rounds": self.rounds,
            "tool_calls": len(self.tool_calls),
            "used_tools": self.used_tools,
            "asked_user": len(self.asked_user),
            "stop_reason": self.stop_reason,
        }


def _signature(tool_calls: list[dict]) -> tuple:
    """把一组工具调用压成可比较的签名，用于重复检测。

    两层 sorted：
      - 内层：参数 dict 顺序不影响语义
      - 外层：多个调用的顺序不影响语义
    """
    return tuple(
        sorted(
            (c.get("name", ""), tuple(sorted((c.get("arguments") or {}).items())))
            for c in tool_calls
        )
    )


def run_tool_loop(
    llm: ToolLLM,
    messages: list[dict],
    registry: ToolRegistry,
    *,
    max_rounds: int = 5,
    max_repeat: int = 2,
    temperature: float = 0.2,
) -> LoopResult:
    """让模型自主调用工具，直到它不再请求工具或达到上限。

    ★ 重复检测：同一组调用重复超过 max_repeat 次 → 停止（和轮数上限互补）。
    """
    msgs = list(messages)
    schemas = registry.to_schema()
    all_calls: list[dict] = []
    asked: list[str] = []
    seen: dict[tuple, int] = {}

    for round_idx in range(1, max_rounds + 1):
        resp = llm.chat_with_tools(msgs, schemas, temperature=temperature)
        content = resp.get("content") or ""
        tool_calls = resp.get("tool_calls") or []

        # ① 收敛：不调工具了
        if not tool_calls:
            msgs.append({"role": "assistant", "content": content})
            return LoopResult(
                answer=content,
                rounds=round_idx,
                tool_calls=all_calls,
                messages=msgs,
                stop_reason="converged",
                used_tools=bool(all_calls),
                asked_user=asked,
            )

        # ② 重复检测：同一组调用重复太多次
        sig = _signature(tool_calls)
        seen[sig] = seen.get(sig, 0) + 1
        if seen[sig] > max_repeat:
            detail = ", ".join(c.get("name", "?") for c in tool_calls)
            note = (
                f"（检测到重复的工具调用 [{detail}]，已停止。"
                f"基于目前掌握的信息无法得出确切结论，建议转人工处理。）"
            )
            # ★ 无论模型说了什么，都要附上这段说明——
            # 否则调用方拿到的可能是一句「再查一次」，用户看不懂发生了什么
            answer = f"{content}\n\n{note}" if content.strip() else note
            msgs.append({"role": "assistant", "content": answer})
            return LoopResult(
                answer=answer,
                rounds=round_idx,
                tool_calls=all_calls,
                messages=msgs,
                stop_reason="loop_detected",
                used_tools=True,
                asked_user=asked,
            )

        # 记录 assistant 的这次工具请求
        msgs.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls}
        )

        # ③ 执行工具，把结果回填
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments") or {}
            call_id = call.get("id") or f"call-{round_idx}-{len(all_calls)}"

            result: ToolResult = registry.execute(name, args)
            all_calls.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": args,
                    "ok": result.ok,
                    "error": result.error,
                }
            )
            if result.needs_user:
                q = str(args.get("question", "")).strip()
                if q:
                    asked.append(q)

            msgs.append(result.to_message(call_id))

    # 轮数用尽：软着陆——不带工具再问一次，让模型基于已有信息总结
    msgs.append(
        {
            "role": "user",
            "content": (
                "工具调用轮数已达上限。请不要再调用工具，"
                "直接基于已经获得的信息给出最终回答；"
                "如果信息不足，请如实说明缺什么。"
            ),
        }
    )
    final = llm.chat_with_tools(msgs, [], temperature=temperature)
    return LoopResult(
        answer=final.get("content") or "（未能生成最终回答）",
        rounds=max_rounds,
        tool_calls=all_calls,
        messages=msgs,
        stop_reason="max_rounds",
        used_tools=bool(all_calls),
        asked_user=asked,
    )


def build_tool_messages(system: str, user: str) -> list[dict]:
    """构造初始消息。"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
