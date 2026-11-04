"""多步 Agent demo：算数 → 写文件 → 读回确认。

用来演示 Agent Loop 的「多轮工具调用」，也是录 1 分钟视频的最佳素材。

运行
----
    python examples/demo_multistep.py            # 离线（MockLLM）
    python examples/demo_multistep.py --real     # 接真实模型
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.loop import AgentLoop                         # noqa: E402
from harness.registry import ToolRegistry                  # noqa: E402
from harness.tracing import trace_summary, trace_to_json   # noqa: E402
from harness.mock_llm import (                             # noqa: E402
    MockLLM,
    assistant_text,
    assistant_tool_call,
)
from tools.example_tools import calculator, read_file, write_file  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_demo_output")
TARGET = os.path.join(OUT_DIR, "result.txt")

TASK = (
    "请按顺序完成三件事：\n"
    "1) 计算 (120 + 80) * 3 的结果；\n"
    "2) 把结果写入文件 _demo_output/result.txt；\n"
    "3) 读回这个文件确认内容正确。"
)


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(description="计算数学表达式，输入如 '(120 + 80) * 3'")(calculator)
    reg.tool(description="把内容写入文本文件")(write_file)
    reg.tool(description="读取文本文件内容")(read_file)
    return reg


def build_llm() -> object:
    if "--real" in sys.argv:
        from harness.llm_client import OpenAICompatibleClient

        base = os.environ.get("LLM_BASE_URL")
        key = os.environ.get("LLM_API_KEY")
        model = os.environ.get("LLM_MODEL")
        if not (base and key and model):
            raise SystemExit("请先设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL")
        print(f"[真实模型] {base} / {model}")
        return OpenAICompatibleClient(base_url=base, api_key=key, model=model)

    print("[离线演示] MockLLM —— 固定脚本，不联网")
    return MockLLM(
        [
            assistant_tool_call("calculator", {"expression": "(120 + 80) * 3"}, call_id="c1"),
            assistant_tool_call("write_file", {"path": "_demo_output/result.txt", "content": "600"}, call_id="c2"),
            assistant_tool_call("read_file", {"path": "_demo_output/result.txt"}, call_id="c3"),
            assistant_text("三步已完成：(120+80)×3 = 600，已写入并读回确认。"),
        ]
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    reg = build_registry()
    loop = AgentLoop(build_llm(), reg, max_steps=8, verbose=True)

    print("\n可用工具：")
    for t in reg.list_tools():
        print(f"  - {t.name}: {t.description}")

    print(f"\n任务：\n{TASK}\n" + "-" * 66)
    trace = loop.run(TASK)
    print("-" * 66)

    print(f"最终答案：{trace.final_answer}")
    print(f"停止原因：{trace.stop_reason}")
    print(f"步数：{len(trace.steps)} | 工具调用：{sum(len(s.calls) for s in trace.steps)}")
    print(f"摘要：{trace_summary(trace)}")

    print("\n每一步做了什么：")
    for st in trace.steps:
        for c, r in zip(st.calls, st.results):
            mark = "OK " if r.ok else "ERR"
            print(f"  [{mark}] {c.name}({c.arguments}) -> {r.output[:60]!r} ({r.duration_ms}ms)")

    print("\n完整 trace：")
    print(trace_to_json(trace))


if __name__ == "__main__":
    main()
