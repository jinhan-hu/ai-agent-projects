"""端到端 demo：让 Agent 用工具完成一个多步任务。

运行
----
    python examples/demo.py            # 离线演示（MockLLM，无需 API key）
    python examples/demo.py --real     # 接真实模型（需先设环境变量）

真实模型需要的环境变量：
    LLM_BASE_URL   例如 https://api.deepseek.com
    LLM_API_KEY    sk-...
    LLM_MODEL      例如 deepseek-flash
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.loop import AgentLoop                        # noqa: E402
from harness.registry import ToolRegistry                 # noqa: E402
from harness.tracing import trace_summary, trace_to_json  # noqa: E402
from harness.types import Message                         # noqa: E402
from harness.mock_llm import (                            # noqa: E402
    MockLLM,
    assistant_text,
    assistant_tool_call,
)
from tools.example_tools import calculator, read_file     # noqa: E402


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.tool(description="计算数学表达式")(calculator)
    reg.tool(description="读取文本文件")(read_file)
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

    print("[离线演示] 使用 MockLLM，无需网络")
    return MockLLM(
        [
            assistant_tool_call(
                "calculator", {"expression": "(120 + 80) * 3"}, call_id="c1"
            ),
            assistant_text("(120 + 80) × 3 = 600"),
        ]
    )


def main() -> None:
    reg = build_registry()
    llm = build_llm()

    print("\n已注册工具：")
    for t in reg.list_tools():
        print(f"  - {t.name}: {t.description}")

    loop = AgentLoop(llm, reg, max_steps=5, verbose=True)
    task = "帮我算一下 (120 + 80) * 3 等于多少"

    print(f"\n用户任务：{task}")
    print("-" * 60)

    trace = loop.run(task)

    print("-" * 60)
    print(f"最终答案：{trace.final_answer}")
    print(f"停止原因：{trace.stop_reason}")
    print(f"摘要：{trace_summary(trace)}")

    print("\n完整 trace（前 800 字符）：")
    print(trace_to_json(trace)[:800])


if __name__ == "__main__":
    main()

