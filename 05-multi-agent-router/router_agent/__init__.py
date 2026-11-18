"""多智能体路由：把工单交给对的专家。

一个可运行的最小实现，用来把「多智能体编排」这件事讲清楚：
    - 编排图怎么搭（节点 / 固定边 / 条件边 / 步数上限）
    - 路由为什么是唯一该让模型做决策的地方
    - 工具调用循环怎么让专家「自己决定检索几次」
    - 检索、生成、校验、记忆为什么要拆成独立节点
    - 检索效果怎么量化（Recall@k / MRR / 跨类污染率）
"""

from .eval import (
    EvalCase,
    EvalReport,
    EvalResult,
    compare_routing,
    evaluate,
    format_table,
)
from .experts import EXPERTS, ExpertBase
from .graph import Graph, GraphError
from .knowledge import Doc, KnowledgeBase
from .knowledge_data import DOCS, EVAL_CASES
from .llm import (
    LLM,
    MockLLM,
    MockToolLLM,
    OpenAICompatibleClient,
    extract_json,
)
from .memory import InMemoryStore, Memory
from .orchestrator import (
    RouterConfig,
    RouterOrchestrator,
    default_knowledge_base,
)
from .router import CONFIDENCE_THRESHOLD, next_after_route, route_node
from .state import ROUTE_LABELS, ROUTES, Ticket, TicketState
from .tool_loop import LoopResult, ToolLLM, build_tool_messages, run_tool_loop
from .tools import (
    ToolRegistry,
    ToolResult,
    ToolSpec,
    make_ask_user_tool,
    make_kb_search_tool,
)

__all__ = [
    # 编排
    "Graph",
    "GraphError",
    "RouterConfig",
    "RouterOrchestrator",
    # 路由
    "CONFIDENCE_THRESHOLD",
    "route_node",
    "next_after_route",
    "ROUTES",
    "ROUTE_LABELS",
    # 状态
    "Ticket",
    "TicketState",
    # 专家
    "EXPERTS",
    "ExpertBase",
    # 知识库
    "Doc",
    "KnowledgeBase",
    "DOCS",
    "EVAL_CASES",
    "default_knowledge_base",
    # LLM
    "LLM",
    "MockLLM",
    "MockToolLLM",
    "OpenAICompatibleClient",
    "extract_json",
    # 工具
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolLLM",
    "make_kb_search_tool",
    "make_ask_user_tool",
    "run_tool_loop",
    "LoopResult",
    "build_tool_messages",
    # 记忆
    "Memory",
    "InMemoryStore",
    # 评估
    "EvalCase",
    "EvalResult",
    "EvalReport",
    "evaluate",
    "compare_routing",
    "format_table",
]

__version__ = "0.2.0"
