"""多智能体路由：把工单交给对的专家。

一个可运行的最小实现，用来把「多智能体编排」这件事讲清楚：
    - 编排图怎么搭（节点 / 固定边 / 条件边 / 步数上限）
    - 路由为什么是唯一该让模型做决策的地方
    - 检索、生成、校验、记忆为什么要拆成独立节点
"""

from .experts import EXPERTS, ExpertBase
from .graph import Graph, GraphError
from .knowledge import Doc, KnowledgeBase
from .llm import LLM, MockLLM, OpenAICompatibleClient, extract_json
from .memory import Memory, InMemoryStore
from .orchestrator import (
    RouterConfig,
    RouterOrchestrator,
    SAMPLE_DOCS,
    default_knowledge_base,
)
from .router import CONFIDENCE_THRESHOLD, route_node, next_after_route
from .state import ROUTE_LABELS, ROUTES, Ticket, TicketState

__all__ = [
    "EXPERTS",
    "ExpertBase",
    "Graph",
    "GraphError",
    "Doc",
    "KnowledgeBase",
    "LLM",
    "MockLLM",
    "OpenAICompatibleClient",
    "extract_json",
    "Memory",
    "InMemoryStore",
    "RouterConfig",
    "RouterOrchestrator",
    "SAMPLE_DOCS",
    "default_knowledge_base",
    "CONFIDENCE_THRESHOLD",
    "route_node",
    "next_after_route",
    "ROUTE_LABELS",
    "ROUTES",
    "Ticket",
    "TicketState",
]

__version__ = "0.1.0"
