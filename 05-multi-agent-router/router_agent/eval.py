"""检索评估：Recall@k / MRR / Hit@k，以及「路由先于检索」的收益量化。

为什么评估要单独成一个模块：
    没有评估的检索优化是"我觉得这样更好"。
    有了评估才能说"按类检索比全库检索 Recall@3 高 21 个点"。

三个指标各衡量什么：
    Recall@k  —— 该找到的文档，有多少落在了前 k 条里（覆盖面）
    MRR       —— 第一个正确结果排在第几位（排序质量）
    Hit@k     —— 前 k 条里有没有命中（有没有"颗粒无收"）

★ 这三者不可互相替代：
    一个策略可能 Recall 高但 MRR 低（全找回来了，但排得靠后）；
    也可能 MRR 高但 Recall 低（第一条很准，但漏了不少）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .knowledge import KnowledgeBase


@dataclass
class EvalCase:
    """一条评估样本。"""

    query: str
    expected: set[str]                  # 期望命中的 doc_id 集合
    route: str | None = None            # 该 query 应走的路由（用于按类检索）


@dataclass
class EvalResult:
    """单条样本的评估结果。"""

    query: str
    hit_ids: list[str]
    first_hit_rank: int | None          # 第一个命中的名次（1-based），无命中为 None
    n_expected: int = 1                 # 该 query 期望命中的文档数
    n_found: int = 0                    # 实际命中的期望文档数
    # ★ 返回文档的类别（用于算「跨类污染率」）
    hit_routes: list[str] = field(default_factory=list)
    expected_route: str | None = None

    @property
    def hit(self) -> bool:
        return self.first_hit_rank is not None

    @property
    def recall_at_k(self) -> float:
        """该样本的召回率（支持期望多个文档的情形）。"""
        return self.n_found / self.n_expected if self.n_expected else 0.0

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_hit_rank if self.first_hit_rank else 0.0

    @property
    def off_route_count(self) -> int:
        """返回的文档里，不属于期望类别的篇数。"""
        if not self.expected_route:
            return 0
        return sum(1 for r in self.hit_routes if r != self.expected_route)


@dataclass
class EvalReport:
    """整体报告。"""

    name: str
    k: int
    results: list[EvalResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def recall(self) -> float:
        """平均 Recall@k（逐样本求召回率再平均，支持期望多文档的样本）。"""
        if not self.results:
            return 0.0
        return sum(r.recall_at_k for r in self.results) / self.n

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / self.n

    @property
    def hit_rate(self) -> float:
        return self.recall

    @property
    def off_route_docs(self) -> int:
        """★ 跨类污染：返回文档里不属于期望类别的总篇数。"""
        return sum(r.off_route_count for r in self.results)

    @property
    def total_docs(self) -> int:
        return sum(len(r.hit_routes) for r in self.results)

    @property
    def off_route_ratio(self) -> float:
        """★ 跨类污染率 = 错误类别的文档 / 返回文档总数。

        衡量的是「喂给模型的上下文有多干净」——
        这正是「路由先于检索」最直接、也最容易被忽略的收益。
        """
        return self.off_route_docs / self.total_docs if self.total_docs else 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "k": self.k,
            "n": self.n,
            f"Recall@{self.k}": round(self.recall, 4),
            "MRR": round(self.mrr, 4),
            "跨类污染率": round(self.off_route_ratio, 4),
        }

    def misses(self) -> list[str]:
        return [r.query for r in self.results if not r.hit]


def _score_case(case: EvalCase, hits: list[tuple]) -> EvalResult:
    """统一的打分逻辑（受控检索与全库检索共用）。"""
    hit_ids = [doc.doc_id for doc, _ in hits]
    first_rank = None
    for rank, doc_id in enumerate(hit_ids, start=1):
        if doc_id in case.expected:
            first_rank = rank
            break
    found = len(set(hit_ids) & case.expected)
    return EvalResult(
        query=case.query,
        hit_ids=hit_ids,
        first_hit_rank=first_rank,
        n_expected=len(case.expected),
        n_found=found,
        hit_routes=[doc.route for doc, _ in hits],
        expected_route=case.route,
    )


def evaluate(
    kb: KnowledgeBase,
    cases: list[EvalCase],
    k: int = 3,
    name: str = "default",
) -> EvalReport:
    """跑一组样本（按样本自带的 route 做受控检索）。"""
    report = EvalReport(name=name, k=k)
    for case in cases:
        hits = kb.search(case.query, k=k, route=case.route)
        report.results.append(_score_case(case, hits))
    return report


def compare_routing(
    kb: KnowledgeBase,
    cases: list[EvalCase],
    k: int = 3,
) -> tuple[EvalReport, EvalReport]:
    """★ 量化「路由先于检索」的收益。

    同一批样本跑两次：
      - by_route：路由已知，只在对应类别里检索（检索空间小）
      - global  ：路由未知，全库检索（检索空间大）

    ★ 通常 Recall 差距不大，真正的差距在「跨类污染率」：
      全库检索会把错误的类别的文档也塞进上下文，
      而按类检索天然是 0%。
    """
    by_route = evaluate(kb, cases, k=k, name="按类检索（路由已知）")
    global_ = EvalReport(name="全库检索（路由未知）", k=k)
    for case in cases:
        hits = kb.search(case.query, k=k, route=None)
        global_.results.append(_score_case(case, hits))
    return by_route, global_


def format_table(reports: list[EvalReport]) -> str:
    """把多份报告渲染成 Markdown 表，直接贴进 README。"""
    lines = [
        "| 策略 | n | Recall@k | MRR | ★ 跨类污染率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in reports:
        lines.append(
            f"| {r.name} | {r.n} | {r.recall:.3f} | {r.mrr:.3f} | "
            f"**{r.off_route_ratio:.1%}** |"
        )
    return "\n".join(lines)
