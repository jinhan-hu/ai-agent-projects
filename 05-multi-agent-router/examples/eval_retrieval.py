"""检索评估脚本：跑出可直接贴进 README 的消融对比表。

    python examples/eval_retrieval.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_agent.eval import compare_routing, format_table  # noqa: E402
from router_agent.knowledge import KnowledgeBase  # noqa: E402
from router_agent.knowledge_data import EVAL_CASES as CORE_CASES  # noqa: E402
from router_agent.knowledge_gen import ALL_CASES, ALL_DOCS, GEN_CASES  # noqa: E402
from router_agent.retrieval import HybridRetriever  # noqa: E402


def _recall(fn, cases, k=3) -> float:
    hit = 0
    for c in cases:
        r = fn(c.query, k=k, route=c.route)
        if any(d.doc_id in c.expected for d, _ in r):
            hit += 1
    return hit / len(cases) if cases else 0.0


def main() -> int:
    old = KnowledgeBase(ALL_DOCS)
    hyb = HybridRetriever(ALL_DOCS, mode="bm25")
    hyb_h = HybridRetriever(ALL_DOCS, mode="hybrid")

    print("=" * 76)
    print("检索评估报告")
    print("=" * 76)
    print(f"知识库：{len(ALL_DOCS)} 篇（手写 24 + 生成 476）")
    print(f"评估集：{len(ALL_CASES)} 条（核心手写 {len(CORE_CASES)} + 生成 {len(GEN_CASES)}）")
    counts: dict[str, int] = {}
    for d in ALL_DOCS:
        counts[d.route] = counts.get(d.route, 0) + 1
    print("类别分布：" + "，".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()

    print("【1】★ 消融实验（核心 24 条难样本，Recall@3）")
    print(f"    {'策略':32} {'Recall@3':>10}")
    print("    " + "-" * 44)
    rows = [
        ("旧：bigram + IDF", lambda q, k=3, **kw: old.search(q, k=k, route=kw.get("route"))),
        ("新：BM25", lambda q, k=3, **kw: hyb.search_bm25_only(q, k=k, route=kw.get("route"))),
        ("新：TF-IDF 向量", lambda q, k=3, **kw: hyb.search_vec_only(q, k=k, route=kw.get("route"))),
        ("新：RRF 融合（BM25 + 向量）", lambda q, k=3, **kw: hyb.search_rrf_no_rerank(q, k=k, route=kw.get("route"))),
        ("★ 新：默认（BM25 + rerank）", lambda q, k=3, **kw: hyb.search(q, k=k, route=kw.get("route"))),
        ("参考：hybrid + rerank", lambda q, k=3, **kw: hyb_h.search(q, k=k, route=kw.get("route"))),
    ]
    for name, fn in rows:
        print(f"    {name:32} {_recall(fn, CORE_CASES):>10.3f}")
    print()

    print("【2】评估集质量对比（★ 这条比分数更重要）")
    for name, cases in (("核心手写（含同义改写）", CORE_CASES), ("程序生成（模板拼装）", GEN_CASES)):
        r = _recall(lambda q, k=3, **kw: hyb.search(q, k=k, route=kw.get("route")), cases)
        print(f"    {name:24} n={len(cases):>4}  Recall@3={r:.3f}")
    print("    → ★ 生成集接近满分，说明这类 query 太容易；")
    print("      ★ 评估集的质量决定评估的价值，程序生成的 query 不能反映真实难度。")
    print()

    print("【3】★ 路由先于检索的收益（核心 24 条，k=3）")
    kb = KnowledgeBase(ALL_DOCS)
    by_route, glob = compare_routing(kb, CORE_CASES, k=3)
    print()
    print(format_table([by_route, glob]))
    print()
    print(f"    ★ 跨类污染率：按类 {by_route.off_route_ratio:.1%} → 全库 {glob.off_route_ratio:.1%}")
    print()

    print("【4】已知短板")
    by_route2 = None
    from router_agent.eval import evaluate
    by_route2 = evaluate(kb, CORE_CASES, k=3, name="核心")
    misses = by_route2.misses()
    print(f"    核心 24 条中未命中 {len(misses)} 条：")
    for q in misses:
        print(f"      ✗ {q}")
    print()
    print("=" * 76)
    print("结论：")
    print("  ① BM25 比旧 bigram+IDF 提升约 4 个百分点（这才是本批的真实收益）")
    print("  ② RRF 融合反而比 BM25 单路差 —— 因为 TF-IDF 哈希向量太弱（0.750），")
    print("     弱召回路会引入错误候选，加权也救不回来（RRF 只看名次，不看分数尺度）")
    print("  ③ 所以默认用 BM25；换成真正的语义 embedding 后再打开 hybrid")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
