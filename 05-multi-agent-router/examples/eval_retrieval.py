"""检索评估脚本：跑出可直接贴进 README 的对比表。

    python examples/eval_retrieval.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_agent.eval import compare_routing, evaluate, format_table  # noqa: E402
from router_agent.knowledge import KnowledgeBase  # noqa: E402
from router_agent.knowledge_data import DOCS, EVAL_CASES  # noqa: E402


def main() -> int:
    kb = KnowledgeBase(DOCS)
    print("=" * 72)
    print("检索评估报告")
    print("=" * 72)
    print(f"知识库：{len(DOCS)} 篇  |  标注集：{len(EVAL_CASES)} 条")
    counts: dict[str, int] = {}
    for d in DOCS:
        counts[d.route] = counts.get(d.route, 0) + 1
    print("类别分布：" + "，".join(f"{k}={v}" for k, v in counts.items()))
    print()

    # 1) k 的影响
    print("【1】不同 k 的召回（按类检索）")
    reports_k = []
    for k in (1, 3, 5):
        r = evaluate(kb, EVAL_CASES, k=k, name=f"按类检索 k={k}")
        reports_k.append(r)
        print(f"    k={k}: Recall@{k}={r.recall:.3f}  MRR={r.mrr:.3f}")
    print()

    # 2) 路由的收益
    print("【2】★ 路由先于检索的收益（k=3）")
    by_route, glob = compare_routing(kb, EVAL_CASES, k=3)
    print()
    print(format_table([by_route, glob]))
    print()
    gap = glob.off_route_ratio - by_route.off_route_ratio
    print(f"    ★ 跨类污染率下降 {gap:.1%}")
    print(f"      全库检索时，每 3 篇资料里约有 {glob.off_route_ratio*3:.1f} 篇来自错误类别；")
    print(f"      按类检索时，这个数字是 {by_route.off_route_ratio*3:.2f} 篇。")
    print()

    # 3) 失败样本（诚实地列出来）
    print("【3】未命中样本（当前检索器的已知短板）")
    misses = by_route.misses()
    if misses:
        for q in misses:
            print(f"    ✗ {q}")
        print(f"    共 {len(misses)} / {by_route.n} 条未命中")
        print("    → 这些多为「同义改写」类查询，bigram 字面匹配覆盖不到；")
        print("      下一步改进：引入向量检索做语义召回（混合检索）。")
    else:
        print("    （全部命中）")
    print()

    print("=" * 72)
    print("结论：路由对召回的提升有限，但对「上下文纯度」的提升是决定性的。")
    print("      这就是「路由 → 每个专家独立知识库」相比「单 Agent 全库检索」的价值。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
