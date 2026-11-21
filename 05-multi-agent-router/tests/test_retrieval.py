"""测试：混合检索（BM25 / 向量 / RRF / Rerank）与 500 篇规模的评估。

★ 重点不是"分数好看"，而是：
    - 每一路的单独效果能被量化（消融）
    - RRF 的数学性质正确
    - rerank 的特征可解释
    - 大数据集下不崩、不慢
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_agent import Doc, KnowledgeBase  # noqa: E402
from router_agent.knowledge_data import EVAL_CASES as CORE_CASES  # noqa: E402
from router_agent.knowledge_gen import ALL_CASES, ALL_DOCS, GEN_CASES  # noqa: E402
from router_agent.retrieval import (  # noqa: E402
    BM25,
    HybridRetriever,
    Reranker,
    VectorIndex,
    rrf_fuse,
    tokenize,
)

PASS = 0
FAIL = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {name}")
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  FAIL {name}  {detail}")


# ══════════════════ 1. 分词 ══════════════════
def test_tokenize() -> None:
    print("\n[1] 分词")
    t = tokenize("E1024 报错怎么处理")
    check("英文数字按词切", "e1024" in t, str(t[:5]))
    check("中文切出 bigram", "报错" in t, str(t[:12]))
    check("中文保留单字", "报" in t)
    check("空串不崩", tokenize("") == [])


# ══════════════════ 2. BM25 ══════════════════
def test_bm25() -> None:
    print("\n[2] BM25")
    docs = [
        Doc("D1", "安装前置条件", "需要 64 位系统，内存 8GB", "install", "手册"),
        Doc("D2", "报表入口", "报表在分析菜单", "usage", "手册"),
        Doc("D3", "启动报错 E1024", "E1024 是许可证校验失败", "troubleshoot", "手册"),
    ]
    bm = BM25()
    bm.index(docs)

    r = bm.search("E1024", k=3)
    check("精确词命中排第一", r and r[0][0].doc_id == "D3", str([(d.doc_id, round(s, 2)) for d, s in r]))

    r = bm.search("安装", k=3)
    check("中文词可检索", r and r[0][0].doc_id == "D1")

    r = bm.search("安装", k=3, route="usage")
    check("route 过滤生效", all(d.route == "usage" for d, _ in r))

    check("无命中返回空", bm.search("完全无关的词", k=3) == [])
    check("空查询返回空", bm.search("", k=3) == [])

    # IDF 方向：常见词权重应低于稀有词
    check("IDF 为正", all(v > 0 for v in bm.idf.values()), str(list(bm.idf.values())[:3]))


# ══════════════════ 3. 向量索引 ══════════════════
def test_vector() -> None:
    print("\n[3] 向量索引")
    docs = [
        Doc("D1", "安装前置条件", "需要 64 位系统，内存 8GB", "install", "手册"),
        Doc("D2", "报表入口", "报表在分析菜单", "usage", "手册"),
    ]
    vi = VectorIndex(dim=256)
    vi.index(docs)
    check("索引后向量数正确", len(vi.vectors) == 2)
    check("向量维度不超 dim", all(b < 256 for v in vi.vectors for b in v))
    check("范数为正", all(n > 0 for n in vi.norms), str(vi.norms))

    r = vi.search("安装", k=2)
    check("相似度在 0~1", all(0 <= s <= 1.0001 for _, s in r), str([round(s, 3) for _, s in r]))
    check("向量检索能返回结果", len(r) > 0)


# ══════════════════ 4. RRF ══════════════════
def test_rrf() -> None:
    print("\n[4] RRF 融合")
    a = Doc("A", "a", "a", "install", "s")
    b = Doc("B", "b", "b", "usage", "s")
    c = Doc("C", "c", "c", "fallback", "s")

    # A 在两路都第一 → 应排第一
    fused = rrf_fuse([[(a, 10.0), (b, 5.0)], [(a, 0.9), (c, 0.1)]])
    check("两路都命中者排第一", fused[0][0].doc_id == "A", str([(d.doc_id, round(s, 4)) for d, s in fused]))

    # 分数尺度完全不同的两路，融合结果应只看名次
    f1 = rrf_fuse([[(a, 1000.0), (b, 1.0)], [(b, 0.99), (a, 0.01)]])
    f2 = rrf_fuse([[(a, 1.0), (b, 0.5)], [(b, 0.9), (a, 0.1)]])
    check("★ RRF 免疫量纲（同序不同分 → 同结果）",
          [d.doc_id for d, _ in f1] == [d.doc_id for d, _ in f2],
          f"{[d.doc_id for d,_ in f1]} vs {[d.doc_id for d,_ in f2]}")

    # 只在一路出现
    fused = rrf_fuse([[(a, 1.0)], [(c, 1.0)]])
    check("单路命中也能保留", set(d.doc_id for d, _ in fused) == {"A", "C"})

    check("空输入返回空", rrf_fuse([]) == [])


# ══════════════════ 5. Rerank ══════════════════
def test_rerank() -> None:
    print("\n[5] Rerank")
    rr = Reranker()
    d1 = Doc("D1", "安装前置条件", "需要 64 位系统", "install", "安装手册")
    d2 = Doc("D2", "报表入口", "报表在分析菜单", "usage", "用户手册")

    s1, f1 = rr.score("安装前置条件", d1, "install")
    s2, f2 = rr.score("安装前置条件", d2, "install")
    check("相关文档得分更高", s1 > s2, f"{s1:.2f} vs {s2:.2f}")
    check("特征可解释（4 个）", set(f1) == {"title", "phrase", "route", "len"}, str(set(f1)))
    check("标题命中率高", f1["title"] > 0.5, str(f1))

    # rerank 应能改变顺序
    cands = [(d2, 1.0), (d1, 0.5)]     # 初始顺序是错的
    out = rr.rerank("安装前置条件", cands, k=2, route="install")
    check("★ rerank 能纠正错误顺序", out[0][0].doc_id == "D1", str([d.doc_id for d, _ in out]))


# ══════════════════ 6. 三路消融 ══════════════════
def test_ablation() -> None:
    print("\n[6] 消融实验（500 篇库 · 核心 24 条难样本）")

    old = KnowledgeBase(ALL_DOCS)
    hyb = HybridRetriever(ALL_DOCS, mode="bm25")

    def rec(fn, k=3):
        hit = 0
        for c in CORE_CASES:
            r = fn(c.query, k=k, route=c.route)
            if any(d.doc_id in c.expected for d, _ in r):
                hit += 1
        return hit / len(CORE_CASES)

    r_old = rec(lambda q, k=3, **kw: old.search(q, k=k, route=kw.get("route")))
    r_bm = rec(lambda q, k=3, **kw: hyb.search_bm25_only(q, k=k, route=kw.get("route")))
    r_vec = rec(lambda q, k=3, **kw: hyb.search_vec_only(q, k=k, route=kw.get("route")))
    r_rrf = rec(lambda q, k=3, **kw: hyb.search_rrf_no_rerank(q, k=k, route=kw.get("route")))
    r_full = rec(lambda q, k=3, **kw: hyb.search(q, k=k, route=kw.get("route")))

    print(f"     旧 bigram = {r_old:.3f}")
    print(f"     BM25      = {r_bm:.3f}")
    print(f"     Vector    = {r_vec:.3f}")
    print(f"     RRF       = {r_rrf:.3f}")
    print(f"     BM25+rerank = {r_full:.3f}")

    check("★ BM25 优于旧 bigram", r_bm > r_old, f"{r_bm:.3f} vs {r_old:.3f}")
    check("★ 默认模式（bm25+rerank）不低于旧基线", r_full >= r_old, f"{r_full:.3f} vs {r_old:.3f}")
    check("★ 诚实结论：弱向量拖累融合（RRF <= BM25）", r_rrf <= r_bm + 1e-9, f"RRF={r_rrf:.3f} BM25={r_bm:.3f}")
    check("各指标在 [0,1]", all(0 <= x <= 1 for x in (r_old, r_bm, r_vec, r_rrf, r_full)))


# ══════════════════ 7. 规模与稳定性 ══════════════════
def test_scale() -> None:
    print("\n[7] 规模与稳定性")
    check("总文档数 = 500", len(ALL_DOCS) == 500, str(len(ALL_DOCS)))
    check("评估集 = 500", len(ALL_CASES) == 500, str(len(ALL_CASES)))
    check("生成集非空", len(GEN_CASES) > 400, str(len(GEN_CASES)))
    check("doc_id 唯一", len({d.doc_id for d in ALL_DOCS}) == len(ALL_DOCS))

    t0 = time.perf_counter()
    hyb = HybridRetriever(ALL_DOCS, mode="bm25")
    build_ms = (time.perf_counter() - t0) * 1000
    check("建索引 < 2s", build_ms < 2000, f"{build_ms:.0f} ms")

    t0 = time.perf_counter()
    for c in CORE_CASES:
        hyb.search(c.query, k=3, route=c.route)
    per_ms = (time.perf_counter() - t0) / len(CORE_CASES) * 1000
    check("单次检索 < 50ms", per_ms < 50, f"{per_ms:.2f} ms")

    # 模式切换
    h2 = HybridRetriever(ALL_DOCS, mode="hybrid")
    r = h2.search("启动报 E1024", k=3, route="troubleshoot")
    check("hybrid 模式可运行", len(r) > 0)

    h3 = HybridRetriever(ALL_DOCS, mode="hybrid_norerank")
    check("hybrid_norerank 模式可运行", len(h3.search("启动报 E1024", k=3)) > 0)

    try:
        HybridRetriever(ALL_DOCS, mode="bogus")
        check("未知 mode 应报错", False)
    except ValueError:
        check("未知 mode 报错", True)


def main() -> int:
    print("=" * 62)
    print("混合检索 · 测试")
    print("=" * 62)
    test_tokenize()
    test_bm25()
    test_vector()
    test_rrf()
    test_rerank()
    test_ablation()
    test_scale()

    print("\n" + "=" * 62)
    print(f"通过 {PASS} / {PASS + FAIL}")
    if FAILED:
        print("失败项：")
        for f in FAILED:
            print("  -", f)
    print("=" * 62)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
