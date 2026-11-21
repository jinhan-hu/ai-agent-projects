"""混合检索：BM25（稀疏）+ 向量（稠密）+ RRF 融合 + Rerank。

为什么要有这一层：
    bigram + IDF 只能做「字面匹配」，遇到同义改写就失效
    （实测：'装之前要准备什么环境' 检索不到 '安装前置条件'）。
    混合检索用两路互补的信号把召回补回来：
        BM25  —— 擅长精确词命中
        向量   —— 擅长近义表达
        RRF    —— 融合两路排序（只看名次，不看分数尺度，天然免调权）

★ 关于「向量」的诚实说明：
    这里用的是 TF-IDF 加权向量 + 余弦相似度，
    它是「稠密向量检索」的简化版，但**不是神经语义向量**。
    真正的语义召回应由 embedding 模型提供（见 README「扩展点」）。
    ★ 但 RRF 融合、rerank、评估这一整套框架是通用的——
      把 VectorIndex 换成真 embedding 索引，上层不用改。
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from .knowledge import Doc

# ══════════════════════ 分词 ══════════════════════

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混合分词。

    - 英文/数字：按词切，统一小写
    - 中文：切**字符 bigram**（无词典时的通用做法，召回高）
    """
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    cjk = "".join(_CJK_RE.findall(text))
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i : i + 2])
    # 单字也留一份（处理单字查询）
    tokens.extend(list(cjk))
    return tokens


# ══════════════════════ BM25 ══════════════════════

@dataclass
class BM25:
    """Okapi BM25 —— 经典的稀疏检索基线。

    score(q, d) = Σ_t IDF(t) * (f(t,d) * (k1+1)) / (f(t,d) + k1 * (1 - b + b * |d|/avgdl))
    """

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.docs: list[Doc] = []
        self.tf: list[Counter] = []
        self.lengths: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self._postings: dict[str, list[int]] = {}

    def index(self, docs: list[Doc]) -> None:
        self.docs = list(docs)
        self.tf = []
        self.lengths = []
        self._postings = {}
        df: Counter = Counter()

        for i, doc in enumerate(self.docs):
            toks = tokenize(doc.title + " " + doc.text)
            c = Counter(toks)
            self.tf.append(c)
            self.lengths.append(len(toks))
            for t in c:
                df[t] += 1
                self._postings.setdefault(t, []).append(i)

        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        n = len(self.docs)
        # BM25 的 IDF（加 1 防止为负）
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score_doc(self, q_tokens: list[str], i: int) -> float:
        s = 0.0
        tf = self.tf[i]
        dl = self.lengths[i] or 1
        denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
        for t in q_tokens:
            f = tf.get(t)
            if not f:
                continue
            idf = self.idf.get(t, 0.0)
            s += idf * (f * (self.k1 + 1)) / (f + denom_norm)
        return s

    def search(self, query: str, k: int = 10, route: str | None = None) -> list[tuple[Doc, float]]:
        q = tokenize(query)
        cand: set[int] = set()
        for t in q:
            cand.update(self._postings.get(t, ()))
        if not cand:
            return []
        scored = []
        for i in cand:
            doc = self.docs[i]
            if route is not None and doc.route != route:
                continue
            s = self.score_doc(q, i)
            if s > 0:
                scored.append((doc, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ══════════════════════ 向量检索 ══════════════════════

class VectorIndex:
    """TF-IDF 加权稀疏向量 + 余弦相似度。

    ★ 用「哈希到固定维度」的方式把 token 压成稠密向量，
      这样两个词的共现模式能以向量内积的形式体现出来。
    ★ 它不是神经语义向量 —— 但对同义/近义的中文表达，
      字符 bigram 的向量重叠比纯 IDF 打分更宽容。
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.docs: list[Doc] = []
        self.vectors: list[dict[int, float]] = []
        self.norms: list[float] = []
        self.idf: dict[str, float] = {}

    def _bucket(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") % self.dim

    def _vec(self, tokens: list[str]) -> dict[int, float]:
        v: dict[int, float] = {}
        for t in tokens:
            w = self.idf.get(t, 1.0)
            b = self._bucket(t)
            v[b] = v.get(b, 0.0) + w
        return v

    @staticmethod
    def _norm(v: dict[int, float]) -> float:
        return math.sqrt(sum(x * x for x in v.values())) or 1.0

    def index(self, docs: list[Doc]) -> None:
        self.docs = list(docs)
        tokenized = [tokenize(d.title + " " + d.text) for d in self.docs]
        df: Counter = Counter()
        for toks in tokenized:
            for t in set(toks):
                df[t] += 1
        n = len(self.docs)
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        self.vectors = [self._vec(toks) for toks in tokenized]
        self.norms = [self._norm(v) for v in self.vectors]

    def search(self, query: str, k: int = 10, route: str | None = None) -> list[tuple[Doc, float]]:
        qvec = self._vec(tokenize(query))
        qn = self._norm(qvec)
        scored = []
        for i, dv in enumerate(self.vectors):
            doc = self.docs[i]
            if route is not None and doc.route != route:
                continue
            # 稀疏 dict 的内积
            dot = 0.0
            small, big = (qvec, dv) if len(qvec) < len(dv) else (dv, qvec)
            for b, w in small.items():
                x = big.get(b)
                if x:
                    dot += w * x
            if dot <= 0:
                continue
            scored.append((doc, dot / (qn * self.norms[i])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ══════════════════════ RRF 融合 ══════════════════════

def rrf_fuse(
    rankings: list[list[tuple[Doc, float]]],
    k: int = 10,
    rrf_k: int = 60,
) -> list[tuple[Doc, float]]:
    """Reciprocal Rank Fusion。

    RRF(d) = Σ_i 1 / (rrf_k + rank_i(d))

    ★ 为什么用 RRF 而不是加权求和：
      BM25 的分数是「无界正数」，余弦是「0~1」——量纲完全不同，
      直接加权需要精细调参，而且换数据集就失效。
      RRF 只看**名次**，天然免疫量纲问题，几乎没有超参。
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Doc] = {}
    for ranking in rankings:
        for rank, (doc, _score) in enumerate(ranking, start=1):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + 1.0 / (rrf_k + rank)
            by_id[doc.doc_id] = doc
    fused = [(by_id[did], s) for did, s in scores.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:k]


# ══════════════════════ Rerank ══════════════════════

class Reranker:
    """轻量 rerank：用可解释的交叉特征重排候选。

    ★ 为什么需要 rerank：
      召回阶段（BM25/向量）是「粗排」，只看单一信号。
      rerank 阶段能同时看 query 和 doc 的**多个交叉特征**，
      把真正相关的往前挪。

    特征（都可解释，不做黑盒）：
      1. 标题命中率   —— query 的 token 有多少出现在标题里
      2. 短语命中     —— query 的连续子串是否原样出现在正文
      3. 路由匹配     —— 文档类别与查询类别是否一致
      4. 长度惩罚     —— 过长的文档稀释相关信息
    """

    def __init__(self, w_title: float = 2.0, w_phrase: float = 1.5,
                 w_route: float = 1.0, w_len: float = 0.5) -> None:
        self.w_title = w_title
        self.w_phrase = w_phrase
        self.w_route = w_route
        self.w_len = w_len

    def score(self, query: str, doc: Doc, route: str | None) -> tuple[float, dict]:
        q_tokens = set(tokenize(query))
        title_tokens = set(tokenize(doc.title))
        feats: dict[str, float] = {}

        feats["title"] = (len(q_tokens & title_tokens) / len(q_tokens)) if q_tokens else 0.0

        # 短语命中：query 里长度 >= 3 的连续片段是否原样出现
        ph = 0.0
        body = doc.title + doc.text
        for n in (6, 5, 4, 3):
            for i in range(len(query) - n + 1):
                seg = query[i : i + n]
                if seg in body:
                    ph = max(ph, n / 6)
                    break
            if ph:
                break
        feats["phrase"] = ph

        feats["route"] = 1.0 if (route and doc.route == route) else 0.0
        feats["len"] = min(len(doc.text) / 400.0, 1.0)

        s = (
            self.w_title * feats["title"]
            + self.w_phrase * feats["phrase"]
            + self.w_route * feats["route"]
            - self.w_len * feats["len"]
        )
        return s, feats

    def rerank(
        self, query: str, cands: list[tuple[Doc, float]], k: int = 10,
        route: str | None = None, rrf_weight: float = 1.0,
    ) -> list[tuple[Doc, float]]:
        out = []
        for doc, rrf_score in cands:
            s, _ = self.score(query, doc, route)
            out.append((doc, rrf_weight * rrf_score + s))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:k]


# ══════════════════════ 混合检索器 ══════════════════════

class HybridRetriever:
    """BM25 + 向量 → RRF → Rerank 的完整管线。

    ★ 默认模式是 "bm25"，不是 "hybrid"。这是**实测数据决定的**，不是偷懒：

        500 篇库 · 核心 24 条难样本 · Recall@3
            bigram+IDF（旧）        0.917
            ★ BM25                 0.958   ← 默认
            TF-IDF 向量             0.750
            RRF 融合（1:1）          0.875   ← 比 BM25 单路更差！
            RRF 融合（8:1）          0.875   ← 加权也救不回来
            RRF + rerank            0.875

    ★ 为什么加权也无效：
        RRF 的分数是 w/(rrf_k + rank)。即使把向量权重压到 1/8，
        只要某篇文档**只被向量召回**，它仍然能挤进 top-k。
        所以问题不是"排序权重"，是**向量那一路引入了错误候选**。

    ★ 由此得到一条通用结论：
        混合检索的前提是「两路各自都有独立价值」。
        如果一路明显弱于另一路，用它做**兜底**（fallback）比做**融合**（fusion）更合理。
        等换成真正的语义 embedding，再打开 "hybrid" 模式即可。
    """

    def __init__(
        self,
        docs: list[Doc],
        *,
        mode: str = "bm25",          # "bm25" | "hybrid" | "hybrid_norerank"
        use_rerank: bool = True,
        recall_k: int = 20,
        dim: int = 512,
    ) -> None:
        if mode not in ("bm25", "hybrid", "hybrid_norerank"):
            raise ValueError(f"未知 mode：{mode}")
        self.docs = list(docs)
        self.mode = mode
        self.use_rerank = use_rerank
        self.recall_k = recall_k
        self.bm25 = BM25()
        self.vec = VectorIndex(dim=dim)
        self.reranker = Reranker()
        self.bm25.index(self.docs)
        self.vec.index(self.docs)
        # 统计用
        self.stats = {"calls": 0, "bm25_only": 0, "vec_only": 0, "both": 0}

    def search(
        self, query: str, k: int = 3, route: str | None = None
    ) -> list[tuple[Doc, float]]:
        self.stats["calls"] += 1

        # 模式 1：只用 BM25（默认，实测最优）
        if self.mode == "bm25":
            cands = self.bm25.search(query, k=max(self.recall_k, k), route=route)
            return self._maybe_rerank(query, cands, k, route)

        # 模式 2/3：两路召回
        b = self.bm25.search(query, k=self.recall_k, route=route)
        v = self.vec.search(query, k=self.recall_k, route=route)

        b_ids = {d.doc_id for d, _ in b}
        v_ids = {d.doc_id for d, _ in v}
        if b_ids and v_ids:
            self.stats["both"] += 1
        elif b_ids:
            self.stats["bm25_only"] += 1
        elif v_ids:
            self.stats["vec_only"] += 1

        # 模式 3：BM25 不足时才用向量补齐（兜底式混合）
        if self.mode == "hybrid_norerank" and b and len(b) >= k:
            return b[:k]

        fused = rrf_fuse([b, v], k=self.recall_k)
        if self.mode == "hybrid_norerank":
            return fused[:k]
        return self._maybe_rerank(query, fused, k, route)

    def _maybe_rerank(
        self, query: str, cands: list[tuple[Doc, float]], k: int, route: str | None
    ) -> list[tuple[Doc, float]]:
        if self.use_rerank and cands:
            return self.reranker.rerank(query, cands, k=k, route=route)
        return cands[:k]

    # 便于消融实验：只走某一路
    def search_bm25_only(self, query: str, k: int = 3, route: str | None = None):
        return self.bm25.search(query, k=k, route=route)

    def search_vec_only(self, query: str, k: int = 3, route: str | None = None):
        return self.vec.search(query, k=k, route=route)

    def search_rrf_no_rerank(self, query: str, k: int = 3, route: str | None = None):
        b = self.bm25.search(query, k=self.recall_k, route=route)
        v = self.vec.search(query, k=self.recall_k, route=route)
        return rrf_fuse([b, v], k=k)
