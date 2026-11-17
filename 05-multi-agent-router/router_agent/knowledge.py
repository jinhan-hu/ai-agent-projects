"""知识库检索（可解释的简化版）。

真实的 RAG 生产系统用「向量 + BM25 混合检索 + Rerank」。
本项目刻意只用**字符二元组 + IDF 加权**，原因有二：

1. 零依赖、结果可解释——每一分都能说清是哪几个词贡献的，便于教学与测试。
2. 它正好演示了混合检索里 BM25 那一半的思路（词频 × 逆文档频率）。

想升级成生产级：把 `KnowledgeBase.search()` 换成向量库 + BM25 + RRF 融合即可，
上层（专家 Agent）的接口不需要改。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    route: str        # 该文档属于哪一类（install / usage / troubleshoot）
    source: str = ""  # 出处，用于回答里的引用


def _bigrams(text: str) -> list[str]:
    """字符二元组。中文没有空格，用 bigram 比按字切更稳。"""
    clean = "".join(ch for ch in text if ch.isalnum())
    if len(clean) < 2:
        return [clean] if clean else []
    return [clean[i : i + 2] for i in range(len(clean) - 1)]


class KnowledgeBase:
    """极简检索：bigram 命中 + IDF 加权 + 长度归一。"""

    def __init__(self, docs: list[Doc]) -> None:
        self.docs = docs
        self._tokens: dict[str, Counter] = {
            d.doc_id: Counter(_bigrams(d.title + " " + d.text)) for d in docs
        }
        self._df: Counter = Counter()
        for counter in self._tokens.values():
            for token in counter:
                self._df[token] += 1
        self._n = len(docs)

    # ---------- 检索 ----------
    def search(self, query: str, k: int = 3, route: str | None = None) -> list[tuple[Doc, float]]:
        """返回 [(文档, 得分)]。

        route 不为空时只在同类文档里检索——这是「路由先于检索」的直接体现：
        分对类，检索空间就从 N 篇缩到该类的几篇。
        """
        pool = [d for d in self.docs if route is None or d.route == route]
        if not pool:
            return []

        q_tokens = _bigrams(query)
        if not q_tokens:
            return []

        scored: list[tuple[Doc, float]] = []
        for doc in pool:
            tf = self._tokens[doc.doc_id]
            score = 0.0
            for token in q_tokens:
                if token not in tf:
                    continue
                # IDF：越少见的词越有区分度
                idf = math.log((self._n + 1) / (self._df[token] + 1)) + 1.0
                score += (1 + math.log(tf[token])) * idf
            if score <= 0:
                continue
            # 长度归一，避免长文档靠词多取胜
            score /= math.sqrt(sum(tf.values()) or 1)
            scored.append((doc, round(score, 4)))

        scored.sort(key=lambda x: (-x[1], x[0].doc_id))
        return scored[:k]

    def search_text(self, query: str, k: int = 3) -> str:
        """给专家 Agent 用的检索文本（带出处，方便生成引用）。"""
        hits = self.search(query, k=k)
        if not hits:
            return ""
        blocks = []
        for doc, score in hits:
            blocks.append(f"[{doc.source or doc.doc_id}] {doc.title}\n{doc.text}")
        return "\n\n".join(blocks)
