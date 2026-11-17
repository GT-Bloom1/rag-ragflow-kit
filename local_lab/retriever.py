"""混合检索：向量余弦相似度 + 关键词相似度，按权重加权。

最终得分 = 向量权重 × 余弦相似度 + (1 − 向量权重) × 关键词相似度
两项都在 0~1 之间。RAGFlow 内部实现更复杂（同义词、细粒度分词、字段加权），这里保留核心思路，
用来在课堂上观察权重的作用。
"""
from __future__ import annotations

import math
import re
from collections import Counter

import jieba
import numpy as np

jieba.setLogLevel(60)
_PUNCT = re.compile(r"^[\W_]+$")
STOPWORDS = set("的 了 是 在 有 和 与 及 或 吗 呢 吧 啊 什么 怎么 怎样 如何 为什么 哪些 哪个 多少 几 个 这 那 "
                "一个 可以 能 会 要 就 也 都 而 并 被 把 对 从 到 用 时 中 上 下 后 前 么 之 其 该".split())


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in jieba.lcut_for_search(text)
            if t.strip() and not _PUNCT.match(t) and t not in STOPWORDS]


class TermSimilarity:
    """关键词相似度：问题中的词有多少（按 IDF 加权）出现在片段里，取值 0~1。

    思路与 RAGFlow 的 term similarity 一致：罕见词（术语、编号）权重大，常见词权重小。
    """

    def __init__(self, docs: list[list[str]]):
        self.sets = [set(d) for d in docs]
        n = len(docs)
        df = Counter(t for d in self.sets for t in d)
        self.default_idf = math.log(1 + (n + 0.5) / 0.5)  # 语料中没出现过的词
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        q = set(query)
        weights = {t: self.idf.get(t, self.default_idf) for t in q}
        total = sum(weights.values())
        if not total:
            return np.zeros(len(self.sets))
        return np.array([sum(w for t, w in weights.items() if t in d) / total for d in self.sets])


class HybridRetriever:
    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = embedder.embed([c.text for c in chunks])
        self.terms = TermSimilarity([tokenize(c.text) for c in chunks])

    def search(self, question: str, vector_weight: float = 0.7, top_n: int = 5,
               threshold: float = 0.2) -> list[tuple[float, float, float, object]]:
        """返回 [(混合分, 向量分, 关键词分, Chunk)]，按混合分降序，已过滤低于阈值的片段。"""
        qv = self.embedder.embed([question])[0]
        vec = self.vectors @ qv
        kw = self.terms.scores(tokenize(question))
        score = vector_weight * vec + (1 - vector_weight) * kw
        order = np.argsort(-score)
        hits = [(float(score[i]), float(vec[i]), float(kw[i]), self.chunks[i]) for i in order[:top_n]]
        return [h for h in hits if h[0] >= threshold]
