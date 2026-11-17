"""本地实验模块的单元测试（不需要 Embedding 模型）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from local_lab.chunker import chunk_text, num_tokens, split_sentences  # noqa: E402
from local_lab.retriever import HybridRetriever, TermSimilarity, tokenize  # noqa: E402

TEXT = (ROOT / "sample_docs" / "04_检索增强生成RAG.md").read_text(encoding="utf-8")


def test_split_keeps_all_text():
    assert "".join(split_sentences(TEXT)).replace("\n", "") == TEXT.replace("\n", "")


def test_chunk_size_respected_and_text_covered():
    chunks = chunk_text(TEXT, "d", chunk_size=128, overlap_percent=0)
    assert max(c.tokens for c in chunks) <= 128  # 本文没有超过 128 tokens 的单句
    joined = "".join(c.text for c in chunks).replace("\n", "")
    assert joined == TEXT.replace("\n", "").strip()  # 无重叠时拼回原文


def test_overlap_repeats_previous_tail():
    chunks = chunk_text(TEXT, "d", chunk_size=128, overlap_percent=15)
    plain = chunk_text(TEXT, "d", chunk_size=128, overlap_percent=0)
    assert len(chunks) > len(plain)
    assert sum(c.text[:5] in chunks[i].text for i, c in enumerate(chunks[1:])) >= len(chunks) // 2


def test_term_similarity_prefers_rare_terms():
    docs = [tokenize("课程编号 CS-AI-301 的实验课在 B-305 机房"), tokenize("机器学习是人工智能的分支")]
    s = TermSimilarity(docs).scores(tokenize("B-305 机房在哪"))
    assert s[0] > 0.5 and s[1] == 0


class FakeEmbedder:
    name = "fake"

    def embed(self, texts):
        return np.array([[1.0, 0.0] if "幻觉" in t else [0.0, 1.0] for t in texts])


def test_hybrid_weight_extremes():
    from local_lab.chunker import Chunk
    chunks = [Chunk("a", "减少幻觉的方法", 5), Chunk("b", "模型编造答案的定义", 6)]
    r = HybridRetriever(chunks, FakeEmbedder())
    assert r.search("幻觉 编造答案", vector_weight=1.0, threshold=0)[0][3].doc == "a"
    assert r.search("编造答案", vector_weight=0.0, threshold=0)[0][3].doc == "b"
    assert num_tokens("检索增强生成") > 0
