"""课堂演示：查看某个配置下的分块结果，或某个问题的检索结果（含向量分、关键词分）。

    python -m local_lab.show chunks --doc 04_检索增强生成RAG.md --chunk-size 128 --overlap 15
    python -m local_lab.show search "如何防止模型编造答案" --vector-weight 0.7
    python -m local_lab.show search "如何防止模型编造答案" --vector-weight 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .chunker import chunk_text
from .embedder import get_embedder
from .retriever import HybridRetriever
from .run_experiments import ROOT, build_chunks, load_docs


def main() -> None:
    ap = argparse.ArgumentParser(description="查看分块与检索结果")
    ap.add_argument("mode", choices=["chunks", "search"])
    ap.add_argument("question", nargs="?", default="如何防止模型编造答案？")
    ap.add_argument("--doc", default="04_检索增强生成RAG.md")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--vector-weight", type=float, default=0.7)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--backend", default="auto", choices=["auto", "ollama", "local"])
    args = ap.parse_args()

    if args.mode == "chunks":
        text = (ROOT / "sample_docs" / args.doc).read_text(encoding="utf-8")
        chunks = chunk_text(text, args.doc, args.chunk_size, args.overlap)
        print(f"{args.doc}：Chunk Size={args.chunk_size}，Overlap={args.overlap}% → {len(chunks)} 个片段\n")
        for i, c in enumerate(chunks, 1):
            print(f"── 片段 {i}（{c.tokens} tokens）──\n{c.text}\n")
        return

    docs = load_docs(Path(ROOT / "sample_docs"))
    retriever = HybridRetriever(build_chunks(docs, args.chunk_size, args.overlap), get_embedder(args.backend))
    hits = retriever.search(args.question, args.vector_weight, args.top_n, args.threshold)
    print(f"问题：{args.question}\n向量权重 {args.vector_weight}，关键词权重 {1 - args.vector_weight:.1f}，"
          f"阈值 {args.threshold}\n")
    for i, (score, vec, kw, c) in enumerate(hits, 1):
        print(f"#{i} 综合 {score:.3f} = {args.vector_weight}×向量 {vec:.3f} + {1 - args.vector_weight:.1f}×关键词 {kw:.3f}"
              f"  [{c.doc}]\n   {c.text[:110].replace(chr(10), ' ')}…\n")


if __name__ == "__main__":
    main()
