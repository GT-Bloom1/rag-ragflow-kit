"""课堂参数对比实验：Chunk Size、Overlap Ratio、Embedding 权重对检索效果的影响。

不依赖 RAGFlow，只需要 Ollama 中的 bge-large-zh-v1.5（或本地模型目录）。在项目根目录运行：
    python -m local_lab.run_experiments
    python -m local_lab.run_experiments --backend local     # LOCAL_EMBED_MODEL=模型目录
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ragflow_client.metrics import evidence_of, first_hit_rank, load_questions, summarize

from .chunker import chunk_text
from .embedder import get_embedder
from .retriever import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = {
    "A_chunk_size": [dict(chunk_size=s, overlap=10, vector_weight=0.7) for s in (64, 128, 256, 512)],
    "B_overlap": [dict(chunk_size=128, overlap=o, vector_weight=0.7) for o in (0, 10, 15, 30)],
    "C_vector_weight": [dict(chunk_size=256, overlap=10, vector_weight=w) for w in (0.0, 0.3, 0.5, 0.7, 1.0)],
}


def load_docs(folder: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))}


def build_chunks(docs: dict[str, str], chunk_size: int, overlap: int):
    return [c for name, text in docs.items() for c in chunk_text(text, name, chunk_size, overlap)]


def split_rate(chunks, questions) -> float:
    """答案依据被分块边界切断（没有任何一个片段同时包含全部证据句）的问题比例。"""
    texts = [c.text for c in chunks]
    return sum(first_hit_rank(texts, evidence_of(q)) is None for q in questions) / len(questions)


def evaluate(retriever, questions, vector_weight, top_n, threshold) -> dict:
    ranks, ctx_tokens, per_type = [], [], {"semantic": [], "keyword": []}
    for q in questions:
        hits = retriever.search(q["question"], vector_weight, top_n, threshold)
        rank = first_hit_rank([h[3].text for h in hits], evidence_of(q))
        ranks.append(rank)
        per_type[q["type"]].append(rank)
        ctx_tokens.append(sum(h[3].tokens for h in hits))
    row = summarize(ranks)
    row["mrr_semantic"] = summarize(per_type["semantic"])["mrr"]
    row["mrr_keyword"] = summarize(per_type["keyword"])["mrr"]
    row["context_tokens"] = sum(ctx_tokens) / len(ctx_tokens)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="本地 RAG 参数对比实验")
    ap.add_argument("--backend", default="auto", choices=["auto", "ollama", "local"])
    ap.add_argument("--docs", default=str(ROOT / "sample_docs"))
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    embedder = get_embedder(args.backend)
    print(f"Embedding 后端：{embedder.name}")
    docs, questions = load_docs(Path(args.docs)), load_questions(args.questions)
    retrievers: dict[tuple[int, int], HybridRetriever] = {}
    rows = []
    for exp, settings in EXPERIMENTS.items():
        for s in settings:
            key = (s["chunk_size"], s["overlap"])
            if key not in retrievers:
                retrievers[key] = HybridRetriever(build_chunks(docs, *key), embedder)
            r = retrievers[key]
            row = {"experiment": exp, **s, "n_chunks": len(r.chunks),
                   "avg_chunk_tokens": sum(c.tokens for c in r.chunks) / len(r.chunks),
                   "evidence_split_rate": split_rate(r.chunks, questions),
                   **evaluate(r, questions, s["vector_weight"], args.top_n, args.threshold)}
            rows.append(row)
            print(f"{exp:<16} size={s['chunk_size']:<4} overlap={s['overlap']:<3}% vec_w={s['vector_weight']:<4}"
                  f" chunks={row['n_chunks']:<3} split={row['evidence_split_rate']:.2f}"
                  f" hit@1={row['hit@1']:.2f} hit@5={row['hit@5']:.2f} MRR={row['mrr']:.3f}"
                  f" ctx={row['context_tokens']:.0f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "local_experiments.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from .plot import plot_results  # matplotlib 只在画图时导入
    plot_results(rows, out / "local_experiments.png", embedder.name, len(questions))
    print(f"\n结果：{out / 'local_experiments.csv'}，图：{out / 'local_experiments.png'}")


if __name__ == "__main__":
    main()
