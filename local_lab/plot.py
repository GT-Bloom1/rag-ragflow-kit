"""把三组实验画成一张三联图。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_results(rows: list[dict], path: Path, backend: str, n_questions: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    panels = [
        ("A_chunk_size", "chunk_size", "Chunk Size（tokens），Overlap 10%，向量权重 0.7"),
        ("B_overlap", "overlap", "Overlap Ratio（%），Chunk Size 128，向量权重 0.7"),
        ("C_vector_weight", "vector_weight", "向量权重（关键词权重 = 1 − 向量权重），Chunk 256"),
    ]
    for ax, (exp, xkey, title) in zip(axes, panels):
        sub = [r for r in rows if r["experiment"] == exp]
        xs = [str(r[xkey]) for r in sub]
        ax.plot(xs, [r["hit@1"] for r in sub], "o-", label="Hit@1")
        ax.plot(xs, [r["hit@5"] for r in sub], "s-", label="Hit@5")
        ax.plot(xs, [r["mrr"] for r in sub], "^-", label="MRR")
        if exp == "C_vector_weight":
            ax.plot(xs, [r["mrr_semantic"] for r in sub], "--", color="gray", label="MRR（语义类问题）")
            ax.plot(xs, [r["mrr_keyword"] for r in sub], ":", color="black", label="MRR（术语/编号类问题）")
        else:
            ax.plot(xs, [r["evidence_split_rate"] for r in sub], "x--", color="tab:red", label="证据被切断比例")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    fig.suptitle(f"检索参数对比（{n_questions} 个标注问题，Top-5，Embedding：{backend}）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
