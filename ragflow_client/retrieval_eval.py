"""用标注问题集评测 RAGFlow 检索效果，比较不同知识库（分块配置）和不同关键词权重。

用法：
    python -m ragflow_client.retrieval_eval                         # 评测 DATASET_NAME
    python -m ragflow_client.retrieval_eval --datasets 课程库_cs128,课程库_cs256,课程库_cs512
    python -m ragflow_client.retrieval_eval --keyword-weights 0.7,0.3,0 --show q20
"""
from __future__ import annotations

import argparse
import csv

from .client import RagflowClient
from .metrics import evidence_of, first_hit_rank, load_questions, summarize
from .settings import ROOT, load_settings


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGFlow 检索评测")
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--datasets", help="知识库名称，逗号分隔；默认 DATASET_NAME")
    ap.add_argument("--keyword-weights", default="0.7,0.3,0",
                    help="Keyword similarity weight 列表；0.7=官方默认，0.3=课堂推荐（向量 0.7），0=纯向量")
    ap.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--show", help="打印某个问题 id 的检索结果，便于课堂讲解")
    args = ap.parse_args()

    st = load_settings()
    client = RagflowClient(st.base_url, st.api_key)
    questions = load_questions(args.questions)
    names = [n.strip() for n in (args.datasets or st.dataset_name).split(",") if n.strip()]
    weights = [float(w) for w in args.keyword_weights.split(",")]

    summary_rows, detail_rows = [], []
    for name in names:
        ds = client.find_dataset(name)
        if not ds:
            print(f"[跳过] 找不到知识库 {name}")
            continue
        for kw in weights:
            ranks = []
            for q in questions:
                res = client.retrieve(q["question"], [ds["id"]], keyword_weight=kw,
                                      similarity_threshold=args.threshold, top_n=args.top_n)
                chunks = res.get("chunks", [])
                rank = first_hit_rank([c.get("content", "") for c in chunks], evidence_of(q))
                ranks.append(rank)
                detail_rows.append({"dataset": name, "keyword_weight": kw, "id": q["id"],
                                    "type": q["type"], "question": q["question"], "rank": rank or ""})
                if args.show == q["id"]:
                    print(f"\n--- {name} | keyword={kw} | {q['question']} ---")
                    for i, c in enumerate(chunks, 1):
                        print(f"#{i} sim={c.get('similarity') or 0:.3f} vec={c.get('vector_similarity') or 0:.3f} "
                              f"kw={c.get('term_similarity') or 0:.3f} [{c.get('document_keyword', '')}]")
                        print("   " + c.get("content", "").replace("\n", " ")[:120])
            for qtype in ("all", "semantic", "keyword"):
                sel = [r for r, q in zip(ranks, questions) if qtype == "all" or q["type"] == qtype]
                summary_rows.append({"dataset": name, "keyword_weight": kw, "vector_weight": round(1 - kw, 2),
                                     "subset": qtype, "n": len(sel), **summarize(sel)})

    if not summary_rows:
        return
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    for fname, rows in (("ragflow_eval_summary.csv", summary_rows), ("ragflow_eval_detail.csv", detail_rows)):
        with open(out_dir / fname, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"\n{'知识库':<16}{'关键词权重':>8}{'子集':>10}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}")
    for r in summary_rows:
        print(f"{r['dataset']:<16}{r['keyword_weight']:>8.1f}{r['subset']:>10}"
              f"{r['hit@1']:>8.2f}{r['hit@3']:>8.2f}{r['hit@5']:>8.2f}{r['mrr']:>8.3f}")
    print(f"\n结果已写入 {out_dir / 'ragflow_eval_summary.csv'} 和 ragflow_eval_detail.csv")


if __name__ == "__main__":
    main()
