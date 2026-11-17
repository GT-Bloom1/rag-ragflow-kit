"""检索评测指标：命中判定、Hit@K、MRR。RAGFlow 评测和本地实验共用。"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """去掉空白，避免解析器插入的换行/空格导致证据句匹配失败。"""
    return _SPACE.sub("", text or "")


def load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evidence_of(question: dict) -> list[str]:
    """答案依据。evidence_all 表示答案跨越多句，必须同时出现在同一个片段里才算命中。"""
    return question.get("evidence_all") or [question["evidence"]]


def first_hit_rank(chunks: list[str], evidence: str | list[str]) -> int | None:
    """所有证据句都完整出现在同一个检索结果中，返回它的名次（从 1 开始）；否则返回 None。"""
    items = [normalize(e) for e in ([evidence] if isinstance(evidence, str) else evidence)]
    for rank, text in enumerate(chunks, start=1):
        t = normalize(text)
        if all(e in t for e in items):
            return rank
    return None


def summarize(ranks: list[int | None], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {**{f"hit@{k}": 0.0 for k in ks}, "mrr": 0.0}
    out = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out
