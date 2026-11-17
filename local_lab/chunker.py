"""通用分块器，按 RAGFlow General 分块的思路实现，便于在本地观察参数影响。

1. 按分隔符（换行、句号、问号等）切成句子；
2. 依次把句子拼进当前块，token 数超过 chunk_size × (1 − overlap) 时开始新块；
3. 新块开头复制上一块末尾 overlap 比例的文字，形成重叠。
token 数用 tiktoken 的 cl100k_base 计算，与 RAGFlow 的计数方式一致。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

DEFAULT_DELIMITERS = "\n!?;。；！？"


@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding("cl100k_base")


def num_tokens(text: str) -> int:
    return len(_encoder().encode(text))


@dataclass
class Chunk:
    doc: str
    text: str
    tokens: int


def split_sentences(text: str, delimiters: str = DEFAULT_DELIMITERS) -> list[str]:
    pattern = "([" + re.escape(delimiters) + "])"
    pieces = re.split(pattern, text)
    sentences, buf = [], ""
    for piece in pieces:
        buf += piece
        if piece and piece in delimiters:  # 分隔符留在句尾
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)
    return [s for s in sentences if s.strip()]


def chunk_text(text: str, doc: str, chunk_size: int = 256, overlap_percent: int = 10,
               delimiters: str = DEFAULT_DELIMITERS) -> list[Chunk]:
    if not 0 <= overlap_percent < 100:
        raise ValueError("overlap_percent 必须在 0~99 之间")
    limit = chunk_size * (100 - overlap_percent) / 100
    texts: list[str] = [""]
    counts: list[int] = [0]
    for sent in split_sentences(text, delimiters):
        n = num_tokens(sent)
        if counts[-1] > 0 and counts[-1] + n > limit:
            prev = texts[-1]
            tail = prev[int(len(prev) * (100 - overlap_percent) / 100):] if overlap_percent else ""
            tail = tail.lstrip(delimiters + " ，、")  # 不让新片段以标点开头
            texts.append(tail)
            counts.append(num_tokens(tail))
        texts[-1] += sent
        counts[-1] += n
    return [Chunk(doc, t.strip(), num_tokens(t)) for t in texts if t.strip()]
