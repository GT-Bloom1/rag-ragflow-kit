"""把“推荐配置”翻译成两种版本的 parser_config。"""
from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass
class KbConfig:
    chunk_size: int = 256        # Chunk Size（tokens）
    overlap_percent: int = 10    # Overlap Ratio（%）
    auto_keywords: int = 3       # 每块自动关键词数，0 为关闭
    auto_questions: int = 1      # 每块自动问题数，0 为关闭
    layout_recognize: str = "DeepDOC"


def legacy_parser_config(cfg: KbConfig) -> dict:
    """RAGFlow 0.2x 的扁平写法（chunk_method = naive）。分隔符不传，沿用服务端默认值。"""
    return {
        "chunk_token_num": cfg.chunk_size,
        "overlapped_percent": cfg.overlap_percent,
        "auto_keywords": cfg.auto_keywords,
        "auto_questions": cfg.auto_questions,
        "layout_recognize": cfg.layout_recognize,
    }


def v1_parser_config(current: dict, cfg: KbConfig) -> tuple[dict, list[str], list[str]]:
    """RAGFlow 1.0 的组件化写法：在服务端返回的 parser_config 上原位修改。

    1.0 文档只公开了 GeneralChunker 组件的 chunk_token_size / delimiters / parent_child，
    其余键名随版本变化，所以这里只修改“已经存在且名字明确”的键，其他项交给界面设置。
    返回 (新配置, 已通过 API 设置的项, 需要在界面手动设置的项)。
    """
    new = copy.deepcopy(current)
    applied: list[str] = []
    chunkers = [k for k in new if k.split(":")[0] == "GeneralChunker" and isinstance(new[k], dict)]
    overlap_done = False
    for key in chunkers:
        comp = new[key]
        comp["chunk_token_size"] = cfg.chunk_size
        applied.append(f"{key}.chunk_token_size = {cfg.chunk_size}")
        for field in list(comp):
            name = field.lower()
            if "overlap" not in name:
                continue
            # 按字段名判断单位：含 percent 写百分数，含 ratio 写 0~1 小数；判断不了就留给界面
            if "percent" in name:
                comp[field] = cfg.overlap_percent
            elif "ratio" in name:
                comp[field] = cfg.overlap_percent / 100
            else:
                continue
            applied.append(f"{key}.{field} = {comp[field]}")
            overlap_done = True

    manual: list[str] = []
    if not chunkers:
        manual.append(f"Chunk Size = {cfg.chunk_size}（未在 parser_config 中找到 GeneralChunker 组件）")
    if not overlap_done:
        manual.append(f"Overlap Ratio = {cfg.overlap_percent}%")
    extract_done = {"auto_keywords": False, "auto_questions": False}
    wanted = {"auto_keywords": cfg.auto_keywords, "auto_questions": cfg.auto_questions}
    for comp in new.values():
        if isinstance(comp, dict):
            for field in extract_done:
                if field in comp:
                    comp[field] = wanted[field]
                    extract_done[field] = True
    applied += [f"{f} = {wanted[f]}" for f, ok in extract_done.items() if ok]
    labels = {"auto_keywords": "Auto Keywords", "auto_questions": "Auto-question"}
    manual += [f"{labels[f]} = {wanted[f]}" for f, ok in extract_done.items() if not ok]
    return new, applied, manual
