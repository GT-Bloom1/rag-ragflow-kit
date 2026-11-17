"""一键建库：创建知识库 → 写入推荐分块配置 → 上传文档 → 写入元数据 → 解析并等待完成。

用法（在项目根目录）：
    python -m ragflow_client.setup_kb
    python -m ragflow_client.setup_kb --name 课程库_cs128 --chunk-size 128 --overlap 0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .client import DONE_STATES, RagflowClient, RagflowError
from .kb_config import KbConfig, legacy_parser_config, v1_parser_config
from .settings import ROOT, load_settings

STATE_FILE = ROOT / "results" / "ragflow_state.json"
SUFFIXES = {".md", ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html"}


def save_state(name: str, dataset_id: str, api_style: str) -> None:
    state = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    state.setdefault("datasets", {})[name] = dataset_id
    state["api_style"] = api_style
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_config(client: RagflowClient, ds: dict, cfg: KbConfig, created_legacy: bool) -> list[str]:
    """写入分块和知识增强配置，返回需要到界面手动设置的项。"""
    if client.api_style == "legacy":
        if not created_legacy:
            client.update_dataset(ds["id"], {"chunk_method": "naive", "parser_config": legacy_parser_config(cfg)})
        print("  已按 0.2x 写法设置：", legacy_parser_config(cfg))
        return []
    current = client.get_dataset(ds["id"]).get("parser_config") or {}
    new_cfg, applied, manual = v1_parser_config(current, cfg)
    if applied:
        try:
            client.update_dataset(ds["id"], {"parse_type": 1, "parser_id": "general", "parser_config": new_cfg})
            print("  已通过 API 设置：", "; ".join(applied))
        except RagflowError as err:
            print(f"  [警告] 写入 parser_config 失败：{err}")
            manual = [a for a in applied] + manual
    return manual


def main() -> None:
    ap = argparse.ArgumentParser(description="创建 RAGFlow 知识库并导入文档")
    ap.add_argument("--docs", default=str(ROOT / "sample_docs"), help="文档目录")
    ap.add_argument("--name", help="知识库名称，默认取 config/.env 中的 DATASET_NAME")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=10, help="重叠比例，单位 %%")
    ap.add_argument("--auto-keywords", type=int, default=3)
    ap.add_argument("--auto-questions", type=int, default=1)
    ap.add_argument("--no-parse", action="store_true", help="只建库和上传，先去界面补全配置再解析")
    args = ap.parse_args()

    st = load_settings()
    name = args.name or st.dataset_name
    cfg = KbConfig(args.chunk_size, args.overlap, args.auto_keywords, args.auto_questions)
    client = RagflowClient(st.base_url, st.api_key)

    print(f"[1/5] 知识库：{name}")
    ds = client.find_dataset(name)
    created_legacy = False
    if ds:
        print(f"  已存在，复用 id={ds['id']}，接口风格={client.detect_style(ds)}")
    else:
        ds = client.create_dataset(name, st.embed_model, legacy_parser_config(cfg))
        created_legacy = client.api_style == "legacy"
        print(f"  已创建 id={ds['id']}，Embedding={st.embed_model}，接口风格={client.api_style}")
    save_state(name, ds["id"], client.api_style)

    print("[2/5] 写入分块与知识增强配置")
    manual = apply_config(client, ds, cfg, created_legacy)

    print("[3/5] 上传文档")
    files = sorted(p for p in Path(args.docs).iterdir() if p.suffix.lower() in SUFFIXES)
    existing = {d["name"]: d for d in client.list_documents(ds["id"])}
    new_files = [p for p in files if p.name not in existing]
    if new_files:
        for doc in client.upload_documents(ds["id"], new_files):
            existing[doc["name"]] = doc
    print(f"  共 {len(files)} 个文件，本次新上传 {len(new_files)} 个")

    print("[4/5] 写入元数据（文件名、更新时间）")
    for p in files:
        doc = existing[p.name]
        mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
        meta = {"file_name": p.name, "update_time": mtime, "course": "CS-AI-301"}
        try:
            client.update_document(ds["id"], doc["id"], {"meta_fields": meta})
        except RagflowError as err:
            print(f"  [警告] {p.name} 元数据写入失败：{err}")

    print("[5/5] 解析文档（向量化 + 自动关键词/问题，CPU 上每个文件约需 1~5 分钟）")
    todo = [existing[p.name]["id"] for p in files
            if str(existing[p.name].get("run", "")).upper() not in DONE_STATES]
    if args.no_parse:
        print("  已按 --no-parse 跳过；界面中补全配置后，重新运行本命令（不带 --no-parse）即可解析")
    elif todo:
        client.parse_documents(ds["id"], todo)
        client.wait_parsed(ds["id"], todo)
    for d in client.list_documents(ds["id"]):
        print(f"  {d.get('name')}: {d.get('chunk_count')} 个 chunk，{d.get('token_count')} tokens")

    if manual:
        print("\n以下配置当前版本的 API 未公开对应字段，请在界面中设置后点击“保存”，再对文档重新解析：")
        print("  知识库 → 配置（Configuration）：" + "；".join(manual))
    print(f"\n完成。知识库 id 已保存到 {STATE_FILE}")


if __name__ == "__main__":
    main()
