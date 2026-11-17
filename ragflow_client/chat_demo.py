"""创建（或复用）聊天助手，并通过 OpenAI 兼容接口提问。

用法：
    python -m ragflow_client.chat_demo "如何防止模型编造答案？"
    python -m ragflow_client.chat_demo            # 进入交互模式，输入 exit 退出
"""
from __future__ import annotations

import argparse

from .client import RagflowClient
from .settings import load_settings

SYSTEM_PROMPT = (
    "你是《人工智能导论与实践》课程的助教。请只依据下面的知识库内容回答问题，回答要简洁，"
    "并尽量引用原文要点。如果知识库中没有与问题相关的内容，请直接回答“知识库中没有找到相关内容”，"
    "不要编造。\n以下是知识库：\n{knowledge}\n以上是知识库。"
)
EMPTY_RESPONSE = "知识库中没有找到相关内容。"


def print_answer(resp: dict) -> None:
    msg = resp["choices"][0]["message"]
    print("\n助教：", msg.get("content", "").strip())
    refs = msg.get("reference")
    chunks = refs.get("chunks", []) if isinstance(refs, dict) else (refs or [])
    for i, ref in enumerate(chunks, 1):
        if isinstance(ref, dict):
            doc = ref.get("document_name") or ref.get("doc_name") or ref.get("document_keyword", "")
            text = (ref.get("content") or "").replace("\n", " ")[:80]
            print(f"  [引用{i}] {doc}：{text}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGFlow 聊天助手示例")
    ap.add_argument("question", nargs="*", help="要问的问题；不填则进入交互模式")
    ap.add_argument("--keyword-weight", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--top-n", type=int, default=6)
    args = ap.parse_args()

    st = load_settings()
    client = RagflowClient(st.base_url, st.api_key)
    ds = client.find_dataset(st.dataset_name)
    if not ds:
        raise SystemExit(f"找不到知识库 {st.dataset_name}，请先运行 python -m ragflow_client.setup_kb")
    client.detect_style(ds)

    chat = client.find_chat(st.chat_name)
    if chat:
        print(f"复用聊天助手 {st.chat_name}（id={chat['id']}）")
    else:
        chat = client.create_chat(st.chat_name, [ds["id"]], st.chat_model, SYSTEM_PROMPT, EMPTY_RESPONSE,
                                  keyword_weight=args.keyword_weight, similarity_threshold=args.threshold,
                                  top_n=args.top_n)
        print(f"已创建聊天助手 {st.chat_name}（id={chat['id']}，模型={st.chat_model}）")

    history: list[dict] = []
    pending = [" ".join(args.question)] if args.question else None  # None 表示交互模式
    while True:
        if pending is None:
            q = input("\n你：").strip()
        elif pending:
            q = pending.pop(0)
        else:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": q})
        resp = client.ask(chat["id"], history)
        print_answer(resp)
        history.append({"role": "assistant", "content": resp["choices"][0]["message"].get("content", "")})


if __name__ == "__main__":
    main()
