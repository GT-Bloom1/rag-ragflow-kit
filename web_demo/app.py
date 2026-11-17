"""CPU 版 RAG 网页演示：不依赖 Docker 和 RAGFlow，在一台普通电脑上演示 知识库 → 检索测试 → 聊天 的完整流程。

Embedding：bge-large-zh-v1.5（优先用 Ollama，否则用 LOCAL_EMBED_MODEL 指定的本地目录）
对话模型：Qwen2.5 GGUF（llama.cpp，纯 CPU），模型文件由 LLM_GGUF 指定；也可以用 OLLAMA_CHAT_MODEL 改走 Ollama
检索：与 local_lab 相同的混合检索（向量余弦 × 权重 + 关键词相似度 × (1 − 权重)）

    LOCAL_EMBED_MODEL=/path/bge-large-zh-v1.5 LLM_GGUF=/path/qwen2.5-3b-instruct-q4_k_m.gguf \
        python -m web_demo.app --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from local_lab.embedder import get_embedder
from local_lab.retriever import HybridRetriever
from local_lab.run_experiments import ROOT, build_chunks, load_docs

HERE = Path(__file__).resolve().parent
UPLOAD_DIR = HERE / "uploads"
SYSTEM_PROMPT = ("你是《人工智能导论与实践》课程的助教。请只依据下面的知识库内容回答问题，回答要简洁，"
                 "并尽量引用原文要点。每个要点后面用方括号写出依据的片段编号，例如 [1]、[2]。如果知识库中没有与问题相关的内容，"
                 "请直接回答“知识库中没有找到相关内容”，不要编造。\n以下是知识库：\n{knowledge}\n以上是知识库。")
EMPTY_RESPONSE = "知识库中没有找到相关内容。"


class KnowledgeBase:
    """文档 + 分块参数 + 检索器。改分块参数或上传文件后重建索引（向量有磁盘缓存，重建很快）。"""

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunk_size, self.overlap = 256, 10
        self.lock = threading.Lock()
        self.rebuild()

    def documents(self) -> dict[str, str]:
        docs = load_docs(ROOT / "sample_docs")
        if UPLOAD_DIR.exists():
            docs.update({p.name: p.read_text(encoding="utf-8") for p in sorted(UPLOAD_DIR.iterdir())
                         if p.suffix in (".md", ".txt")})
        return docs

    def rebuild(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        with self.lock:
            self.chunk_size = chunk_size or self.chunk_size
            self.overlap = self.overlap if overlap is None else overlap
            start = time.time()
            self.docs = self.documents()
            self.retriever = HybridRetriever(build_chunks(self.docs, self.chunk_size, self.overlap), self.embedder)
            self.build_seconds = round(time.time() - start, 1)


class ChatModel:
    def __init__(self):
        self.ollama_model = os.environ.get("OLLAMA_CHAT_MODEL")
        self.url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.lock = threading.Lock()
        if self.ollama_model:
            self.name = f"{self.ollama_model}（Ollama）"
            return
        gguf = os.environ.get("LLM_GGUF")
        if not gguf or not Path(gguf).exists():
            raise SystemExit("请用 LLM_GGUF 指定 GGUF 模型文件，或用 OLLAMA_CHAT_MODEL 指定 Ollama 中的对话模型")
        from llama_cpp import Llama
        self.llm = Llama(model_path=gguf, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
        self.name = f"{Path(gguf).stem}（llama.cpp，CPU）"

    def stream(self, messages: list[dict], temperature: float):
        with self.lock:  # 单进程 CPU 推理，一次只回答一个问题
            if self.ollama_model:
                with requests.post(f"{self.url}/api/chat", stream=True, timeout=600, json={
                        "model": self.ollama_model, "messages": messages, "stream": True,
                        "options": {"temperature": temperature}}) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            yield json.loads(line).get("message", {}).get("content", "")
                return
            for part in self.llm.create_chat_completion(messages=messages, temperature=temperature,
                                                        max_tokens=512, stream=True):
                yield part["choices"][0]["delta"].get("content", "")


class SearchReq(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    vector_weight: float = Field(0.7, ge=0, le=1)
    top_n: int = Field(6, ge=1, le=20)
    threshold: float = Field(0.2, ge=0, le=1)


class ChatReq(SearchReq):
    temperature: float = Field(0.1, ge=0, le=1.5)


class ConfigReq(BaseModel):
    chunk_size: int = Field(ge=32, le=2048)
    overlap: int = Field(ge=0, le=50)


def hit_json(i: int, hit) -> dict:
    score, vec, kw, chunk = hit
    return {"id": i, "doc": chunk.doc, "text": chunk.text, "tokens": chunk.tokens,
            "score": round(score, 4), "vector": round(vec, 4), "keyword": round(kw, 4)}


def create_app(kb: KnowledgeBase, chat: ChatModel) -> FastAPI:
    app = FastAPI(title="RAG 课堂演示")

    @app.get("/")
    def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/api/status")
    def status():
        return {"embedding": kb.embedder.name, "chat_model": chat.name, "chunk_size": kb.chunk_size,
                "overlap": kb.overlap, "documents": len(kb.docs), "chunks": len(kb.retriever.chunks),
                "build_seconds": kb.build_seconds}

    @app.get("/api/documents")
    def documents():
        counts: dict[str, int] = {}
        for c in kb.retriever.chunks:
            counts[c.doc] = counts.get(c.doc, 0) + 1
        return [{"name": n, "chars": len(t), "chunks": counts.get(n, 0),
                 "source": "上传" if (UPLOAD_DIR / n).exists() else "示例资料"} for n, t in kb.docs.items()]

    @app.get("/api/chunks")
    def chunks(doc: str):
        return [{"text": c.text, "tokens": c.tokens} for c in kb.retriever.chunks if c.doc == doc]

    @app.post("/api/config")
    def config(req: ConfigReq):
        kb.rebuild(req.chunk_size, req.overlap)
        return status()

    @app.post("/api/upload")
    async def upload(file: UploadFile):
        name = Path(file.filename or "").name
        if not re.fullmatch(r"[\w一-鿿 .()（）-]{1,80}\.(md|txt)", name):
            raise HTTPException(400, "只支持 .md / .txt 文件，文件名不能含特殊字符")
        data = await file.read()
        if len(data) > 2_000_000:
            raise HTTPException(400, "文件不能超过 2 MB")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "请使用 UTF-8 编码的文本文件")
        UPLOAD_DIR.mkdir(exist_ok=True)
        (UPLOAD_DIR / name).write_text(text, encoding="utf-8")
        kb.rebuild()
        return status()

    @app.post("/api/search")
    def search(req: SearchReq):
        start = time.time()
        hits = kb.retriever.search(req.question, req.vector_weight, req.top_n, req.threshold)
        return {"hits": [hit_json(i, h) for i, h in enumerate(hits, 1)], "seconds": round(time.time() - start, 2)}

    @app.post("/api/chat")
    def chat_api(req: ChatReq):
        hits = [hit_json(i, h) for i, h in enumerate(
            kb.retriever.search(req.question, req.vector_weight, req.top_n, req.threshold), 1)]

        def events():
            yield f"data: {json.dumps({'type': 'refs', 'hits': hits}, ensure_ascii=False)}\n\n"
            if not hits:
                yield f"data: {json.dumps({'type': 'token', 'text': EMPTY_RESPONSE}, ensure_ascii=False)}\n\n"
            else:
                knowledge = "\n\n".join(f"[{h['id']}] 来源：{h['doc']}\n{h['text']}" for h in hits)
                messages = [{"role": "system", "content": SYSTEM_PROMPT.format(knowledge=knowledge)},
                            {"role": "user", "content": req.question}]
                start, n = time.time(), 0
                for text in chat.stream(messages, req.temperature):
                    n += 1
                    yield f"data: {json.dumps({'type': 'token', 'text': text}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'stats', 'seconds': round(time.time() - start, 1), 'pieces': n})}\n\n"
            yield 'data: {"type": "done"}\n\n'
        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU 版 RAG 网页演示")
    ap.add_argument("--host", default="127.0.0.1", help="只在本机访问用 127.0.0.1；让同一局域网的学生访问用 0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", default="auto", choices=["auto", "ollama", "local"], help="Embedding 后端")
    args = ap.parse_args()
    print("加载 Embedding 模型并建立索引……")
    kb = KnowledgeBase(get_embedder(args.backend))
    print(f"索引完成：{len(kb.docs)} 个文档，{len(kb.retriever.chunks)} 个片段，用时 {kb.build_seconds} 秒")
    print("加载对话模型……")
    chat = ChatModel()
    print(f"打开浏览器访问 http://localhost:{args.port}")
    uvicorn.run(create_app(kb, chat), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
