"""网页演示后端的接口测试（用假的 Embedding 和对话模型，不需要下载模型）。"""
import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import web_demo.app as demo  # noqa: E402


class FakeEmbedder:
    name = "fake"

    def embed(self, texts):
        return np.array([[1.0, 0.0] if "幻觉" in t else [0.0, 1.0] for t in texts], dtype=np.float32)


class FakeChat:
    name = "fake-chat"

    def __init__(self):
        self.messages = None

    def stream(self, messages, temperature):
        self.messages = messages
        yield from ["依据资料，", "答案见 [1]。"]


def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "UPLOAD_DIR", tmp_path / "uploads")
    chat = FakeChat()
    return TestClient(demo.create_app(demo.KnowledgeBase(FakeEmbedder()), chat)), chat


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_status_documents_and_rechunk(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    s = client.get("/api/status").json()
    assert s["documents"] == 5 and s["chunk_size"] == 256
    docs = client.get("/api/documents").json()
    assert sum(d["chunks"] for d in docs) == s["chunks"]
    small = client.post("/api/config", json={"chunk_size": 64, "overlap": 0}).json()
    assert small["chunks"] > s["chunks"]
    assert client.post("/api/config", json={"chunk_size": 8, "overlap": 0}).status_code == 422


def test_search_weights_and_chat_stream(tmp_path, monkeypatch):
    client, chat = make_client(tmp_path, monkeypatch)
    body = {"question": "模型幻觉是什么", "vector_weight": 1.0, "top_n": 3, "threshold": 0}
    hits = client.post("/api/search", json=body).json()["hits"]
    assert len(hits) == 3 and all(h["score"] == h["vector"] for h in hits)
    evs = events(client.post("/api/chat", json=body))
    assert evs[0]["type"] == "refs" and len(evs[0]["hits"]) == 3
    assert "".join(e["text"] for e in evs if e["type"] == "token") == "依据资料，答案见 [1]。"
    assert "[1] 来源：" in chat.messages[0]["content"]  # 片段编号写进了系统提示词


def test_chat_without_hits_returns_empty_response(tmp_path, monkeypatch):
    client, chat = make_client(tmp_path, monkeypatch)
    evs = events(client.post("/api/chat", json={"question": "完全无关", "threshold": 1.0}))
    assert [e["text"] for e in evs if e["type"] == "token"] == [demo.EMPTY_RESPONSE]
    assert chat.messages is None  # 没有片段时不调用大模型


def test_upload_validation(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, monkeypatch)
    ok = client.post("/api/upload", files={"file": ("新资料.md", "# 新资料\n\n检索增强生成。".encode("utf-8"))})
    assert ok.status_code == 200 and ok.json()["documents"] == 6
    assert client.post("/api/upload", files={"file": ("../evil.md", b"x")}).status_code == 200
    assert (tmp_path / "uploads" / "evil.md").exists() and not (tmp_path / "evil.md").exists()  # 路径部分被去掉
    assert client.post("/api/upload", files={"file": ("a.exe", b"x")}).status_code == 400
    assert client.post("/api/upload", files={"file": ("b.md", "中文".encode("gbk"))}).status_code == 400
