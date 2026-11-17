"""Embedding 后端：优先调用本机 Ollama（与 RAGFlow 使用同一个 bge-large-zh-v1.5），
连不上时退回 sentence-transformers 加载本地模型目录。向量做 L2 归一化，并缓存到磁盘。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / ".embed_cache"


class Embedder:
    name: str

    def _embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed(self, texts: list[str], batch: int = 32) -> np.ndarray:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / (hashlib.md5(self.name.encode()).hexdigest() + ".jsonl")
        cache: dict[str, list[float]] = {}
        if cache_file.exists():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                cache[item["k"]] = item["v"]
        keys = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
        with cache_file.open("a", encoding="utf-8") as f:
            for i in range(0, len(missing), batch):
                part = missing[i:i + batch]
                vecs = self._embed([t for _, t in part])
                for (k, _), v in zip(part, vecs):
                    cache[k] = [round(float(x), 6) for x in v]
                    f.write(json.dumps({"k": k, "v": cache[k]}) + "\n")
        arr = np.array([cache[k] for k in keys], dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)


class OllamaEmbedder(Embedder):
    def __init__(self, model: str = "quentinz/bge-large-zh-v1.5", url: str = "http://localhost:11434"):
        self.model, self.url = model, url.rstrip("/")
        self.name = f"ollama:{model}"

    def available(self) -> bool:
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=3).json()
        except (requests.RequestException, ValueError):
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def _embed(self, texts: list[str]) -> np.ndarray:
        r = requests.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts}, timeout=600)
        r.raise_for_status()
        return np.array(r.json()["embeddings"], dtype=np.float32)


class LocalSTEmbedder(Embedder):
    def __init__(self, path: str):
        from sentence_transformers import SentenceTransformer  # 只有走本地后端时才需要
        self.model = SentenceTransformer(path, device="cpu")
        self.name = f"st:{Path(path).name}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=16, normalize_embeddings=True)


def get_embedder(backend: str = "auto") -> Embedder:
    """backend: auto | ollama | local。local 模型目录由环境变量 LOCAL_EMBED_MODEL 指定。"""
    ollama = OllamaEmbedder(os.environ.get("OLLAMA_EMBED_MODEL", "quentinz/bge-large-zh-v1.5"),
                            os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    if backend in ("auto", "ollama"):
        if ollama.available():
            return ollama
        if backend == "ollama":
            raise RuntimeError(f"Ollama 不可用或未下载 {ollama.model}，请先运行 02_pull_and_test_models.ps1")
    path = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
    return LocalSTEmbedder(path)
