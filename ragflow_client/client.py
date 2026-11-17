"""RAGFlow HTTP API 的精简客户端。

同时兼容两套接口：
  * v1 风格：RAGFlow 1.0（parse_type/parser_id、组件化 parser_config、llm_id/prompt_config）
  * legacy 风格：RAGFlow 0.2x（chunk_method、扁平 parser_config、llm/prompt）
首次建库时先按 1.0 写法请求，服务端拒绝后再退回旧写法，并记住所用风格。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

DONE_STATES = {"DONE", "3"}
FAIL_STATES = {"FAIL", "4", "CANCEL", "2"}


class RagflowError(RuntimeError):
    def __init__(self, code: Any, message: str, path: str = ""):
        super().__init__(f"[{code}] {message} ({path})")
        self.code = code
        self.message = message


class RagflowClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 120):
        if not api_key:
            raise ValueError("缺少 RAGFLOW_API_KEY：在 RAGFlow 界面 头像 → API 中创建，写入 config/.env")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.api_style: str | None = None  # "v1" 或 "legacy"

    # ---------- 基础请求 ----------
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}/api/v1{path}"
        resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
        try:
            body = resp.json()
        except ValueError:
            raise RagflowError(resp.status_code, resp.text[:300], path) from None
        if resp.status_code >= 400 or body.get("code", 0) != 0:
            raise RagflowError(body.get("code", resp.status_code), body.get("message", resp.text[:300]), path)
        return body.get("data")

    @staticmethod
    def _as_list(data: Any, *keys: str) -> list[dict]:
        """不同版本的列表接口有的直接返回数组，有的包一层 {docs: [...]}。"""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in keys:
                if isinstance(data.get(key), list):
                    return data[key]
        return []

    # ---------- 知识库（dataset） ----------
    def find_dataset(self, name: str) -> dict | None:
        try:
            data = self._request("GET", "/datasets", params={"name": name, "page_size": 100})
        except RagflowError:
            return None  # 0.2x 中按名称查不到时会返回错误码
        for ds in self._as_list(data, "datasets", "items", "data"):
            if ds.get("name") == name:
                return ds
        return None

    def get_dataset(self, dataset_id: str) -> dict:
        data = self._request("GET", "/datasets", params={"id": dataset_id})
        items = self._as_list(data, "datasets", "items", "data")
        if not items:
            raise RagflowError(404, f"找不到知识库 {dataset_id}", "/datasets")
        return items[0]

    def create_dataset(self, name: str, embedding_model: str, legacy_parser_config: dict) -> dict:
        """先用 1.0 写法创建 General 知识库，失败后改用 0.2x 写法。"""
        v1_body = {"name": name, "embedding_model": embedding_model, "parse_type": 1, "parser_id": "general"}
        try:
            ds = self._request("POST", "/datasets", json=v1_body)
            self.api_style = "v1"
            return ds
        except RagflowError as v1_err:
            legacy_body = {"name": name, "embedding_model": embedding_model,
                           "chunk_method": "naive", "parser_config": legacy_parser_config}
            try:
                ds = self._request("POST", "/datasets", json=legacy_body)
            except RagflowError as legacy_err:
                raise RagflowError(legacy_err.code, f"两种接口写法都失败。1.0: {v1_err}; 0.2x: {legacy_err}") from None
            self.api_style = "legacy"
            return ds

    def update_dataset(self, dataset_id: str, body: dict) -> None:
        self._request("PUT", f"/datasets/{dataset_id}", json=body)

    def detect_style(self, dataset: dict) -> str:
        """已有知识库：parser_config 里出现 "组件名:实例名" 形式的键即为 1.0。"""
        cfg = dataset.get("parser_config") or {}
        self.api_style = "v1" if any(":" in k for k in cfg) else "legacy"
        return self.api_style

    # ---------- 文档 ----------
    def upload_documents(self, dataset_id: str, paths: list[Path]) -> list[dict]:
        files = [("file", (p.name, p.read_bytes())) for p in paths]
        data = self._request("POST", f"/datasets/{dataset_id}/documents", files=files)
        return self._as_list(data, "docs", "documents")

    def list_documents(self, dataset_id: str) -> list[dict]:
        docs: list[dict] = []
        page = 1
        while True:
            data = self._request("GET", f"/datasets/{dataset_id}/documents",
                                 params={"page": page, "page_size": 100})
            batch = self._as_list(data, "docs", "documents")
            docs.extend(batch)
            if len(batch) < 100:
                return docs
            page += 1

    def update_document(self, dataset_id: str, document_id: str, body: dict) -> None:
        self._request("PATCH", f"/datasets/{dataset_id}/documents/{document_id}", json=body)

    def parse_documents(self, dataset_id: str, document_ids: list[str]) -> None:
        self._request("POST", f"/datasets/{dataset_id}/chunks", json={"document_ids": document_ids})

    def wait_parsed(self, dataset_id: str, document_ids: list[str], timeout_s: int = 1800,
                    interval_s: int = 5, log=print) -> list[dict]:
        """轮询直到所有文档解析完成；任何文档失败即报错。"""
        deadline = time.time() + timeout_s
        wanted = set(document_ids)
        while time.time() < deadline:
            docs = [d for d in self.list_documents(dataset_id) if d.get("id") in wanted]
            states = {d["id"]: str(d.get("run", "")).upper() for d in docs}
            failed = [d for d in docs if states[d["id"]] in FAIL_STATES]
            if failed:
                msgs = "; ".join(f"{d.get('name')}: {(d.get('progress_msg') or '')[-200:]}" for d in failed)
                raise RagflowError("PARSE_FAIL", msgs, "/documents")
            done = sum(states[i] in DONE_STATES for i in states)
            log(f"  解析进度 {done}/{len(wanted)}  " + "  ".join(
                f"{d.get('name')}:{float(d.get('progress') or 0):.0%}" for d in docs))
            if done == len(wanted):
                return docs
            time.sleep(interval_s)
        raise TimeoutError(f"{timeout_s} 秒内未解析完成")

    # ---------- 检索 ----------
    def retrieve(self, question: str, dataset_ids: list[str], keyword_weight: float = 0.3,
                 similarity_threshold: float = 0.2, top_n: int = 8, metadata_condition: dict | None = None,
                 rerank_id: str | None = None) -> dict:
        """混合检索。keyword_weight 即界面上的 Keyword similarity weight，向量权重 = 1 - keyword_weight。
        同时发送两个字段：1.0 读 keywords_similarity_weight，0.2x 读 vector_similarity_weight，二者之和为 1。"""
        body: dict[str, Any] = {
            "question": question,
            "dataset_ids": dataset_ids,
            "similarity_threshold": similarity_threshold,
            "keywords_similarity_weight": round(keyword_weight, 4),
            "vector_similarity_weight": round(1 - keyword_weight, 4),
            "page": 1,
            "page_size": top_n,
            "highlight": False,
        }
        if metadata_condition:
            body["metadata_condition"] = metadata_condition
        if rerank_id:
            body["rerank_id"] = rerank_id
        return self._request("POST", "/retrieval", json=body) or {}

    # ---------- 聊天助手 ----------
    def find_chat(self, name: str) -> dict | None:
        try:
            data = self._request("GET", "/chats", params={"name": name, "page_size": 100})
        except RagflowError:
            return None
        for chat in self._as_list(data, "chats", "items", "data"):
            if chat.get("name") == name:
                return chat
        return None

    def delete_chats(self, ids: list[str]) -> None:
        self._request("DELETE", "/chats", json={"ids": ids})

    def create_chat(self, name: str, dataset_ids: list[str], llm_id: str, system_prompt: str,
                    empty_response: str, keyword_weight: float = 0.3, similarity_threshold: float = 0.2,
                    top_n: int = 6, temperature: float = 0.1) -> dict:
        v1_body = {
            "name": name,
            "dataset_ids": dataset_ids,
            "llm_id": llm_id,
            "llm_setting": {"model_type": "chat", "temperature": temperature},
            "prompt_config": {
                "system": system_prompt,
                "parameters": [{"key": "knowledge", "optional": False}],
                "empty_response": empty_response,
                "quote": True,
            },
            "similarity_threshold": similarity_threshold,
            "keywords_similarity_weight": keyword_weight,
            "top_n": top_n,
        }
        legacy_body = {
            "name": name,
            "dataset_ids": dataset_ids,
            "llm": {"model_name": llm_id, "temperature": temperature},
            "prompt": {
                "prompt": system_prompt,
                "variables": [{"key": "knowledge", "optional": False}],
                "empty_response": empty_response,
                "show_quote": True,
                "similarity_threshold": similarity_threshold,
                "keywords_similarity_weight": keyword_weight,
                "top_n": top_n,
            },
        }
        order = [legacy_body, v1_body] if self.api_style == "legacy" else [v1_body, legacy_body]
        errors = []
        for body in order:
            try:
                return self._request("POST", "/chats", json=body)
            except RagflowError as err:
                errors.append(str(err))
        raise RagflowError("CHAT_CREATE", " | ".join(errors), "/chats")

    def ask(self, chat_id: str, messages: list[dict], with_reference: bool = True) -> dict:
        """OpenAI 兼容接口（非流式）。1.0 路径为 /openai/{id}/chat/completions，0.2x 为 /chats_openai/...。"""
        body = {"model": "model", "messages": messages, "stream": False,
                "extra_body": {"reference": with_reference}}
        last_err: RagflowError | None = None
        for path in (f"/openai/{chat_id}/chat/completions", f"/chats_openai/{chat_id}/chat/completions"):
            url = f"{self.base_url}/api/v1{path}"
            resp = self.session.post(url, json=body, timeout=self.timeout)
            if resp.status_code == 404:
                last_err = RagflowError(404, "接口不存在", path)
                continue
            try:
                data = resp.json()
            except ValueError:
                last_err = RagflowError(resp.status_code, resp.text[:300], path)
                continue
            if "choices" in data:
                return data
            last_err = RagflowError(data.get("code", resp.status_code), data.get("message", str(data)[:300]), path)
        raise last_err or RagflowError("ASK", "未知错误")
