"""测试用的假 RAGFlow 服务：按 1.0（v1）或 0.2x（legacy）接口规则校验请求字段。

只模拟脚本用到的接口，用来验证客户端的请求格式和版本回退逻辑，不模拟真实的检索算法。
"""
from __future__ import annotations

from email.parser import BytesParser
from email.policy import default as email_policy
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class FakeState:
    def __init__(self, mode: str):
        self.mode = mode  # "v1" or "legacy"
        self.datasets: dict[str, dict] = {}
        self.docs: dict[str, dict] = {}
        self.chats: dict[str, dict] = {}
        self.calls: list[tuple[str, str, dict]] = []


def make_handler(state: FakeState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # 保持测试输出干净
            pass

        def _send(self, status: int, payload: dict | None = None, raw: str | None = None):
            body = (raw if raw is not None else json.dumps(payload, ensure_ascii=False)).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json" if raw is None else "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def ok(self, data=None):
            self._send(200, {"code": 0, "data": data})

        def err(self, message: str, code: int = 101):
            self._send(200, {"code": code, "message": message})

        def _json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _files(self) -> list[tuple[str, bytes]]:
            n = int(self.headers.get("Content-Length") or 0)
            head = f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode()
            msg = BytesParser(policy=email_policy).parsebytes(head + self.rfile.read(n))
            return [(p.get_filename(), p.get_payload(decode=True)) for p in msg.iter_parts()]

        def _route(self, method: str):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if self.headers.get("Authorization") != "Bearer test-key":
                return self.err("auth", 109)
            parts = url.path.removeprefix("/api/v1").strip("/").split("/")
            body = {} if method == "GET" or "multipart" in (self.headers.get("Content-Type") or "") else self._json()
            state.calls.append((method, "/" + "/".join(parts), body))
            if parts[0] == "datasets":
                return self.datasets(method, parts, q, body)
            if parts == ["retrieval"]:
                return self.retrieval(body)
            if parts[0] == "chats":
                return self.chats(method, body)
            if parts[0] in ("openai", "chats_openai") and parts[-1] == "completions":
                if (parts[0] == "openai") != (state.mode == "v1"):
                    return self._send(404, raw="<h1>404</h1>")
                return self._send(200, {"choices": [{"message": {"role": "assistant", "content": "回答",
                                                                 "reference": {"chunks": []}}}]})
            return self._send(404, raw="<h1>404</h1>")

        do_GET = lambda self: self._route("GET")
        do_POST = lambda self: self._route("POST")
        do_PUT = lambda self: self._route("PUT")
        do_PATCH = lambda self: self._route("PATCH")
        do_DELETE = lambda self: self._route("DELETE")

        # ---- /datasets ----
        def datasets(self, method, parts, q, body):
            if len(parts) == 1 and method == "GET":
                items = [d for d in state.datasets.values()
                         if q.get("name") in (None, d["name"]) and q.get("id") in (None, d["id"])]
                if state.mode == "legacy" and not items:
                    return self.err("You don't own the dataset", 102)  # 0.2x 的真实行为
                return self.ok(items)
            if len(parts) == 1 and method == "POST":
                if state.mode == "v1":
                    if "chunk_method" in body or body.get("parse_type") != 1 or body.get("parser_id") != "general":
                        return self.err("chunk_method is not supported; use parse_type + parser_id")
                    cfg = {"GeneralChunker:SixApplesFall": {"chunk_token_size": 512, "delimiters": ["\n"]},
                           "Extractor:AutoExtractDefault": {"llm_id": ""}}
                else:
                    if "parse_type" in body or body.get("chunk_method") != "naive":
                        return self.err("Extra inputs are not permitted: parse_type")
                    cfg = body["parser_config"]
                ds = {"id": uuid.uuid4().hex, "name": body["name"], "embedding_model": body["embedding_model"],
                      "parser_config": cfg}
                state.datasets[ds["id"]] = ds
                return self.ok(ds)
            ds = state.datasets[parts[1]]
            if len(parts) == 2 and method == "PUT":
                if state.mode == "v1" and any(":" not in k for k in body.get("parser_config", {})):
                    return self.err("legacy flat keys are ignored")
                ds["parser_config"] = body.get("parser_config", ds["parser_config"])
                return self.ok()
            if parts[2] == "documents" and method == "POST":
                docs = []
                for name, content in self._files():
                    doc = {"id": uuid.uuid4().hex, "name": name, "run": "UNSTART", "dataset_id": ds["id"],
                           "content": content.decode("utf-8"), "chunk_count": 0, "token_count": 0}
                    state.docs[doc["id"]] = doc
                    docs.append(doc)
                return self.ok(docs)
            if parts[2] == "documents" and method == "GET":
                docs = [{k: v for k, v in d.items() if k != "content"}
                        for d in state.docs.values() if d["dataset_id"] == ds["id"]]
                return self.ok({"docs": docs, "total": len(docs)} if state.mode == "v1" else docs)
            if parts[2] == "documents" and method == "PATCH":
                state.docs[parts[3]]["meta_fields"] = body["meta_fields"]
                return self.ok()
            if parts[2] == "chunks" and method == "POST":
                for doc_id in body["document_ids"]:
                    d = state.docs[doc_id]
                    d.update(run="DONE" if state.mode == "legacy" else "3", progress=1.0,
                             chunk_count=d["content"].count("\n\n") + 1, token_count=len(d["content"]))
                return self.ok()
            return self.err("unknown dataset route")

        # ---- /retrieval ----
        def retrieval(self, body):
            kw, vec = body.get("keywords_similarity_weight"), body.get("vector_similarity_weight")
            if kw is not None and vec is not None and abs(kw + vec - 1) > 1e-6:
                return self.err("keywords_similarity_weight + vector_similarity_weight must equal 1")
            paras = []
            for d in state.docs.values():
                if d["dataset_id"] in body["dataset_ids"]:
                    paras += [(p, d["name"]) for p in d["content"].split("\n\n") if p.strip()]
            qset = set(body["question"])
            scored = sorted(paras, key=lambda p: -len(qset & set(p[0])))[: body.get("page_size", 30)]
            chunks = [{"content": p, "document_keyword": name, "similarity": 0.5,
                       "vector_similarity": 0.5, "term_similarity": 0.5} for p, name in scored]
            return self.ok({"chunks": chunks, "total": len(chunks)})

        # ---- /chats ----
        def chats(self, method, body):
            if method == "GET":
                return self.ok(list(state.chats.values()))
            if state.mode == "v1" and ("llm" in body or "prompt" in body):
                return self.err("llm/prompt removed; use llm_id/prompt_config")
            if state.mode == "legacy" and "llm_id" in body:
                return self.err("Extra inputs are not permitted: llm_id")
            system = body.get("prompt_config", {}).get("system") or body.get("prompt", {}).get("prompt", "")
            if "{knowledge}" not in system:
                return self.err("Parameter 'knowledge' is not used")
            chat = {"id": uuid.uuid4().hex, "name": body["name"], "dataset_ids": body["dataset_ids"]}
            state.chats[chat["id"]] = chat
            return self.ok(chat)

    return Handler


def start(mode: str) -> tuple[ThreadingHTTPServer, FakeState, str]:
    state = FakeState(mode)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state, f"http://127.0.0.1:{server.server_address[1]}"
