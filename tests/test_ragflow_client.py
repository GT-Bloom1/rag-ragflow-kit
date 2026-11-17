"""用假 RAGFlow 服务端到端跑一遍：建库 → 上传 → 解析 → 检索评测 → 对话，1.0 与 0.2x 各一次。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import fake_ragflow  # noqa: E402
from ragflow_client import chat_demo, retrieval_eval, setup_kb  # noqa: E402
from ragflow_client.kb_config import KbConfig, v1_parser_config  # noqa: E402
from ragflow_client.metrics import first_hit_rank, summarize  # noqa: E402


@pytest.fixture(params=["v1", "legacy"])
def server(request, monkeypatch, tmp_path):
    srv, state, url = fake_ragflow.start(request.param)
    monkeypatch.setenv("RAGFLOW_BASE_URL", url)
    monkeypatch.setenv("RAGFLOW_API_KEY", "test-key")
    monkeypatch.setenv("DATASET_NAME", "测试库")
    monkeypatch.setattr(setup_kb, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(retrieval_eval, "ROOT", tmp_path)
    yield state
    srv.shutdown()


def run(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_end_to_end(server, monkeypatch, capsys):
    run(setup_kb, monkeypatch, "--chunk-size", "256", "--overlap", "10")
    ds = next(iter(server.datasets.values()))
    if server.mode == "v1":
        assert ds["parser_config"]["GeneralChunker:SixApplesFall"]["chunk_token_size"] == 256
    else:
        assert ds["parser_config"]["chunk_token_num"] == 256
        assert ds["parser_config"]["overlapped_percent"] == 10
    assert len(server.docs) == 5
    assert all(d["meta_fields"]["course"] == "CS-AI-301" for d in server.docs.values())
    assert all(str(d["run"]) in {"DONE", "3"} for d in server.docs.values())

    # 再跑一次不应重复上传
    run(setup_kb, monkeypatch)
    assert len(server.docs) == 5

    run(retrieval_eval, monkeypatch, "--questions", str(ROOT / "eval" / "questions.jsonl"),
        "--keyword-weights", "0.7,0.3")
    rq = [b for m, p, b in server.calls if p == "/retrieval"]
    assert {(b["keywords_similarity_weight"], b["vector_similarity_weight"]) for b in rq} == {(0.7, 0.3), (0.3, 0.7)}
    assert (retrieval_eval.ROOT / "results" / "ragflow_eval_summary.csv").exists()

    run(chat_demo, monkeypatch, "如何防止模型编造答案？")
    assert "助教： 回答" in capsys.readouterr().out
    assert len(server.chats) == 1


def test_v1_config_keeps_unknown_components():
    cur = {"GeneralChunker:A": {"chunk_token_size": 512, "overlapped_percent": 0},
           "Extractor:B": {"llm_id": "x", "auto_keywords": 0}}
    new, applied, manual = v1_parser_config(cur, KbConfig(chunk_size=128, overlap_percent=15, auto_keywords=3))
    assert new["GeneralChunker:A"] == {"chunk_token_size": 128, "overlapped_percent": 15}
    assert new["Extractor:B"] == {"llm_id": "x", "auto_keywords": 3}
    assert manual == ["Auto-question = 1"]
    assert cur["GeneralChunker:A"]["chunk_token_size"] == 512  # 不修改入参


def test_metrics():
    chunks = ["第一段", "过拟合是指模型在训练集上\n表现很好，但……"]
    assert first_hit_rank(chunks, "过拟合是指模型在训练集上表现很好") == 2
    assert first_hit_rank(chunks, "不存在") is None
    s = summarize([1, 2, None, 5])
    assert s["hit@1"] == 0.25 and s["hit@3"] == 0.5 and s["hit@5"] == 0.75
    assert abs(s["mrr"] - (1 + 0.5 + 0.2) / 4) < 1e-9
