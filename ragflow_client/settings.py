"""读取配置：优先环境变量，其次项目根目录下的 config/.env。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "config" / ".env"


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    embed_model: str
    chat_model: str
    dataset_name: str
    chat_name: str


def load_settings() -> Settings:
    file_values = _load_env_file(ENV_FILE)

    def get(key: str, default: str = "") -> str:
        return os.environ.get(key) or file_values.get(key) or default

    return Settings(
        base_url=get("RAGFLOW_BASE_URL", "http://localhost").rstrip("/"),
        api_key=get("RAGFLOW_API_KEY"),
        embed_model=get("EMBED_MODEL", "quentinz/bge-large-zh-v1.5@Ollama"),
        chat_model=get("CHAT_MODEL", "qwen2.5:7b@Ollama"),
        dataset_name=get("DATASET_NAME", "AI课程知识库"),
        chat_name=get("CHAT_NAME", "AI课程助教"),
    )
