#!/usr/bin/env bash
# Linux / macOS 版部署脚本。用法：
#   bash setup.sh models   # 下载并测试 Embedding / 对话模型
#   bash setup.sh deploy   # 克隆 RAGFlow、切换版本、启动 Docker Compose
#   bash setup.sh check    # 检查容器状态以及容器到 Ollama 的连通性
set -euo pipefail

EMBED_MODEL="${EMBED_MODEL:-quentinz/bge-large-zh-v1.5}"
CHAT_MODEL="${CHAT_MODEL:-qwen2.5:7b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
RAGFLOW_DIR="${RAGFLOW_DIR:-$HOME/ragflow}"
RAGFLOW_TAG="${RAGFLOW_TAG:-v1.0.0}"
WEB_PORT="${WEB_PORT:-80}"

models() {
  command -v ollama >/dev/null || { echo "请先安装 Ollama: https://ollama.com/download"; exit 1; }
  ollama pull "$EMBED_MODEL"
  ollama pull "$CHAT_MODEL"
  ollama list
  dim=$(curl -s "$OLLAMA_URL/api/embed" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$EMBED_MODEL\",\"input\":\"检索增强生成通过检索外部知识来辅助大语言模型回答问题。\"}" \
    | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["embeddings"][0]))')
  echo "向量维度 = $dim（预期 1024）"
}

deploy() {
  if [[ "$(uname)" == "Linux" ]]; then
    cur=$(sysctl -n vm.max_map_count)
    if (( cur < 262144 )); then
      echo "vm.max_map_count=$cur，需要 >= 262144，正在设置（需要 sudo）"
      sudo sysctl -w vm.max_map_count=262144
    fi
  fi
  [[ -d "$RAGFLOW_DIR/.git" ]] || git clone https://github.com/infiniflow/ragflow.git "$RAGFLOW_DIR"
  cd "$RAGFLOW_DIR/docker"
  git fetch --tags && git checkout -f "$RAGFLOW_TAG"
  grep '^RAGFLOW_IMAGE' .env || true
  docker compose -f docker-compose.yml up -d
  echo "等待 RAGFlow 就绪..."
  for _ in $(seq 1 90); do
    if curl -sf "http://localhost:$WEB_PORT/api/v1/system/healthz" >/dev/null; then
      echo "RAGFlow 已就绪: http://localhost:$WEB_PORT"; return 0
    fi
    sleep 10
  done
  echo "超时，请执行 bash setup.sh check 或 docker compose logs 查看原因"; return 1
}

check() {
  docker ps --format 'table {{.Names}}\t{{.Status}}'
  name=$(docker ps --filter "label=com.docker.compose.service=ragflow-cpu" --format '{{.Names}}' | head -n1)
  [[ -n "$name" ]] || name=$(docker ps --filter "name=ragflow-server" --format '{{.Names}}' | head -n1)
  [[ -n "$name" ]] || { echo "未找到 RAGFlow 主容器"; exit 1; }
  echo "RAGFlow 主容器: $name"
  # Linux 上若容器解析不了 host.docker.internal，在 docker-compose.yml 的 RAGFlow 服务下加：
  #   extra_hosts: ["host.docker.internal:host-gateway"]
  # 并确认 Ollama 监听 0.0.0.0（systemd 安装：sudo systemctl edit ollama，加 Environment="OLLAMA_HOST=0.0.0.0:11434"）
  docker exec "$name" curl -s -m 10 http://host.docker.internal:11434/api/tags && echo && echo "容器可以访问 Ollama"
}

case "${1:-}" in
  models) models ;;
  deploy) deploy ;;
  check) check ;;
  *) echo "用法: bash setup.sh {models|deploy|check}"; exit 1 ;;
esac
