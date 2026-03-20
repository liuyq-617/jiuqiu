#!/usr/bin/env bash
# ==========================================================
#  CRM 知识库问答系统 - 一键启动脚本
# ==========================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "========================================================"
echo "  CRM 知识库问答系统"
echo "========================================================"

# ---------- 1. 检查 .env ----------
if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[警告] 已从 .env.example 创建 .env，请先编辑填写 OPENAI_API_KEY"
    echo "  编辑命令: nano .env"
    echo ""
  fi
fi

# 加载环境变量
export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null || true

if [ "$CHAT_API_KEY" = "sk-your-chat-api-key-here" ]; then
  echo "[错误] 请在 .env 文件中设置有效的 CHAT_API_KEY"
  exit 1
fi

if [ "$EMBEDDING_API_KEY" = "sk-your-embedding-api-key-here" ]; then
  echo "[错误] 请在 .env 文件中设置有效的 EMBEDDING_API_KEY"
  exit 1
fi

# ---------- 2. 启动 Milvus ----------
echo "[步骤 1] 启动 Milvus..."
docker compose up -d
echo "  等待 Milvus 就绪（3 秒）..."
sleep 3

# 检查健康状态
RETRY=0
until curl -sf http://localhost:9091/healthz > /dev/null || [ $RETRY -ge 5 ]; do
  echo "  Milvus 未就绪，等待 10 秒..."
  sleep 10
  RETRY=$((RETRY+1))
done
echo "  Milvus 就绪 ✓"

# ---------- 3. 安装 Python 依赖 ----------
echo ""
echo "[步骤 2] 安装 Python 依赖..."
pip3 install -r requirements.txt -q

# ---------- 4. 构建索引（可选，无文档时跳过）----------
echo ""
echo "[步骤 3] 构建知识库索引..."
python3 scripts/build_index.py || echo "  [跳过] 暂无文档，可后续通过 http://localhost:8000/upload 上传 MD 文件后重建索引"

# ---------- 5. 启动 Web 服务 ----------
echo ""
echo "[步骤 4] 启动 Web 服务..."
echo "  访问地址: http://localhost:8000"
echo "  API 文档: http://localhost:8000/docs"
echo "  按 Ctrl+C 停止服务"
echo "========================================================"
echo ""
PYTHONPATH="$SCRIPT_DIR" uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
