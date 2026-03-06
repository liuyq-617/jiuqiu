# CRM 知识库问答系统

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

基于 **OpenAI** + **Milvus** + **FastAPI** 的本地 CRM 知识库 RAG 问答系统。

## 系统架构

```
CRM Markdown 文档
       │
       ▼
  文档解析 & 分块  (document_loader.py)
       │
       ▼
  OpenAI Embedding  (text-embedding-3-small)
       │
       ▼
  Milvus 向量存储  (本地 Docker)
       │
  ─────┼───── 用户查询
       │          │
       ▼          ▼
  向量检索  ←  问题向量化
       │
       ▼
  GPT-4o-mini 生成答案  (RAG)
       │
       ▼
  FastAPI + SSE 流式返回
       │
       ▼
  浏览器网页界面
```

## 项目结构

```
crm_kb/
├── app/
│   ├── config.py          # 全局配置（读取环境变量）
│   ├── document_loader.py # Markdown 解析与分块
│   ├── vector_store.py    # Milvus 连接 + Embedding + 检索
│   ├── rag.py             # RAG 问答核心（普通 + 流式）
│   └── main.py            # FastAPI 应用入口
├── scripts/
│   └── build_index.py     # 一键构建向量索引
├── static/
│   └── index.html         # 前端聊天界面
├── data/                  # 放置 CRM markdown 文档（可为空，自动向上查找）
├── docker-compose.yml     # Milvus + etcd + minio
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量模板
└── start.sh               # 一键启动脚本
```

## 快速开始

### 1. 配置环境变量

```bash
cd /root/crm/crm_kb
cp .env.example .env
nano .env   # 填写 OPENAI_API_KEY
```

### 2. 一键启动

```bash
./start.sh
```

或者分步执行：

```bash
# 启动 Milvus
docker compose up -d
sleep 30  # 等待就绪

# 安装 Python 依赖
pip install -r requirements.txt

# 构建知识库索引（首次或文档更新后执行）
python scripts/build_index.py

# 启动 Web 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 访问

- **聊天界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/chat` | 普通问答 |
| `POST` | `/api/chat/stream` | 流式问答（SSE） |
| `POST` | `/api/index` | 重建知识库索引 |
| `GET` | `/api/stats` | 知识库统计 |

### 问答接口示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "最近有哪些客户进展？", "top_k": 5}'
```

## 文档管理

- 将 CRM Markdown 文件放入 `crm_kb/data/` 目录，或保留在上级 `/root/crm/` 目录
- 添加/更新文档后，运行 `python scripts/build_index.py` 或调用 `POST /api/index` 重建索引

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | 必填 | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 兼容第三方代理 |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | 问答对话模型 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 向量化模型 |
| `MILVUS_HOST` | `localhost` | Milvus 地址 |
| `MILVUS_PORT` | `19530` | Milvus 端口 |
| `MILVUS_COLLECTION` | `crm_knowledge_base` | 集合名称 |
