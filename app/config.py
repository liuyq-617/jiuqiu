"""
配置文件 - CRM 知识库问答系统
"""
import os
from pathlib import Path

# 自动加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

# ========== 基础路径 ==========
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_TEMP_DIR = BASE_DIR / "data" / "uploads_temp"  # 临时上传文件目录

# ========== Chat 模型配置 ==========
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")  # 必填，在 .env 中设置
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://api.openai.com/v1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# chat 模式: "completions"（标准 /v1/chat/completions）或 "responses"（新版 /v1/responses）
CHAT_API_MODE = os.getenv("CHAT_API_MODE", "responses")

# ========== Embedding 模型配置 ==========
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")  # 必填，在 .env 中设置
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small 维度

# ========== Milvus 配置 ==========
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "crm_knowledge_base")

# ========== 文档分块配置 ==========
CHUNK_SIZE = 1500         # 每块最大字符数（足够容纳一条完整活动记录）
CHUNK_OVERLAP = 100       # 块间重叠字符数（仅超长记录二次拆分时生效）
CHUNK_SEPARATOR = "---"   # 主分隔符 (markdown 横线)

# ========== RAG 检索配置 ==========
TOP_K = 5                 # 检索返回的相关片段数量
SCORE_THRESHOLD = 0.3     # 相似度阈值 (余弦相似度，越高越相关)

# ========== FastAPI 配置 ==========
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "CRM 知识库问答系统"
API_VERSION = "1.0.0"

# ========== Advanced RAG 配置 (11.1) ==========
# 总开关：为 False 时回退到原始向量检索
ADVANCED_RAG_ENABLED = os.getenv("ADVANCED_RAG_ENABLED", "true").lower() == "true"

# 1. 查询改写：用 LLM 生成 N 个查询变体，扩大召回
ADVANCED_RAG_QUERY_REWRITE = os.getenv("ADVANCED_RAG_QUERY_REWRITE", "true").lower() == "true"
ADVANCED_RAG_REWRITE_N = int(os.getenv("ADVANCED_RAG_REWRITE_N", "2"))       # 生成变体数量

# 2. 混合检索：向量检索 + BM25 RRF 融合
#    expand_factor 决定每个查询召回 top_k * N 条候选，再融合裁剪
ADVANCED_RAG_HYBRID_SEARCH = os.getenv("ADVANCED_RAG_HYBRID_SEARCH", "true").lower() == "true"
ADVANCED_RAG_EXPAND_FACTOR = int(os.getenv("ADVANCED_RAG_EXPAND_FACTOR", "3"))  # 候选扩展倍数

# 3. LLM 重排序：对前 N 条候选用 LLM 重打分
ADVANCED_RAG_RERANKER = os.getenv("ADVANCED_RAG_RERANKER", "true").lower() == "true"
ADVANCED_RAG_RERANK_TOP_N = int(os.getenv("ADVANCED_RAG_RERANK_TOP_N", "10"))   # 送入重排序的候选数

# 4. 父文档检索：子块命中时自动拼合完整父活动记录
ADVANCED_RAG_PARENT_DOC = os.getenv("ADVANCED_RAG_PARENT_DOC", "true").lower() == "true"

# ========== 摘要向量检索配置 ==========
# 索引时为每条活动记录生成 LLM 摘要，用摘要向量检索，用原文生成回答
SUMMARY_RAG_ENABLED = os.getenv("SUMMARY_RAG_ENABLED", "false").lower() == "true"
SUMMARY_LLM_MODEL = os.getenv("SUMMARY_LLM_MODEL", OPENAI_CHAT_MODEL)
SUMMARY_MAX_CONCURRENCY = int(os.getenv("SUMMARY_MAX_CONCURRENCY", "5"))

# ========== 飞书机器人配置（长连接模式，无需公网域名）==========
# 飞书开放平台 -> 凭证与基础信息 -> App ID / App Secret
FEISHU_BOT_ENABLED = os.getenv("FEISHU_BOT_ENABLED", "false").lower() == "true"
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
