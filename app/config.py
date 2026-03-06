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
