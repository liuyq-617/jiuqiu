"""
FastAPI 主应用
接口列表:
  GET  /              - 前端页面
  GET  /api/health    - 健康检查
  POST /api/chat      - 普通问答（JSON 响应）
  POST /api/chat/stream - 流式问答（SSE）
  POST /api/index     - 触发知识库重建
  GET  /api/stats     - 知识库统计信息
"""
import sys
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s][%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crm_main")

from app.config import API_TITLE, API_VERSION, TOP_K, BASE_DIR
from app.rag import answer, answer_stream
from app.vector_store import connect_milvus, ensure_collection, get_aggregate_stats
from app.document_loader import load_and_split
from app.vector_store import build_index
from app.feishu_bot import start_ws_client
from app.feedback import init_db, save_qa, save_thumbs, save_manual_scores, trigger_judge, get_stats as get_feedback_stats


# ========== 应用生命周期 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时开启飞书长连接 + 初始化反馈数据库"""
    init_db()
    start_ws_client()
    yield

# ========== OpenAPI Tags ==========
openapi_tags = [
    {
        "name": "问答",
        "description": "向 CRM 知识库提问，获取基于真实销售记录的回答。",
    },
    {
        "name": "知识库",
        "description": "查询知识库状态或触发全量索引重建。",
    },
    {
        "name": "飞书机器人",
        "description": "飞书 WebSocket 长连接，无需公网域名，应用启动时自动接入飞书并响应消息。",
    },
]

# ========== 创建应用 ==========
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan,
    description=(
        "基于 OpenAI Embedding + Milvus 向量检索的 CRM 知识库 RAG 问答系统。\n\n"
        "**使用流程**\n"
        "1. `GET /api/health` 确认服务就绪\n"
        "2. `POST /api/chat` 提交问题，获得含来源引用的完整回答\n"
        "3. `POST /api/chat/stream` 获取流式 SSE 回答（适合实时展示）\n"
    ),
    openapi_tags=openapi_tags,
    servers=[
        {"url": "http://localhost:8000", "description": "本地开发服务器"},
    ],
    contact={"name": "CRM KB"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")



# ========== 请求/响应模型 ==========
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题，例如：蒯歆越最近负责哪些客户？")
    top_k: int = Field(default=TOP_K, ge=1, le=20, description="向量检索返回的相关片段数量，默认 5")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "张三最近跟进了哪些客户？", "top_k": 5}
            ]
        }
    }


class SourceItem(BaseModel):
    source: str = Field(description="来源文件名")
    chunk_id: str = Field(description="片段编号")
    company: str = Field(description="相关公司")
    owner: str = Field(description="负责人")
    date: str = Field(description="活动日期")
    score: float = Field(description="相关度得分（余弦相似度，越高越相关）")
    text: str = Field(description="命中的原始文本片段")


class ChatResponse(BaseModel):
    success: bool = Field(description="请求是否成功")
    data: Dict[str, Any] = Field(
        description="问答结果，包含 answer（回答文本）和 sources（来源列表）"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "data": {
                        "answer": "张三最近跟进了 A 公司和 B 公司……",
                        "sources": [
                            {
                                "source": "crm_activities_recent.md",
                                "chunk_id": "chunk_0042",
                                "company": "A 公司",
                                "owner": "张三",
                                "date": "2026-01-15",
                                "score": 0.87,
                                "text": "……原始记录内容……"
                            }
                        ]
                    }
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str = Field(description="服务状态：ok / error")
    milvus: Optional[str] = Field(default=None, description="Milvus 连接状态")
    collection: Optional[str] = Field(default=None, description="向量集合名称")
    doc_count: Optional[int] = Field(default=None, description="已索引的文档片段数量")
    detail: Optional[str] = Field(default=None, description="错误详情（仅 status=error 时出现）")


class StatsResponse(BaseModel):
    success: bool
    collection: Optional[str] = None
    doc_count: Optional[int] = None
    schema_info: Optional[Dict[str, str]] = Field(default=None, description="各字段类型映射")
    detail: Optional[str] = None


class IndexRequest(BaseModel):
    confirm: bool = Field(default=False, description="必须设置为 true 才会执行重建，防止误触发")


class IndexResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    chunk_count: Optional[int] = Field(default=None, description="文档切分后的总片段数")
    indexed_count: Optional[int] = Field(default=None, description="成功写入向量库的片段数")
    detail: Optional[str] = None


class AggregateStatsResponse(BaseModel):
    success: bool
    total_chunks: Optional[int] = Field(default=None, description="全部活动记录片段总数")
    company_count: Optional[int] = Field(default=None, description="客户公司总数")
    companies: Optional[List[str]] = Field(default=None, description="所有客户公司名称（去重并排序）")
    owner_count: Optional[int] = Field(default=None, description="负责人总数")
    owners: Optional[List[str]] = Field(default=None, description="所有负责人姓名（去重并排序）")
    detail: Optional[str] = None


# ========== 路由 ==========
@app.get("/", include_in_schema=False)
async def index_page():
    """返回前端页面"""
    html_path = BASE_DIR / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="前端页面未找到，请检查 static/index.html")
    return FileResponse(str(html_path))


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查服务是否正常运行，返回 Milvus 连接状态和已索引的文档片段数量。",
    operation_id="healthCheck",
    tags=["知识库"],
)
async def health_check():
    """健康检查"""
    try:
        connect_milvus()
        col = ensure_collection()
        count = col.num_entities
        return {
            "status": "ok",
            "milvus": "connected",
            "collection": col.name,
            "doc_count": count,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="CRM 知识库问答",
    description=(
        "向 CRM 知识库提问，系统通过向量检索找到最相关的销售活动记录，"
        "再由大模型综合生成回答。\n\n"
        "响应的 `data.answer` 是自然语言回答，`data.sources` 是支撑回答的来源片段列表。"
    ),
    operation_id="askCrm",
    tags=["问答"],
)
async def chat(req: ChatRequest):
    """普通问答，返回完整 JSON（含 answer_id，可用于提交反馈）"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        start_ms = int(time.time() * 1000)
        result = answer(req.question, top_k=req.top_k)
        response_ms = int(time.time() * 1000) - start_ms
        answer_text = result.get("answer", "")
        sources = result.get("sources", [])
        if answer_text:
            aid = save_qa(req.question, answer_text, sources, response_ms)
            result["answer_id"] = aid
        return {"success": True, "data": result}
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"向量数据库连接失败: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/chat/stream",
    summary="CRM 知识库流式问答（SSE）",
    description=(
        "与 `/api/chat` 功能相同，但以 Server-Sent Events 流式方式逐 token 返回回答，"
        "适合需要实时展示打字效果的前端。\n\n"
        "事件格式：`data: <JSON>\\n\\n`，其中 JSON 包含 `type` 字段：\n"
        "- `type=sources`：来源片段列表\n"
        "- `type=token`：单个回答 token\n"
        "- `type=done`：流式结束\n"
        "- `type=error`：错误信息\n"
    ),
    operation_id="askCrmStream",
    tags=["问答"],
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE 流式回答"}},
)
async def chat_stream(req: ChatRequest):
    """流式问答，使用 SSE（Server-Sent Events）
    流结束后额外推送 type=answer_id 事件，前端可用于提交赞/踩反馈。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    def _stream_with_feedback():
        """包装 answer_stream，流结束后注入 answer_id 并写入反馈数据库"""
        tokens: list[str] = []
        sources: list = []
        start_ms = int(time.time() * 1000)

        for raw in answer_stream(req.question, top_k=req.top_k):
            # 先解析，再原样 yield（不改变下游行为）
            data_str = raw.removeprefix("data:").strip()
            if data_str:
                try:
                    chunk = json.loads(data_str)
                    ctype = chunk.get("type", "")
                    if ctype == "token":
                        tokens.append(chunk.get("content", ""))
                    elif ctype == "sources":
                        sources = chunk.get("sources", [])
                except Exception:
                    pass
            yield raw

        # 所有 chunk 发完后，保存到 DB 并推送 answer_id
        response_ms = int(time.time() * 1000) - start_ms
        full_answer = "".join(tokens).strip()
        if full_answer:
            aid = save_qa(req.question, full_answer, sources, response_ms)
            yield f"data: {json.dumps({'type': 'answer_id', 'id': aid}, ensure_ascii=False)}\n\n"

    try:
        return StreamingResponse(
            _stream_with_feedback(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/index",
    response_model=IndexResponse,
    summary="重建知识库索引",
    description=(
        "重新读取数据目录下的 Markdown 文档，全量重建向量索引。\n\n"
        "**注意**：必须将 `confirm` 设置为 `true`，否则接口拒绝执行，防止误操作覆盖现有数据。"
    ),
    operation_id="rebuildIndex",
    tags=["知识库"],
)
async def rebuild_index(req: IndexRequest):
    """触发知识库重建（全量）"""
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="请设置 confirm=true 以确认重建索引"
        )
    try:
        chunks = load_and_split()
        if not chunks:
            raise HTTPException(status_code=404, detail="未找到任何文档，请检查 data 目录")
        count = build_index(chunks)
        return {
            "success": True,
            "message": f"索引重建完成",
            "chunk_count": len(chunks),
            "indexed_count": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/stats",
    response_model=StatsResponse,
    summary="知识库统计信息",
    description="返回向量集合名称、已索引片段总数以及字段 Schema 概览。",
    operation_id="getStats",
    tags=["知识库"],
)
async def get_stats():
    """获取知识库统计信息"""
    try:
        connect_milvus()
        col = ensure_collection()
        return {
            "success": True,
            "collection": col.name,
            "doc_count": col.num_entities,
            "schema_info": {
                field.name: str(field.dtype)
                for field in col.schema.fields
            },
        }
    except Exception as e:
        return {"success": False, "detail": str(e)}


# ========== 反馈 & 质量看板 ==========

class FeedbackRequest(BaseModel):
    answer_id: str = Field(..., description="问答记录 ID（由 answer_id 事件或响应体中获取）")
    thumbs: Optional[int] = Field(default=None, description="1 = 赞，-1 = 踩；与手动打分不互斥")
    trigger_judge: bool = Field(default=True, description="是否异步触发 LLM-as-Judge 评分")
    # 手动打分（1~5 整数）
    manual_relevance:    Optional[float] = Field(default=None, ge=1, le=5, description="相关性 1~5")
    manual_completeness: Optional[float] = Field(default=None, ge=1, le=5, description="完整性 1~5")
    manual_accuracy:     Optional[float] = Field(default=None, ge=1, le=5, description="准确性 1~5")
    manual_comment:      Optional[str]   = Field(default=None, description="手动评语（可选）")


@app.post(
    "/api/feedback",
    summary="提交回答反馈（赞/踩 + 可选手动打分）",
    description=(
        "提交对某条回答的评价。支持两种模式：\n\n"
        "1. **快速反馈**：仅传 `thumbs`（1=赞 / -1=踩）\n"
        "2. **手动打分**：传 `manual_relevance` / `manual_completeness` / `manual_accuracy`（1~5 分）\n\n"
        "两种模式可同时使用。手动打分会根据平均分自动推算 `thumbs`（≥3.5→赞）。"
    ),
    operation_id="submitFeedback",
    tags=["质量评价"],
)
async def submit_feedback(req: FeedbackRequest):
    has_thumbs = req.thumbs is not None
    has_manual = all(v is not None for v in [req.manual_relevance, req.manual_completeness, req.manual_accuracy])

    if not has_thumbs and not has_manual:
        raise HTTPException(status_code=422, detail="请提供 thumbs 或 manual_relevance/completeness/accuracy")

    if has_thumbs:
        ok = save_thumbs(req.answer_id, req.thumbs)
        if not ok:
            raise HTTPException(status_code=404, detail=f"找不到 answer_id={req.answer_id}")

    if has_manual:
        ok = save_manual_scores(
            req.answer_id,
            req.manual_relevance,
            req.manual_completeness,
            req.manual_accuracy,
            req.manual_comment or "",
        )
        if not ok:
            raise HTTPException(status_code=404, detail=f"找不到 answer_id={req.answer_id}")

    if req.trigger_judge:
        trigger_judge(req.answer_id)

    return {
        "success": True,
        "answer_id": req.answer_id,
        "thumbs": req.thumbs,
        "manual_scored": has_manual,
    }


@app.get(
    "/api/feedback/stats",
    summary="质量看板统计数据",
    description=(
        "返回全部问答的聚合质量指标：\n\n"
        "- 总问答数、好评率\n"
        "- LLM-as-Judge 平均相关性 / 完整性 / 准确性\n"
        "- 平均响应时间\n"
        "- 近 30 条问答明细（含用户反馈 + LLM 评分）"
    ),
    operation_id="getFeedbackStats",
    tags=["质量评价"],
)
async def feedback_stats():
    try:
        return {"success": True, "data": get_feedback_stats()}
    except Exception as e:
        logger.error(f"[feedback_stats] 失败: {e}", exc_info=True)
        return {"success": False, "detail": str(e)}


@app.get(
    "/api/companies",
    response_model=AggregateStatsResponse,
    summary="列出所有客户和负责人",
    description=(
        "遍历知识库中**全部**活动记录，返回去重后的客户公司列表、负责人列表及总条数统计。\n\n"
        "此接口直接查询 Milvus 元数据字段，不经过向量检索，"
        "适合回答'共有多少客户'、'所有负责人有哪些'等聚合型问题。\n\n"
        "**注意**：数据量大时（>5000条）查询约需 2~5 秒，请耐心等待。"
    ),
    operation_id="listCompanies",
    tags=["知识库"],
)
async def list_companies():
    """获取全量客户和负责人列表"""
    try:
        stats = get_aggregate_stats()
        return {"success": True, **stats}
    except Exception as e:
        logger.error(f"[list_companies] 失败: {e}", exc_info=True)
        return {"success": False, "detail": str(e)}
