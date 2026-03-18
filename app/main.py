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
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
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

from app.config import API_TITLE, API_VERSION, TOP_K, BASE_DIR, UPLOAD_TEMP_DIR
from app.rag import answer, answer_stream
from app.vector_store import connect_milvus, ensure_collection, get_aggregate_stats, build_index, clear_index
from app.document_loader import load_and_split, process_uploaded_file
from app.feishu_bot import start_ws_client
from app.feedback import init_db, save_qa, save_thumbs, save_manual_scores, trigger_judge, get_stats as get_feedback_stats


# ========== 应用生命周期 ==========
def _cleanup_old_temp_files():
    """清理 uploads_temp 中超过 24 小时的旧临时文件"""
    import time
    try:
        cutoff = time.time() - 24 * 3600
        removed = 0
        for f in UPLOAD_TEMP_DIR.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                removed += 1
        if removed:
            logger.info(f"[cleanup] 清理了 {removed} 个过期临时文件")
    except Exception as e:
        logger.warning(f"[cleanup] 清理临时文件失败: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时开启飞书长连接 + 初始化反馈数据库 + 清理过期临时文件"""
    init_db()
    start_ws_client()
    _cleanup_old_temp_files()
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


@app.get("/upload", include_in_schema=False)
async def upload_page():
    """返回文件上传页面"""
    html_path = BASE_DIR / "static" / "upload.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="上传页面未找到，请检查 static/upload.html")
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


# ========== 文件上传（分步流程）==========

import uuid
import shutil

# 确保临时目录存在
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

@app.post(
    "/api/upload",
    summary="上传文件（步骤1）",
    description=(
        "上传文件到临时目录，返回文件ID和解析后的内容预览。\n\n"
        "支持的文件格式：.md, .pdf, .txt\n"
        "单个文件最大 10MB"
    ),
    operation_id="uploadFile",
    tags=["知识库"],
)
async def upload_file(file: UploadFile = File(...)):
    """上传文件到临时目录"""
    # 检查文件大小
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=400, detail=f"文件过大，最大支持 10MB")

    # 检查文件格式
    filename = file.filename or "unknown"
    ext = filename.lower().split('.')[-1]
    if ext not in ('md', 'pdf', 'txt'):
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: .{ext}")

    try:
        # 生成唯一文件ID
        file_id = str(uuid.uuid4())

        # 保存原始文件
        file_path = UPLOAD_TEMP_DIR / f"{file_id}.{ext}"
        with open(file_path, 'wb') as f:
            f.write(content)

        # 解析文件内容
        from app.document_loader import parse_uploaded_file
        text_content = parse_uploaded_file(filename, content)

        # 保存解析后的文本
        text_path = UPLOAD_TEMP_DIR / f"{file_id}.txt"
        text_path.write_text(text_content, encoding='utf-8')

        logger.info(f"[文件上传] 文件已保存: {file_id}, 原文件: {filename}, 大小: {len(content)} bytes")

        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "size": len(content),
            "content_preview": text_content[:1000],  # 前1000字符预览
            "content_length": len(text_content),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[文件上传] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")


@app.get(
    "/api/upload/{file_id}/preview",
    summary="预览文件内容（步骤2）",
    description="获取已上传文件的完整文本内容",
    operation_id="previewFile",
    tags=["知识库"],
)
async def preview_file(file_id: str):
    """预览文件完整内容"""
    text_path = UPLOAD_TEMP_DIR / f"{file_id}.txt"
    if not text_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在或已过期")

    try:
        content = text_path.read_text(encoding='utf-8')
        return {
            "success": True,
            "file_id": file_id,
            "content": content,
            "length": len(content),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChunkStrategyRequest(BaseModel):
    file_id: str = Field(..., description="文件ID")
    strategy: str = Field(..., description="分块策略：llm_suggest（LLM建议）或 custom（自定义正则）")
    custom_pattern: Optional[str] = Field(default=None, description="自定义正则表达式（strategy=custom时必填）")


@app.post(
    "/api/upload/chunk-strategy",
    summary="获取分块策略建议（步骤2）",
    description="使用LLM分析文件内容并建议分块策略，或验证用户自定义正则",
    operation_id="getChunkStrategy",
    tags=["知识库"],
)
async def get_chunk_strategy(req: ChunkStrategyRequest):
    """获取分块策略"""
    text_path = UPLOAD_TEMP_DIR / f"{req.file_id}.txt"
    if not text_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在或已过期")

    try:
        content = text_path.read_text(encoding='utf-8')

        if req.strategy == "llm_suggest":
            from app.config import CHAT_API_KEY, CHAT_BASE_URL, OPENAI_CHAT_MODEL, CHAT_API_MODE

            prompt = f"""你是一个文档分析专家。分析下面的文档，找出记录之间的**分隔符**，给出最简洁有效的分块策略。

文档内容（前3000字符）：
{content[:3000]}

## 第一步：识别分隔符

仔细观察文档中每条记录之间的分隔方式，常见模式：
- 固定分隔线：`\\n---\\n`、`\\n===\\n`、`\\n***\\n`
- 空行分隔：`\\n\\n`
- 标题行分隔：`\\n\\n# `、`\\n\\n## `、`\\n\\n### `

## 第二步：选择最简洁的 pattern

**核心原则：优先用最短、最简单的 pattern。**

- 如果记录之间有 `---`，直接用 `\\n---\\n`，不要加 lookahead
- 如果记录之间有空行+标题，用 `\\n\\n(?=## )` 或 `\\n\\n(?=### )`，不要加日期正则
- **禁止** 用复杂的非捕获组嵌套组合，例如 `(?:\\n---\\n\\n(?=...)|\\n(?=...))` 这类模式
- **禁止** 使用 `(?m)` `(?s)` `(?i)` 等内联修饰符
- **禁止** 使用 Python 专有 `(?P<name>...)` 命名捕获组
- pattern 必须能被 Python `re.split(pattern, text)` 切出 **2 条以上**记录

## 第三步：识别元数据字段

从每条记录中找可提取的结构化字段（日期、公司、负责人、金额等），提供提取正则。

以JSON格式返回（不要加代码块标记，不要加 ```json）：
{{
  "pattern": "分隔符正则（尽量简洁，如 \\\\n---\\\\n）",
  "description": "分块策略说明，包括每条记录的大致结构",
  "metadata_fields": [
    {{
      "field": "字段名（英文小写，date/company/owner/title/tags 直接存固定列，其他存动态列）",
      "label": "显示名称",
      "extract_regex": "提取正则（第一个捕获组为值，同时兼容 JS new RegExp() 和 Python re.search()）",
      "recommended": true,
      "reason": "建议理由"
    }}
  ]
}}"""

            base = CHAT_BASE_URL.rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"

            llm_headers = {
                "Authorization": f"Bearer {CHAT_API_KEY}",
                "Content-Type": "application/json",
            }
            if CHAT_API_MODE == "responses":
                llm_url = base + "/responses"
                llm_payload = {
                    "model": OPENAI_CHAT_MODEL,
                    "input": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_output_tokens": 1024,
                    "stream": True,
                }
            else:
                llm_url = base + "/chat/completions"
                llm_payload = {
                    "model": OPENAI_CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "stream": True,
                }

            async def _sse_stream():
                collected = ""
                try:
                    async with httpx.AsyncClient(timeout=60) as aclient:
                        async with aclient.stream("POST", llm_url, headers=llm_headers, json=llm_payload) as resp:
                            if resp.status_code != 200:
                                body = await resp.aread()
                                yield f"data: {json.dumps({'type': 'error', 'detail': f'API 错误 {resp.status_code}: {body.decode()[:200]}'}, ensure_ascii=False)}\n\n"
                                return
                            async for line in resp.aiter_lines():
                                line = line.strip()
                                if not line or not line.startswith("data:"):
                                    continue
                                data_str = line[5:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                except Exception:
                                    continue
                                token = None
                                if CHAT_API_MODE == "responses":
                                    if chunk.get("type") == "response.output_text.delta":
                                        token = chunk.get("delta", "")
                                    elif chunk.get("type") in ("response.completed", "response.done"):
                                        break
                                else:
                                    choices = chunk.get("choices", [])
                                    if choices:
                                        token = choices[0].get("delta", {}).get("content")
                                if token:
                                    collected += token
                                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

                    try:
                        suggestion_json = json.loads(collected.strip())
                    except Exception:
                        suggestion_json = {
                            "pattern": r"\n---\n",
                            "description": collected,
                            "metadata_fields": [
                                {"field": "date", "label": "日期", "extract_regex": ""},
                                {"field": "company", "label": "公司", "extract_regex": ""},
                                {"field": "owner", "label": "负责人", "extract_regex": ""},
                            ],
                        }
                    yield f"data: {json.dumps({'type': 'done', 'suggestion': suggestion_json}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    logger.error(f"[chunk-strategy stream] 失败: {e}", exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                _sse_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        elif req.strategy == "custom":
            if not req.custom_pattern:
                raise HTTPException(status_code=400, detail="自定义策略需要提供 custom_pattern")

            import re
            try:
                re.compile(req.custom_pattern)
            except re.error as e:
                raise HTTPException(status_code=400, detail=f"正则表达式无效: {e}")

            return {
                "success": True,
                "strategy": "custom",
                "pattern": req.custom_pattern,
            }

        else:
            raise HTTPException(status_code=400, detail="不支持的策略类型")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[分块策略] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------- 步骤3：按正则分块预览 ----------

class ChunkPreviewRequest(BaseModel):
    file_id: str = Field(..., description="文件ID")
    pattern: str = Field(..., description="分隔符正则表达式（用于 re.split）")
    max_preview: int = Field(default=10, ge=1, le=50, description="最多预览块数")


@app.post(
    "/api/upload/chunks-preview",
    summary="分块预览（步骤3）",
    description="按给定正则将文件分块，返回每块的摘要预览和总块数",
    operation_id="previewChunks",
    tags=["知识库"],
)
async def preview_chunks_endpoint(req: ChunkPreviewRequest):
    """应用正则分块策略，返回预览结果"""
    text_path = UPLOAD_TEMP_DIR / f"{req.file_id}.txt"
    if not text_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在或已过期")

    import re
    try:
        re.compile(req.pattern)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"正则表达式无效: {e}")

    try:
        content = text_path.read_text(encoding="utf-8")
        raw = [c.strip() for c in re.split(req.pattern, content) if c.strip()]
        return {
            "success": True,
            "total_chunks": len(raw),
            "preview": [
                {
                    "index": i,
                    "text": c[:500] + ("…" if len(c) > 500 else ""),
                    "full_length": len(c),
                }
                for i, c in enumerate(raw[: req.max_preview])
            ],
        }
    except Exception as e:
        logger.error(f"[分块预览] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------- 步骤4：确认元数据 + 入库 ----------

class MetadataFieldConfig(BaseModel):
    field: str = Field(..., description="字段名，建议英文小写（如 date/company/owner/title/tags 或自定义）")
    label: str = Field(default="", description="显示名称")
    mode: str = Field(..., description="global（全局固定值）或 regex（逐块提取）或 skip（跳过）")
    value: Optional[str] = Field(default=None, description="mode=global 时的固定值")
    extract_regex: Optional[str] = Field(default=None, description="mode=regex 时的提取正则（第一个捕获组）")


class ConfirmIndexRequest(BaseModel):
    file_id: str = Field(..., description="文件ID")
    filename: str = Field(..., description="原始文件名，用作 source 字段")
    pattern: str = Field(..., description="分块正则表达式")
    metadata_configs: List[MetadataFieldConfig] = Field(
        default_factory=list,
        description="元数据配置列表",
    )


@app.post(
    "/api/upload/confirm-index",
    summary="确认并入向量库（步骤4）",
    description=(
        "按分块正则将文件切分，为每块填充元数据（全局固定值或正则提取），"
        "向量化后增量写入 Milvus。不会清空已有数据。"
    ),
    operation_id="confirmIndex",
    tags=["知识库"],
)
async def confirm_index_endpoint(req: ConfirmIndexRequest):
    """分块 + 元数据填充 + 增量入向量库"""
    text_path = UPLOAD_TEMP_DIR / f"{req.file_id}.txt"
    if not text_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在或已过期")

    import re
    try:
        re.compile(req.pattern)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"分块正则无效: {e}")

    # 校验元数据正则
    for mc in req.metadata_configs:
        if mc.mode == "regex" and mc.extract_regex:
            try:
                re.compile(mc.extract_regex)
            except re.error as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"字段 {mc.field} 的提取正则无效: {e}",
                )

    try:
        content = text_path.read_text(encoding="utf-8")
        raw_chunks = [c.strip() for c in re.split(req.pattern, content) if c.strip()]

        if not raw_chunks:
            raise HTTPException(status_code=400, detail="分块结果为空，请检查正则表达式")

        chunks: List[Dict[str, Any]] = []
        for i, text in enumerate(raw_chunks):
            chunk: Dict[str, Any] = {
                "text": text,
                "source": req.filename,
                "chunk_id": hashlib.md5(text.encode("utf-8")).hexdigest()[:32],
                "type": "uploaded",
            }
            for mc in req.metadata_configs:
                if mc.mode == "skip":
                    continue
                if mc.mode == "global" and mc.value:
                    chunk[mc.field] = mc.value.strip()
                elif mc.mode == "regex" and mc.extract_regex:
                    m = re.search(mc.extract_regex, text)
                    if m:
                        chunk[mc.field] = (m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)).strip()
                    else:
                        chunk[mc.field] = ""
                else:
                    chunk[mc.field] = ""
            chunks.append(chunk)

        from app.vector_store import insert_chunks as _insert
        indexed = _insert(chunks)

        # 入库成功后清理临时文件
        for suffix in ('txt', 'md', 'pdf'):
            tmp = UPLOAD_TEMP_DIR / f"{req.file_id}.{suffix}"
            tmp.unlink(missing_ok=True)
        logger.info("[confirm-index] 文件 %s 入库 %d 块，临时文件已清理", req.filename, indexed)
        return {
            "success": True,
            "filename": req.filename,
            "chunk_count": len(chunks),
            "indexed_count": indexed,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[confirm-index] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== 上传配置助手（多轮对话，SSE 流式）==========

class AssistantMessage(BaseModel):
    role: str = Field(..., description="user 或 assistant")
    content: str

class AssistantRequest(BaseModel):
    file_id: str
    messages: List[AssistantMessage]
    current_pattern: str = Field(default="", description="当前分块正则")
    meta_fields: List[Dict[str, Any]] = Field(default_factory=list, description="当前元数据字段配置")
    step: int = Field(default=2, description="当前步骤 2=分块策略 3=元数据配置")

@app.post(
    "/api/upload/assistant",
    summary="配置助手（多轮流式对话）",
    description="帮助用户确定分块策略和元数据提取正则，返回 SSE 流。",
    operation_id="uploadAssistant",
    tags=["知识库"],
)
async def upload_assistant(req: AssistantRequest):
    """上传配置助手 — 基于文件内容的上下文感知对话"""
    text_path = UPLOAD_TEMP_DIR / f"{req.file_id}.txt"
    if not text_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在或已过期")

    from app.config import CHAT_API_KEY, CHAT_BASE_URL, OPENAI_CHAT_MODEL, CHAT_API_MODE

    content_sample = text_path.read_text(encoding="utf-8")[:3000]
    step_name = "分块策略" if req.step == 2 else "元数据配置"
    meta_summary = json.dumps(req.meta_fields, ensure_ascii=False, indent=None) if req.meta_fields else "（暂无）"

    system_prompt = f"""你是一个专业的文档处理配置助手，正在帮助用户配置【{step_name}】。

=== 文档样本（前3000字符）===
{content_sample}
==========================

=== 当前配置 ===
分块正则: {req.current_pattern or "（未设置）"}
元数据字段: {meta_summary}
================

你的职责：
1. 分析文档结构，建议最合适的配置方案
2. 当建议分块正则时，用如下格式输出（系统会识别并提供一键应用按钮）：
```chunk-pattern
\\n---\\n
```
3. 当建议字段提取正则时，用如下格式（字段名: 正则，一行一个）：
```field-regex
date: (\\d{{4}}-\\d{{2}}-\\d{{2}})
company: 客户[：:]\\s*(.+?)(?:\\\\n|$)
```
4. 保持回答简洁实用；正则用于 Python re.split()（分块）或 re.search()（字段提取），第一个捕获组为字段值"""

    # 构建消息列表
    if CHAT_API_MODE == "responses":
        # Responses API: system 消息用 user/assistant 轮换模拟
        llm_input = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "好的，我明白了，我会作为配置助手帮你分析文档结构，提供分块和元数据提取建议。"},
        ]
        for m in req.messages:
            llm_input.append({"role": m.role, "content": m.content})
        llm_url = CHAT_BASE_URL.rstrip("/")
        if not llm_url.endswith("/v1"):
            llm_url += "/v1"
        llm_url += "/responses"
        llm_payload = {
            "model": OPENAI_CHAT_MODEL,
            "input": llm_input,
            "temperature": 0.3,
            "max_output_tokens": 1500,
            "stream": True,
        }
    else:
        llm_messages = [{"role": "system", "content": system_prompt}]
        for m in req.messages:
            llm_messages.append({"role": m.role, "content": m.content})
        llm_url = CHAT_BASE_URL.rstrip("/")
        if not llm_url.endswith("/v1"):
            llm_url += "/v1"
        llm_url += "/chat/completions"
        llm_payload = {
            "model": OPENAI_CHAT_MODEL,
            "messages": llm_messages,
            "temperature": 0.3,
            "max_tokens": 1500,
            "stream": True,
        }

    llm_headers = {
        "Authorization": f"Bearer {CHAT_API_KEY}",
        "Content-Type": "application/json",
    }

    async def _stream():
        try:
            async with httpx.AsyncClient(timeout=60) as aclient:
                async with aclient.stream("POST", llm_url, headers=llm_headers, json=llm_payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"data: {json.dumps({'type': 'error', 'detail': f'API错误 {resp.status_code}: {body.decode()[:200]}'}, ensure_ascii=False)}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue
                        token = None
                        if CHAT_API_MODE == "responses":
                            if chunk.get("type") == "response.output_text.delta":
                                token = chunk.get("delta", "")
                            elif chunk.get("type") in ("response.completed", "response.done"):
                                break
                        else:
                            choices = chunk.get("choices", [])
                            if choices:
                                token = choices[0].get("delta", {}).get("content")
                        if token:
                            yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[assistant] 失败: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


# ========== MCP 数据源管理 ==========

class MCPSourceConfig(BaseModel):
    name: str = Field(..., description="数据源名称")
    type: str = Field(..., description="数据源类型：filesystem/http/database")
    enabled: bool = Field(default=True, description="是否启用")
    config: Dict[str, Any] = Field(default_factory=dict, description="数据源配置")


@app.get(
    "/api/mcp/sources",
    summary="获取 MCP 数据源列表",
    description="返回所有已配置的 MCP 数据源",
    operation_id="getMCPSources",
    tags=["MCP 数据源"],
)
async def get_mcp_sources():
    """获取 MCP 数据源列表"""
    try:
        config_path = BASE_DIR / "mcp_config.json"
        if not config_path.exists():
            return {"success": True, "sources": []}

        config = json.loads(config_path.read_text(encoding="utf-8"))
        sources = config.get("mcp_sources", [])
        return {"success": True, "sources": sources}
    except Exception as e:
        logger.error(f"[get_mcp_sources] 失败: {e}", exc_info=True)
        return {"success": False, "detail": str(e)}


@app.post(
    "/api/mcp/sources",
    summary="添加 MCP 数据源",
    description="添加新的 MCP 数据源配置",
    operation_id="addMCPSource",
    tags=["MCP 数据源"],
)
async def add_mcp_source(source: MCPSourceConfig):
    """添加 MCP 数据源"""
    try:
        config_path = BASE_DIR / "mcp_config.json"

        # 读取现有配置
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            config = {"mcp_sources": []}

        # 检查是否已存在同名数据源
        sources = config.get("mcp_sources", [])
        if any(s.get("name") == source.name for s in sources):
            raise HTTPException(status_code=400, detail=f"数据源 {source.name} 已存在")

        # 添加新数据源
        new_source = {
            "name": source.name,
            "type": source.type,
            "enabled": source.enabled,
            **source.config
        }
        sources.append(new_source)
        config["mcp_sources"] = sources

        # 保存配置
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        return {"success": True, "message": f"数据源 {source.name} 添加成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[add_mcp_source] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put(
    "/api/mcp/sources/{source_name}",
    summary="更新 MCP 数据源",
    description="更新指定的 MCP 数据源配置",
    operation_id="updateMCPSource",
    tags=["MCP 数据源"],
)
async def update_mcp_source(source_name: str, source: MCPSourceConfig):
    """更新 MCP 数据源"""
    try:
        config_path = BASE_DIR / "mcp_config.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="配置文件不存在")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        sources = config.get("mcp_sources", [])

        # 查找并更新数据源
        found = False
        for i, s in enumerate(sources):
            if s.get("name") == source_name:
                sources[i] = {
                    "name": source.name,
                    "type": source.type,
                    "enabled": source.enabled,
                    **source.config
                }
                found = True
                break

        if not found:
            raise HTTPException(status_code=404, detail=f"数据源 {source_name} 不存在")

        config["mcp_sources"] = sources
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        return {"success": True, "message": f"数据源 {source_name} 更新成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[update_mcp_source] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/api/mcp/sources/{source_name}",
    summary="删除 MCP 数据源",
    description="删除指定的 MCP 数据源",
    operation_id="deleteMCPSource",
    tags=["MCP 数据源"],
)
async def delete_mcp_source(source_name: str):
    """删除 MCP 数据源"""
    try:
        config_path = BASE_DIR / "mcp_config.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="配置文件不存在")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        sources = config.get("mcp_sources", [])

        # 过滤掉要删除的数据源
        new_sources = [s for s in sources if s.get("name") != source_name]

        if len(new_sources) == len(sources):
            raise HTTPException(status_code=404, detail=f"数据源 {source_name} 不存在")

        config["mcp_sources"] = new_sources
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        return {"success": True, "message": f"数据源 {source_name} 删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[delete_mcp_source] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/mcp/sources/{source_name}/test",
    summary="测试 MCP 数据源连接",
    description="测试指定数据源的连接是否正常",
    operation_id="testMCPSource",
    tags=["MCP 数据源"],
)
async def test_mcp_source(source_name: str):
    """测试 MCP 数据源连接"""
    try:
        config_path = BASE_DIR / "mcp_config.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="配置文件不存在")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        sources = config.get("mcp_sources", [])

        # 查找数据源
        source_config = None
        for s in sources:
            if s.get("name") == source_name:
                source_config = s
                break

        if not source_config:
            raise HTTPException(status_code=404, detail=f"数据源 {source_name} 不存在")

        # 测试连接
        from app.mcp_loader import MCP_SOURCE_TYPES
        source_type = source_config.get("type")
        source_class = MCP_SOURCE_TYPES.get(source_type)

        if not source_class:
            raise HTTPException(status_code=400, detail=f"不支持的数据源类型: {source_type}")

        source = source_class(source_name, source_config)
        docs = source.fetch_data()

        return {
            "success": True,
            "message": "连接测试成功",
            "doc_count": len(docs),
            "sample": docs[0] if docs else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[test_mcp_source] 失败: {e}", exc_info=True)
        return {"success": False, "detail": str(e)}


# ========== MCP 协议支持（服务发现）==========

@app.post(
    "/api/mcp/discover",
    summary="MCP 服务发现",
    description="连接到 MCP 服务器并发现可用资源（支持 HTTP REST 和 SSE 传输）",
    operation_id="discoverMCPResources",
    tags=["MCP 数据源"],
)
async def discover_mcp_resources(request: Dict[str, str]):
    """MCP 服务发现"""
    try:
        base_url = request.get("base_url", "")
        if not base_url:
            raise HTTPException(status_code=400, detail="base_url 不能为空")

        # 尝试 SSE 传输（标准 MCP-over-SSE）
        try:
            from app.mcp_sse_async_client import MCPSSEAsyncClient
            async with MCPSSEAsyncClient(base_url) as client:
                resources = await client.discover_resources()
                if resources:
                    logger.info(f"[MCP] 使用 SSE 传输成功连接到 {base_url}")
                    return {
                        "success": True,
                        "resources": resources,
                        "server_url": base_url,
                        "transport": "sse"
                    }
        except Exception as sse_error:
            logger.warning(f"[MCP] SSE 传输失败，尝试 HTTP REST: {sse_error}")

        # 回退到 HTTP REST API
        from app.mcp_client import MCPClient
        client = MCPClient(base_url)
        try:
            resources = client.discover_resources()
            logger.info(f"[MCP] 使用 HTTP REST 成功连接到 {base_url}")
            return {
                "success": True,
                "resources": resources,
                "server_url": base_url,
                "transport": "http"
            }
        finally:
            client.close()

    except Exception as e:
        logger.error(f"[discover_mcp] 失败: {e}", exc_info=True)
        return {"success": False, "detail": str(e)}


@app.post(
    "/api/mcp/preview",
    summary="预览 MCP 资源数据",
    description="获取资源的样本数据和分块建议（支持 HTTP REST 和 SSE 传输）",
    operation_id="previewMCPResource",
    tags=["MCP 数据源"],
)
async def preview_mcp_resource(request: Dict[str, Any]):
    """预览 MCP 资源数据"""
    try:
        base_url = request.get("base_url", "")
        resource_uri = request.get("resource_uri", "")
        filters = request.get("filters", {})

        if not base_url or not resource_uri:
            raise HTTPException(status_code=400, detail="base_url 和 resource_uri 不能为空")

        # 尝试 SSE 传输
        try:
            from app.mcp_sse_async_client import MCPSSEAsyncClient
            async with MCPSSEAsyncClient(base_url) as client:
                result = await client.read_resource(resource_uri, filters)
                contents = result.get("contents", [])[:10]

                # 生成分块建议
                chunking_suggestion = client.suggest_chunking(contents, {})

                return {
                    "success": True,
                    "sample_data": contents,
                    "schema": {},
                    "chunking_suggestion": chunking_suggestion,
                    "total_count": len(result.get("contents", []))
                }
        except Exception as sse_error:
            logger.warning(f"[MCP] SSE 传输失败，尝试 HTTP REST: {sse_error}")

        # 回退到 HTTP REST
        from app.mcp_client import MCPClient
        client = MCPClient(base_url)

        try:
            # 读取样本数据（限制数量）
            result = client.read_resource(resource_uri, filters)
            contents = result.get("contents", [])[:10]  # 只取前10条作为样本

            # 获取 Schema
            schema = client.get_resource_schema(resource_uri)

            # 生成分块建议
            chunking_suggestion = client.suggest_chunking(contents, schema)

            return {
                "success": True,
                "sample_data": contents,
                "schema": schema,
                "chunking_suggestion": chunking_suggestion,
                "total_count": len(result.get("contents", []))
            }
        finally:
            client.close()

    except Exception as e:
        logger.error(f"[preview_mcp] 失败: {e}", exc_info=True)
        return {"success": False, "detail": str(e)}


@app.post(
    "/api/mcp/discover/tools",
    summary="发现 MCP Tools",
    description="连接到 MCP 服务器并发现可用的工具（支持 SSE 传输）",
    operation_id="discoverMCPTools",
    tags=["MCP 数据源"],
)
async def discover_mcp_tools(request: Dict[str, str]):
    """发现 MCP 服务器提供的工具"""
    try:
        base_url = request.get("base_url", "")
        if not base_url:
            raise HTTPException(status_code=400, detail="base_url 不能为空")

        # 使用异步 SSE 客户端
        try:
            from app.mcp_sse_async_client import MCPSSEAsyncClient
            async with MCPSSEAsyncClient(base_url) as client:
                tools = await client.list_tools()
                logger.info(f"[MCP] 使用 SSE 传输发现 {len(tools)} 个工具")
                return {
                    "success": True,
                    "tools": tools,
                    "server_url": base_url,
                    "transport": "sse"
                }
        except Exception as sse_error:
            logger.error(f"[MCP] SSE 传输失败: {sse_error}")
            return {"success": False, "detail": str(sse_error)}

    except Exception as e:
        logger.error(f"[discover_mcp_tools] 失败: {e}", exc_info=True)
        return {"success": False, "detail": str(e)}


@app.post(
    "/api/mcp/call/tool",
    summary="调用 MCP Tool",
    description="调用 MCP 服务器的工具并获取数据",
    operation_id="callMCPTool",
    tags=["MCP 数据源"],
)
async def call_mcp_tool(request: Dict[str, Any]):
    """调用 MCP 工具"""
    try:
        base_url = request.get("base_url", "")
        tool_name = request.get("tool_name", "")
        arguments = request.get("arguments", {})

        if not base_url or not tool_name:
            raise HTTPException(status_code=400, detail="base_url 和 tool_name 不能为空")

        # 使用 SSE 传输
        from app.mcp_sse_async_client import MCPSSEAsyncClient
        async with MCPSSEAsyncClient(base_url) as client:
            result = await client.call_tool(tool_name, arguments)

            # 提取内容
            contents = []
            if isinstance(result, dict):
                if "content" in result:
                    # MCP 标准响应格式
                    content_list = result["content"]
                    if isinstance(content_list, list):
                        for item in content_list:
                            if isinstance(item, dict) and "text" in item:
                                contents.append({"text": item["text"]})
                            else:
                                contents.append({"text": str(item)})
                    else:
                        contents.append({"text": str(content_list)})
                else:
                    # 直接返回的数据
                    contents.append({"text": str(result)})
            elif isinstance(result, list):
                contents = [{"text": str(item)} for item in result]
            else:
                contents.append({"text": str(result)})

            logger.info(f"[MCP] 调用工具 {tool_name} 成功，获取 {len(contents)} 条数据")

            return {
                "success": True,
                "data": contents[:10],  # 只返回前10条作为预览
                "total_count": len(contents),
                "raw_result": result
            }

    except Exception as e:
        logger.error(f"[call_mcp_tool] 失败: {e}", exc_info=True)
        return {"success": False, "detail": str(e)}


# ========== 后台管理 ==========

@app.get("/admin", include_in_schema=False)
async def admin_page():
    """返回后台管理页面"""
    html_path = BASE_DIR / "static" / "admin.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="管理页面未找到，请检查 static/admin.html")
    return FileResponse(str(html_path))


class AdminClearRequest(BaseModel):
    confirm: bool = Field(default=False, description="必须设置为 true 才会执行清空")


@app.post(
    "/api/admin/clear",
    summary="清空向量库",
    description="删除 Milvus Collection，彻底清空所有向量数据。必须将 confirm 设为 true。",
    operation_id="adminClearIndex",
    tags=["知识库"],
)
async def admin_clear_index(req: AdminClearRequest):
    """清空向量库（不重建）"""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="请设置 confirm=true 以确认清空操作")
    try:
        dropped = clear_index()
        return {
            "success": True,
            "dropped": dropped,
            "message": "向量库已清空" if dropped else "向量库本就为空，无需操作",
        }
    except Exception as e:
        logger.error(f"[admin_clear_index] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/admin/rebuild/stream",
    summary="流式重建知识库索引（SSE）",
    description=(
        "以 Server-Sent Events 流式方式实时推送重建进度。\n\n"
        "事件格式：`data: {\"type\": \"...\", ...}`\n"
        "- `type=log`：进度日志\n"
        "- `type=done`：完成，附带 chunk_count / indexed_count\n"
        "- `type=error`：错误信息"
    ),
    operation_id="adminRebuildStream",
    tags=["知识库"],
)
async def admin_rebuild_stream(req: IndexRequest):
    """流式重建向量库（SSE 实时进度）"""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="请设置 confirm=true 以确认重建操作")

    import asyncio
    import concurrent.futures
    from io import StringIO

    async def _stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _log(msg: str):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "message": msg})

        def _run_rebuild():
            # 重定向 print 到日志队列
            import builtins
            original_print = builtins.print

            def patched_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                _log(msg)
                original_print(*args, **kwargs)

            builtins.print = patched_print
            try:
                chunks = load_and_split()
                _log(f"[分块] 共切分 {len(chunks)} 个片段")
                count = build_index(chunks)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "done", "chunk_count": len(chunks), "indexed_count": count},
                )
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "detail": str(e)})
            finally:
                builtins.print = original_print

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, _run_rebuild)

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=120)
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                if item["type"] in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'detail': '操作超时'}, ensure_ascii=False)}\n\n"
                break

        executor.shutdown(wait=False)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
