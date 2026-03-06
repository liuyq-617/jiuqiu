# CRM 知识库问答系统 — 修改记录

本文档记录对原始版本所做的全部改动，按问题分类整理，包含根本原因分析、解决方案与关键代码片段。

---

## 目录

1. [Milvus VARCHAR 字节溢出修复](#1-milvus-varchar-字节溢出修复)
2. [Chat API 流式响应兼容](#2-chat-api-流式响应兼容)
3. [日志系统增强](#3-日志系统增强)
4. [OpenAPI 规范化（OpenClaw 接入）](#4-openapi-规范化openclaw-接入)
5. [聚合查询修复](#5-聚合查询修复)
6. [元数据精确过滤路由](#6-元数据精确过滤路由)
7. [聚合关键词量词变体正则修复](#7-聚合关键词量词变体正则修复)
8. [开源前脱敏处理](#8-开源前脱敏处理)
9. [后续优化方向](#9-后续优化方向)
3. [日志系统增强](#3-日志系统增强)
4. [OpenAPI 规范化（OpenClaw 接入）](#4-openapi-规范化openclaw-接入)
5. [聚合查询修复](#5-聚合查询修复)
6. [元数据精确过滤路由](#6-元数据精确过滤路由)
7. [聚合关键词量词变体正则修复](#7-聚合关键词量词变体正则修复)

---

## 1. Milvus VARCHAR 字节溢出修复

**文件**: `app/vector_store.py`

### 问题

```
the length (8286) of 7386th VarChar text exceeds max length (8192)
```

Milvus 的 VARCHAR 字段长度限制按 **UTF-8 字节数** 计算，而原代码按 Python 字符数截断。中文字符每个占 3 字节，截断后实际字节数可能仍超出上限。

### 解决方案

新增 `truncate_to_bytes()` 函数，逐字符拼接并检查 UTF-8 字节长度：

```python
MAX_BYTES = 8000  # 留出余量，Milvus 上限 8192

def truncate_to_bytes(s: str, max_bytes: int = MAX_BYTES) -> str:
    """按 UTF-8 字节数截断字符串，避免 Milvus VARCHAR 溢出。"""
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    # 截断字节后解码（忽略不完整的多字节字符）
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
```

在写入向量库前对 `text` 字段调用该函数。

---

## 2. Chat API 流式响应兼容

**文件**: `app/rag.py`, `app/config.py`

### 问题

前端展示了参考来源后，回答区域始终为空。  
根本原因：`ai.qaq.al` 使用 OpenAI **Responses API**（`/v1/responses`），而非传统的 Chat Completions API（`/v1/chat/completions`）。两者的请求体格式与 SSE 事件类型均不同，OpenAI SDK 静默失败。

### 两种 API 格式对比

| 项目 | Completions API | Responses API |
|------|----------------|---------------|
| 端点 | `/v1/chat/completions` | `/v1/responses` |
| 消息字段 | `messages: [{role, content}]` | `input` + `instructions` |
| SSE 增量事件 | `choices[0].delta.content` | `response.output_text.delta` |
| 结束标志 | `[DONE]` | `response.completed` |

### 解决方案

1. 新增配置项 `CHAT_API_MODE`（`config.py`）：

```python
CHAT_API_MODE = os.getenv("CHAT_API_MODE", "responses")  # "responses" | "completions"
```

2. 用 `httpx` 直接处理 SSE，绕过 SDK（`rag.py`）：

```python
def _chat_url() -> str:
    base = config.OPENAI_BASE_URL.rstrip("/")
    if config.CHAT_API_MODE == "responses":
        return f"{base}/responses"
    return f"{base}/chat/completions"

def _build_payload(messages: list[dict], stream: bool) -> dict:
    if config.CHAT_API_MODE == "responses":
        # 将 system 消息转为 instructions，其余转为 input
        instructions = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_content = "\n".join(m["content"] for m in messages if m["role"] != "system")
        return {
            "model": config.OPENAI_CHAT_MODEL,
            "instructions": instructions,
            "input": user_content,
            "stream": stream,
        }
    # completions 模式
    return {
        "model": config.OPENAI_CHAT_MODEL,
        "messages": messages,
        "stream": stream,
    }
```

3. SSE 事件解析（流式）：

```python
# responses 模式
if event_type == "response.output_text.delta":
    yield data.get("delta", "")
elif event_type == "response.completed":
    break

# completions 模式
if chunk == "[DONE]":
    break
delta = json_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
yield delta
```

---

## 3. 日志系统增强

**文件**: `app/main.py`, `app/rag.py`

### 改动

- `main.py` 中配置全局日志级别为 `INFO`：

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
```

- `rag.py` 中使用独立 logger，记录路由决策、检索结果、Token 数量等关键信息：

```python
logger = logging.getLogger("crm_rag")

# 示例日志输出
logger.info("问题路由：聚合统计")
logger.info("问题路由：元数据过滤 owner=%s date=%s~%s", owner, date_from, date_to)
logger.info("问题路由：向量语义检索")
logger.info("检索到 %d 条记录", len(docs))
```

---

## 4. OpenAPI 规范化（OpenClaw 接入）

**文件**: `app/main.py`

### 改动目标

使 `/openapi.json` 符合 OpenClaw 的导入要求，所有接口具备清晰的元数据。

### 关键改动

1. **FastAPI 应用元数据**：

```python
app = FastAPI(
    title="CRM 知识库问答系统",
    description="基于 Milvus + RAG 的 CRM 知识库智能问答 API",
    version="1.0.0",
    servers=[{"url": "http://localhost:8000", "description": "本地开发服务器"}],
    openapi_tags=[
        {"name": "问答", "description": "RAG 问答相关接口"},
        {"name": "知识库", "description": "知识库管理接口"},
    ],
)
```

2. **Pydantic 响应模型**（新增）：

```python
class HealthResponse(BaseModel): ...
class ChatRequest(BaseModel): ...
class ChatResponse(BaseModel): ...
class SourceItem(BaseModel): ...
class StatsResponse(BaseModel): ...
class AggregateStatsResponse(BaseModel): ...
class IndexRequest(BaseModel): ...
class IndexResponse(BaseModel): ...
```

3. **每个路由添加 `operationId`**：

| operationId | 方法 | 路径 |
|---|---|---|
| `healthCheck` | GET | `/api/health` |
| `chat` | POST | `/api/chat` |
| `chatStream` | POST | `/api/chat/stream` |
| `rebuildIndex` | POST | `/api/index` |
| `getStats` | GET | `/api/stats` |
| `listCompanies` | GET | `/api/companies` |

4. **新增 `/api/companies` 端点**：返回全量客户和负责人列表，供 OpenClaw Agent 使用。

### OpenClaw 接入方式

1. 确保服务运行于 `http://localhost:8000`（或公网可访问地址）
2. 在 OpenClaw 中导入 OpenAPI 描述文件：`http://localhost:8000/openapi.json`
3. 所有接口将自动注册为 Actions

---

## 5. 聚合查询修复

**文件**: `app/vector_store.py`, `app/rag.py`

### 问题

"活动记录中涵盖了多少客户？" 返回 4 个客户（实际有 1147 个）。

根本原因：所有查询走向量检索，`top_k=5` 仅返回最相关的 5 条片段，片段中涉及的公司名称自然只有几个。

### 解决方案

在 `vector_store.py` 新增全量扫描函数：

```python
def get_distinct_values(field: str) -> list[str]:
    """分页遍历 Milvus 全量记录，返回某字段的去重值列表。"""
    PAGE = 1000
    offset = 0
    seen = set()
    while True:
        res = collection.query(
            expr="id >= 0",
            output_fields=[field],
            limit=PAGE,
            offset=offset,
        )
        if not res:
            break
        for row in res:
            val = row.get(field, "")
            if val:
                seen.add(val)
        offset += PAGE
    return sorted(seen)

def get_aggregate_stats() -> dict:
    """返回知识库整体统计信息（~0.7s）。"""
    companies = get_distinct_values("company")
    owners = get_distinct_values("owner")
    total = collection.num_entities
    return {
        "total_chunks": total,
        "company_count": len(companies),
        "companies": companies,
        "owner_count": len(owners),
        "owners": owners,
    }
```

在 `rag.py` 中新增 `_build_aggregate_context()`，将统计结果格式化为 Prompt 上下文后由 LLM 生成自然语言回答。

---

## 6. 元数据精确过滤路由

**文件**: `app/vector_store.py`, `app/rag.py`

### 问题

"刘溢清上周的活动记录" 返回的结果与刘溢清无关（实际返回了杨晨的记录）。

根本原因：向量相似度只考虑语义内容，完全忽略 `owner`（负责人）和 `date` 字段。

### 解决方案

#### `vector_store.py` — 新增 `query_by_metadata()`

```python
def query_by_metadata(
    owner: str | None = None,
    date_from: str | None = None,   # "YYYY-MM-DD"
    date_to: str | None = None,     # "YYYY-MM-DD"
    limit: int = 200,
) -> list[dict]:
    """
    使用 Milvus expr 精确过滤，支持：
    - owner like '%name%'
    - date >= "YYYY-MM-DD" and date <= "YYYY-MM-DD"
    """
    exprs = []
    if owner:
        exprs.append(f'owner like "%{owner}%"')
    if date_from:
        exprs.append(f'date >= "{date_from}"')
    if date_to:
        exprs.append(f'date <= "{date_to}"')
    expr = " and ".join(exprs) if exprs else "id >= 0"

    res = collection.query(
        expr=expr,
        output_fields=["text", "company", "owner", "date", "source"],
        limit=limit,
    )
    # 按日期降序排列
    return sorted(res, key=lambda r: r.get("date", ""), reverse=True)
```

#### `rag.py` — 日期范围解析

支持的时间表达式（`_parse_date_range()`）：

| 查询词 | 解析结果 |
|---|---|
| 今天 | 当天 |
| 昨天 | 前一天 |
| 本周 | 本周一 ~ 本周日 |
| 上周 | 上周一 ~ 上周日 |
| 本月 | 当月 1 日 ~ 末日 |
| 上月 | 上月 1 日 ~ 末日 |
| `3月份` / `2026年3月` | 指定年月 |
| 最近 N 天 | 今天往前 N 天 |
| 最近 N 周 | 今天往前 N×7 天 |

#### `rag.py` — 负责人识别

```python
_owners_cache: list[str] = []  # 从 Milvus 动态加载，lazy init

def _extract_owner(question: str) -> str | None:
    """
    对比已知负责人列表（37 位），
    支持中文名/英文名部分匹配，取最长匹配项。
    """
    global _owners_cache
    if not _owners_cache:
        stats = get_aggregate_stats()
        _owners_cache = stats["owners"]
    
    matched = [o for o in _owners_cache if o in question or question in o]
    return max(matched, key=len) if matched else None
```

#### `rag.py` — 三路查询路由

```
用户问题
    │
    ├─ _is_aggregate_question()  ──→  get_aggregate_stats()
    │   例："有多少个客户"              全量遍历，~0.7s
    │
    ├─ extract_filters() 有结果  ──→  query_by_metadata()
    │   例："刘溢清上周的记录"          Milvus expr 精确过滤
    │
    └─ 其他                      ──→  search()
        例："最近客户有什么进展"         向量语义检索，top_k=5
```

---

## 7. 聚合关键词量词变体正则修复

**文件**: `app/rag.py`

### 问题

"系统里有多少**个**客户" 返回 3 个，"有多少**家**客户" 返回 2 个。

根本原因：原代码使用关键词列表精确匹配，`"多少客户"` 不能匹配含量词的变体（`多少个/家/位/条` 客户）。

### 解决方案

将关键词列表替换为正则表达式，量词设为可选：

```python
_AGGREGATE_PATTERNS = [
    r'多少[个家位条]?客户',
    r'多少[个家]?公司',
    r'[哪所全]+(些|有|部)?客户',
    r'[哪所全]+(些|有|部)?公司',
    r'客户[列名][单表]?|客户有哪些|包含哪些客户|涵盖.{0,4}客户',
    r'公司[列名][单表]?|公司有哪些',
    r'[所全]+(有|部)?负责人|负责人[列名][单表]?|负责人有哪些',
    r'多少[个位名]?(负责人|销售|同事|员工)',
    r'[哪所全]+(些|有|部)?(销售|负责人)',
    r'销售[列名][单表]?|销售人员',
]
_AGGREGATE_RE = re.compile('|'.join(_AGGREGATE_PATTERNS))

def _is_aggregate_question(question: str) -> bool:
    # 若含具体人名，优先走元数据过滤路由
    if _extract_owner(question):
        return False
    return bool(_AGGREGATE_RE.search(question))
```

### 验证测试

15 个测试用例全部通过：

```
✅ 系统里有多少个客户      → 聚合路由
✅ 有多少个客户            → 聚合路由
✅ 有多少家客户            → 聚合路由
✅ 有多少位客户            → 聚合路由
✅ 有多少条客户            → 聚合路由
✅ 有多少客户              → 聚合路由
✅ 活动记录涵盖多少个客户  → 聚合路由
✅ 公司列表                → 聚合路由
✅ 所有负责人              → 聚合路由
✅ 负责人列表              → 聚合路由
✅ 有多少位销售            → 聚合路由
✅ 销售人员名单            → 聚合路由
✅ 刘溢清上周              → 元数据过滤路由（含人名，不走聚合）
✅ 最近客户进展            → 向量检索路由
✅ 帮我分析一下            → 向量检索路由
```

---

## 当前知识库规模

| 指标 | 数值 |
|---|---|
| 总向量记录数 | 7,927 条 |
| 覆盖客户数 | 1,147 家 |
| 负责人数量 | 37 位 |
| 聚合查询耗时 | ~0.7 秒 |

---

## 8. 开源前脱敏处理

**文件**: `app/config.py`, `.gitignore`, `.env.example`

为了将项目安全开源，进行了以下脱敏处理：

### 8.1 移除硬编码密钥

`config.py` 中原来将真实 API Key 作为环境变量的默认值，已全部替换为空字符串：

| 配置项 | 处理前（泄露风险） | 处理后 |
|---|---|---|
| `CHAT_API_KEY` | 硬编码 `sk-ac50...40a08` | `""` （必须通过 `.env` 设置） |
| `CHAT_BASE_URL` | `https://ai.qaq.al` (私有服务) | `https://api.openai.com/v1` |
| `OPENAI_CHAT_MODEL` | `gpt-5.4` | `gpt-4o-mini` |
| `EMBEDDING_API_KEY` | 硬编码 `sk-Vcbd...sNH8` | `""` （必须通过 `.env` 设置） |
| `EMBEDDING_BASE_URL` | `https://router.tumuer.me/` (私有服务) | `https://api.openai.com/v1` |

### 8.2 新增 `.gitignore`

防止敏感文件被提交到仓库：

```gitignore
# 环境变量（含真实密钥）
.env
.env.*
!.env.example

# 业务数据
data/embedding_cache.db
data/chunks_preview.txt
volumes/

# CRM 文档（敏感商业数据）
*.md
!README.md
!docs/*.md

# Python 缓存
__pycache__/
```

### 8.3 开源清单

开源前确认以下事项：

- [x] `config.py` 无硬编码 API Key
- [x] `config.py` 无私有服务地址
- [x] `.env` 被 `.gitignore` 排除
- [x] `.env.example` 仅含占位符
- [x] `volumes/` 数据卷被排除
- [x] `data/*.db` 缓存被排除
- [x] CRM Markdown 业务文档被排除
- [ ] 确认 git history 中无密钥泄露（如已提交过，需用 `git filter-branch` 或 `BFG Repo Cleaner` 清除）

---

## 9. 后续优化方向

### 9.1 回答质量评价体系

引入自动化评价机制，持续量化和改进回答质量。

- **用户反馈收集**: 前端添加“赞/踩”按钮，将问答对 + 评价存入数据库
- **LLM-as-Judge**: 用 GPT-4o 自动评分（相关性 / 完整性 / 准确性），与人工评分对比
- **基准测试集**: 维护一组标准 Q&A，每次修改后自动回归测试，防止质量下降
- **指标看板**: 跟踪准确率 / 响应时间 / 用户满意度等趋势

### 9.2 检索增强 (Advanced RAG)

当前单路向量检索 + top_k 方式存在召回率不足的问题。

- **查询改写 (Query Rewriting)**: 用 LLM 将用户问题改写为更适合检索的形式，或拆解为多个子查询
- **混合检索 (Hybrid Search)**: 结合向量检索 + BM25 关键词检索，Milvus 2.4 已原生支持
- **重排序 (Reranking)**: 检索后用 Cross-Encoder（如 `bge-reranker`）对结果二次排序，提高精度
- **父文档检索 (Parent Document Retrieval)**: 小块检索、大块返回，兆合精准检索和充分上下文

### 9.3 多轮对话与上下文记忆

当前每次问答独立，无法处理连续追问。

- **对话历史管理**: 维护 session 级别的对话历史，将近 N 轮对话作为上下文传入 LLM
- **指代消解**: 在检索前将“他们”“这个客户”等指代词解析为具体实体
- **对话摘要**: 较长对话自动压缩历史上下文，避免 token 溢出

### 9.4 分块策略优化

当前按固定 `---` 分隔符 + 字数上限切块，未考虑语义完整性。

- **语义分块**: 按 CRM 活动记录的结构化字段（客户名 / 负责人 / 日期 / 内容）智能切分
- **元数据丰富化**: 提取更多结构化字段（行业 / 商机阶段 / 金额）存入 Milvus，支持更精细的过滤
- **动态块大小**: 根据内容长度自动调整，短活动记录不拆分，长记录按语义边界拆分

### 9.5 权限与多租户

当前无身份认证，所有用户看到相同数据。

- **数据隔离**: 向量库按团队/角色分区，查询时自动注入 `owner` 过滤，仅返回当前用户可见的记录
- **API 认证**: 接入 API Key / JWT / OAuth2，保护接口安全
- **操作审计**: 记录每次查询的用户、问题、检索范围，便于合规审计

### 9.6 增量索引与实时同步

当前每次更新需要重建全量索引。

- **增量写入**: 检测新增/修改的文档，仅对变更部分重新分块和向量化
- **Webhook 触发**: CRM 系统新增活动记录时通过 Webhook 自动更新向量库
- **删除/更新同步**: 当源文档修改或删除时，自动清理对应向量

### 9.7 可观测性与监控

- **链路追踪 (Tracing)**: 接入 LangSmith / Langfuse，记录每次请求的完整链路（问题分类 → 检索 → Prompt 构建 → LLM 调用）
- **Token 用量统计**: 按用户/时段统计 Embedding 和 Chat 的 token 消耗，控制成本
- **异常告警**: 响应时间超阈值 / 检索结果为空 / API 调用失败时自动告警

### 9.8 前端体验优化

- **Markdown 渲染**: 回答中的表格 / 代码块 / 列表正确渲染，而非纯文本展示
- **来源可视化**: 点击来源可展开原文片段，关键词高亮
- **导出功能**: 支持将问答结果导出为 PDF / Markdown
- **移动端适配**: 响应式布局，支持移动端访问

### 9.9 知识库扩展

- **多数据源接入**: 除 Markdown 外，支持从 CRM 数据库 / 飞书 / 钉钉 / 企微直接拉取数据
- **文档类型扩展**: 支持 PDF / Word / Excel / 图片 OCR 等格式
- **知识图谱**: 从活动记录中提取客户-人员-事件关系，结合图数据库支持复杂关系查询

---

## 环境变量补充

在原有变量基础上新增：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHAT_API_MODE` | `responses` | API 模式：`responses`（新版）或 `completions`（传统） |
