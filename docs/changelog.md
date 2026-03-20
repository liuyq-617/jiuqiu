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
9. [飞书 WebSocket 长连接机器人](#9-飞书-websocket-长连接机器人)
10. [回答质量评价体系](#10-回答质量评价体系)
11. [后续优化方向](#11-后续优化方向)
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

## 9. 飞书 WebSocket 长连接机器人

**文件**: `app/feishu_bot.py`, `app/main.py`, `app/config.py`

### 功能

使用飞书官方 `lark-oapi` SDK 的 WebSocket 长连接模式接入飞书机器人：

- 无需公网域名或 HTTPS 证书
- 应用启动时自动建立出站 WebSocket 连接，断线自动重连
- Card Kit 流式卡片回复，用户实时看到 token 逐字输出

### 延迟优化

`lark_oapi` SDK 默认断线重连有 0~30s 随机等待（`_reconnect_nonce=30`），通过子类化覆盖：

```python
class _FastWSClient(lark.ws.Client):
    def _configure(self, conf):
        super()._configure(conf)
        self._reconnect_nonce    = 0    # 断线立即重连
        self._reconnect_interval = 3    # 重试间隔 3s
        self._ping_interval      = 600  # 心跳 10 分钟（节省 API 调用）
```

重连延迟从 0~30s 降低到 <1s。

---

## 10. 回答质量评价体系

**文件**: `app/feedback.py`（新增）、`app/main.py`、`static/index.html`、`scripts/benchmark.py`（新增）

### 10.1 系统架构

```
用户问答
  │
  ├─ 写入 feedback.db (SQLite)
  │    qa_log: question / answer / sources / response_ms / created_at
  │
  ├─ 前端显示 👍 / 👎 按钮 (type=answer_id 事件触发)
  │    └─ POST /api/feedback → 更新 thumbs 字段 + 异步触发 LLM 评分
  │
  └─ LLM-as-Judge (后台线程)
       │  相关性 / 完整性 / 准确性  1.0~5.0 分
       └─ 写回 llm_relevance / llm_completeness / llm_accuracy

GET /api/feedback/stats → 汇总指标 + 近 30 条明细
```

### 10.2 数据库表结构

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | TEXT (UUID) | 主键，即 answer_id |
| `question` | TEXT | 用户问题 |
| `answer` | TEXT | 完整回答 |
| `sources_json` | TEXT | 来源列表 JSON |
| `response_ms` | INTEGER | 端到端响应毫秒数 |
| `created_at` | TEXT | 创建时间 |
| `thumbs` | INTEGER | 1=赞 / -1=踩 / NULL=未评 |
| `llm_relevance` | REAL | 相关性 1.0~5.0 |
| `llm_completeness` | REAL | 完整性 1.0~5.0 |
| `llm_accuracy` | REAL | 准确性 1.0~5.0 |
| `llm_comment` | TEXT | LLM 评语 |
| `llm_judged_at` | TEXT | 评分时间 |

### 10.3 新增 API 端点

| operationId | 方法 | 路径 | 说明 |
|---|---|---|---|
| `submitFeedback` | POST | `/api/feedback` | 提交赞/踩，可选触发 LLM 评分 |
| `getFeedbackStats` | GET | `/api/feedback/stats` | 看板汇总数据 |

`/api/chat` 和 `/api/chat/stream` 响应中新增 `answer_id` 字段，前端凭此提交反馈。

### 10.4 流式接口 answer_id 注入

流式结束后额外推送一个 SSE 事件：

```
data: {"type": "answer_id", "id": "550e8400-e29b-41d4-a716-446655440000"}
```

前端收到后渲染赞/踩按钮，用 `answer_id` 关联反馈记录。

### 10.5 LLM-as-Judge 评分规则

```
评分维度（1.0~5.0）：
  relevance     相关性 — 回答是否切题、基于问题作答
  completeness  完整性 — 是否覆盖问题各方面
  accuracy      准确性 — 内容是否准确、引用有据可查

触发时机：用户点击 👍/👎 后异步执行，不阻塞响应
模型：复用 OPENAI_CHAT_MODEL 配置
```

### 10.6 基准测试集

`scripts/benchmark.py` 内置 11 条用例（聚合 / 元数据过滤 / 语义检索 / 鲁棒性四类）：

```bash
# 运行全量基准测试
python3 scripts/benchmark.py

# 只测第 5 条
python3 scripts/benchmark.py --case 5

# 跳过 LLM 评分（快速冒烟测试）
python3 scripts/benchmark.py --no-judge
```

结果追加写入 `data/benchmark_results.jsonl`，可用于趋势对比。

### 10.7 前端质量看板

侧边栏新增「回答质量」卡片，展示：
- 总问答数 / 好评率 / 平均响应时间
- 相关性 / 完整性 / 准确性进度条（每 60s 自动刷新）

---

## 11. 后续优化方向

### 9.1 回答质量评价体系

引入自动化评价机制，持续量化和改进回答质量。

- **用户反馈收集**: 前端添加“赞/踩”按钮，将问答对 + 评价存入数据库
- **LLM-as-Judge**: 用 GPT-4o 自动评分（相关性 / 完整性 / 准确性），与人工评分对比
- **基准测试集**: 维护一组标准 Q&A，每次修改后自动回归测试，防止质量下降
- **指标看板**: 跟踪准确率 / 响应时间 / 用户满意度等趋势

### 11.1 检索增强 (Advanced RAG)

**文件**: `app/advanced_rag.py`（新增）、`app/rag.py`、`app/config.py`、`requirements.txt`

#### 背景

原始向量检索（top_k=5 语义最近邻）存在两个核心问题：
1. **召回率不足**：仅找语义最近的 5 条，多义查询、表述差异大的查询容易漏掉相关记录
2. **排序质量差**：向量相似度与"对回答问题有用"的相关性并不等价

本次实现四项增强技术串联为一条 Pipeline，并在 `rag.py` 中作为语义检索路由的默认路径接入。

---

#### 架构总览

```
用户问题
    │
    ▼ [1] 查询改写 (Query Rewriting)         ADVANCED_RAG_QUERY_REWRITE=true
多个查询变体（原始 + N 个重写）
    │
    ▼ [2] 混合检索 (Hybrid Search / RRF)     ADVANCED_RAG_HYBRID_SEARCH=true
    │   · 对每个变体做向量检索（top_k × expand_factor 条）
    │   · BM25 对全量候选重打分
    │   · Reciprocal Rank Fusion 融合向量排名 + BM25 排名
候选命中列表（去重合并）
    │
    ▼ [3] LLM 重排序 (Reranking)             ADVANCED_RAG_RERANKER=true
    │   · 取前 ADVANCED_RAG_RERANK_TOP_N 条送入 LLM
    │   · LLM 为每条片段打 1-10 相关性分
    │   · 按分数重新排序，裁至 top_k
精排后命中列表
    │
    ▼ [4] 父文档展开 (Parent Doc Retrieval)  ADVANCED_RAG_PARENT_DOC=true
    │   · chunk_type=activity_part 子块 → 拼合所有同 chunk_id 前缀的子块
    │   · 替换为完整父活动记录，保留更丰富上下文
最终命中列表
```

---

#### 新增文件：`app/advanced_rag.py`

| 函数 | 说明 |
|---|---|
| `rewrite_query(question, n)` | LLM 改写为 n 个变体，失败时降级为 `[question]` |
| `_bm25_scores(query, texts)` | 用 `rank_bm25.BM25Okapi` 对候选打分并归一化；未安装则降级为 0.0 |
| `_reciprocal_rank_fusion(rankings)` | RRF 融合多个排名列表，参数 k=60 |
| `hybrid_retrieve(queries, top_k)` | 多查询向量召回 → BM25 重打分 → RRF 融合 → 返回 top_k |
| `llm_rerank(question, hits, top_n)` | LLM 批量打 1-10 分，JSON 解析失败时降级为原序 |
| `expand_to_parent(hits)` | activity_part 子块拼合展开，其余类型透传 |
| `advanced_retrieve(question, top_k)` | 四步串联总入口，`rag.py` 直接调用此函数 |

所有组件均有异常捕获 + 降级逻辑，任意步骤失败不影响主流程返回结果。

---

#### `app/config.py` 新增配置项

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `ADVANCED_RAG_ENABLED` | `true` | 总开关，`false` 回退到原始向量检索 |
| `ADVANCED_RAG_QUERY_REWRITE` | `true` | 启用查询改写 |
| `ADVANCED_RAG_REWRITE_N` | `2` | 生成查询变体数量 |
| `ADVANCED_RAG_HYBRID_SEARCH` | `true` | 启用 BM25 + 向量 RRF 混合检索 |
| `ADVANCED_RAG_EXPAND_FACTOR` | `3` | 候选扩展倍数（每查询召回 top_k × N 条） |
| `ADVANCED_RAG_RERANKER` | `true` | 启用 LLM 重排序 |
| `ADVANCED_RAG_RERANK_TOP_N` | `10` | 送入重排序的候选数量 |
| `ADVANCED_RAG_PARENT_DOC` | `true` | 启用父文档展开 |

---

#### `requirements.txt` 新增依赖

```
rank_bm25>=0.2.2   # BM25 混合检索
```

安装：`pip install rank_bm25`

---

#### 路由影响

仅影响 `rag.py` 的**语义检索分支**（既非评价排名、非聚合统计、也无元数据过滤条件的普通问题）：

```python
# 修改前
hits = search(question, top_k=top_k)

# 修改后
if ADVANCED_RAG_ENABLED:
    hits = advanced_retrieve(question, top_k=top_k)
else:
    hits = search(question, top_k=top_k)
```

多人评价、聚合统计、元数据过滤三条路由不受影响。

---

#### 性能预期

| 场景 | 延迟增加 | 准确率提升 |
|---|---|---|
| 仅查询改写 | +0.5~1s（LLM 调用） | 多义查询召回率 ↑ |
| 仅混合检索（BM25） | +50~100ms | 关键词精确匹配 ↑ |
| 仅重排序 | +1~2s（LLM 调用） | 排序相关性 ↑ |
| 全部启用 | +2~4s | 综合准确率显著提升 |

如对延迟敏感，可在 `.env` 中单独关闭 `ADVANCED_RAG_RERANKER=false` 或 `ADVANCED_RAG_QUERY_REWRITE=false`。


### 11.2 多轮对话与上下文记忆

当前每次问答独立，无法处理连续追问。

- **对话历史管理**: 维护 session 级别的对话历史，将近 N 轮对话作为上下文传入 LLM
- **指代消解**: 在检索前将“他们”“这个客户”等指代词解析为具体实体
- **对话摘要**: 较长对话自动压缩历史上下文，避免 token 溢出

### 11.3 分块策略优化

当前按固定 `---` 分隔符 + 字数上限切块，未考虑语义完整性。

- **语义分块**: 按 CRM 活动记录的结构化字段（客户名 / 负责人 / 日期 / 内容）智能切分
- **元数据丰富化**: 提取更多结构化字段（行业 / 商机阶段 / 金额）存入 Milvus，支持更精细的过滤
- **动态块大小**: 根据内容长度自动调整，短活动记录不拆分，长记录按语义边界拆分

### 11.4 权限与多租户

当前无身份认证，所有用户看到相同数据。

- **数据隔离**: 向量库按团队/角色分区，查询时自动注入 `owner` 过滤，仅返回当前用户可见的记录
- **API 认证**: 接入 API Key / JWT / OAuth2，保护接口安全
- **操作审计**: 记录每次查询的用户、问题、检索范围，便于合规审计

### 11.5 增量索引与实时同步

当前每次更新需要重建全量索引。

- **增量写入**: 检测新增/修改的文档，仅对变更部分重新分块和向量化
- **Webhook 触发**: CRM 系统新增活动记录时通过 Webhook 自动更新向量库
- **删除/更新同步**: 当源文档修改或删除时，自动清理对应向量

### 11.6 可观测性与监控

- **链路追踪 (Tracing)**: 接入 LangSmith / Langfuse，记录每次请求的完整链路（问题分类 → 检索 → Prompt 构建 → LLM 调用）
- **Token 用量统计**: 按用户/时段统计 Embedding 和 Chat 的 token 消耗，控制成本
- **异常告警**: 响应时间超阈值 / 检索结果为空 / API 调用失败时自动告警

### 11.7 前端体验优化

- **Markdown 渲染**: 回答中的表格 / 代码块 / 列表正确渲染，而非纯文本展示
- **来源可视化**: 点击来源可展开原文片段，关键词高亮
- **导出功能**: 支持将问答结果导出为 PDF / Markdown
- **移动端适配**: 响应式布局，支持移动端访问

### 11.8 知识库扩展

- **多数据源接入**: 除 Markdown 外，支持从 CRM 数据库 / 飞书 / 钉钉 / 企微直接拉取数据
- **文档类型扩展**: 支持 PDF / Word / Excel / 图片 OCR 等格式
- **知识图谱**: 从活动记录中提取客户-人员-事件关系，结合图数据库支持复杂关系查询

---

## 环境变量补充

在原有变量基础上新增：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHAT_API_MODE` | `responses` | API 模式：`responses`（新版）或 `completions`（传统） |
