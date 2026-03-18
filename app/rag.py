"""
RAG 问答核心模块
流程：用户问题 -> 向量检索 -> 组装 Prompt -> 调用 GPT -> 返回答案
支持流式（SSE）和普通两种模式
"""
import json
import logging
import re
from datetime import date, timedelta
from typing import List, Dict, Any, Iterator, Optional, Tuple

import httpx

from app.config import (
    CHAT_API_KEY, CHAT_BASE_URL, OPENAI_CHAT_MODEL, TOP_K, CHAT_API_MODE,
    ADVANCED_RAG_ENABLED, SUMMARY_RAG_ENABLED,
)
from app.vector_store import search, get_aggregate_stats, query_by_metadata, get_distinct_values, get_field_activity_counts, fetch_originals
from app.advanced_rag import advanced_retrieve

logger = logging.getLogger("crm_rag")

# ========== Chat API 工具 ==========

def _chat_url() -> str:
    base = CHAT_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    if CHAT_API_MODE == "responses":
        return base + "/responses"
    return base + "/chat/completions"


def _build_payload(messages: list, stream: bool) -> dict:
    """根据 CHAT_API_MODE 构建请求体，兼容 completions 和 responses 两种格式"""
    if CHAT_API_MODE == "responses":
        # OpenAI Responses API: messages -> input，system 提取为 instructions
        system_text = ""
        input_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                input_msgs.append({"role": m["role"], "content": m["content"]})
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "input": input_msgs,
            "temperature": 0.1,
            "max_output_tokens": 2048,
            "stream": stream,
        }
        if system_text:
            payload["instructions"] = system_text
        return payload
    else:
        # 标准 Chat Completions API
        return {
            "model": OPENAI_CHAT_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": stream,
        }


def _chat_headers() -> dict:
    return {
        "Authorization": f"Bearer {CHAT_API_KEY}",
        "Content-Type": "application/json",
    }


# ========== 元数据过滤：日期解析 ==========

# 当前日期为 2026-03-06，动态计算
_TODAY = date.today()


def _parse_date_range(question: str) -> Tuple[str, str]:
    """
    从问题中识别日期范围关键词，返回 (date_from, date_to)。
    日期格式均为 'YYYY-MM-DD'，未识别则返回 ('', '')。
    """
    today = _TODAY
    # 本周周一
    monday = today - timedelta(days=today.weekday())

    # (pattern, fn, needs_match_arg)
    patterns = [
        # 今天
        (r'今天|本日',
         lambda m: (str(today), str(today)), False),
        # 昨天
        (r'昨天',
         lambda m: (str(today - timedelta(1)), str(today - timedelta(1))), False),
        # 前天
        (r'前天',
         lambda m: (str(today - timedelta(2)), str(today - timedelta(2))), False),
        # 本周
        (r'本周',
         lambda m: (str(monday), str(monday + timedelta(6))), False),
        # 上礼拜|上周
        (r'上礼拜|上周',
         lambda m: (str(monday - timedelta(7)), str(monday - timedelta(1))), False),
        # 下周
        (r'下礼拜|下周',
         lambda m: (str(monday + timedelta(7)), str(monday + timedelta(13))), False),
        # YYYY年M月（具体年月）
        (r'(\d{4})年(\d{1,2})月',
         lambda m: _specific_month_range(int(m.group(1)), int(m.group(2))), True),
        # N月份（当年）
        (r'(\d{1,2})月份?(?!前|后)',
         lambda m: _specific_month_range(today.year, int(m.group(1))), True),
        # 本月
        (r'本月|这个月',
         lambda m: (f"{today.year}-{today.month:02d}-01",
                    f"{today.year}-{today.month:02d}-{_month_last_day(today)}"), False),
        # 上个月|上月
        (r'上个月|上月',
         lambda m: _last_month_range(today), False),
        # 最近N天
        (r'最近\s*([3-9]|[1-9]\d+)\s*天',
         lambda m: (str(today - timedelta(int(m.group(1)))), str(today)), True),
        # 最近N周 / 近一周
        (r'最近\s*([1-4]?)\s*周|近一周|近七天',
         lambda m: (str(today - timedelta(7 * max(1, int(m.group(1) or 1) if m.group(1) else 1))), str(today)), True),
        # 最近一个月
        (r'最近\s*[1一]个?月|最近一个月',
         lambda m: (str(today - timedelta(30)), str(today)), False),
        # 最近（无具体时长兜底，默认近 30 天）
        # ⚠ 必须放最后，优先级最低，避免遮盖上方更具体的模式
        (r'最近',
         lambda m: (str(today - timedelta(days=30)), str(today)), False),
    ]

    for pat, fn, needs_m in patterns:
        m = re.search(pat, question)
        if m:
            return fn(m) if needs_m else fn(None)

    return ("", "")


def _month_last_day(d: date) -> int:
    import calendar
    return calendar.monthrange(d.year, d.month)[1]


def _specific_month_range(year: int, month: int) -> Tuple[str, str]:
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day}")


def _last_month_range(today: date) -> Tuple[str, str]:
    import calendar
    first = today.replace(day=1)
    last_month_end = first - timedelta(1)
    last_month_start = last_month_end.replace(day=1)
    last_day = calendar.monthrange(last_month_start.year, last_month_start.month)[1]
    return (str(last_month_start), f"{last_month_start.year}-{last_month_start.month:02d}-{last_day}")


# ========== 元数据过滤：负责人识别 ==========
_owners_cache: Optional[List[str]] = None


def _get_owners() -> List[str]:
    global _owners_cache
    if _owners_cache is None:
        try:
            _owners_cache = get_distinct_values("owner")
        except Exception:
            _owners_cache = []
    return _owners_cache


def _extract_owner(question: str) -> str:
    """
    从问题中匹配已知负责人，返回 Milvus 中存储的完整名字字符串。
    支持用中文名或英文名查询。
    匹配最长的名字（防止子字误匹配）。
    """
    owners = _get_owners()
    best = ""
    best_len = 0
    for full_name in owners:
        # 让 "\u200b" 等隐藏字符不干扰，先清洗
        clean = re.sub(r'[\u200b-\u200f\ufeff]', '', full_name)
        # 提取中文部分和括号内英文
        cn_part = re.sub(r'\(.*?\)|\uff08.*?\uff09', '', clean).strip()
        en_match = re.search(r'[A-Za-z][A-Za-z\s]+', clean)
        en_part = en_match.group(0).strip() if en_match else ""

        for part in [cn_part, en_part]:
            if part and part in question and len(part) > best_len:
                best = full_name
                best_len = len(part)
    return best


def extract_filters(question: str) -> Dict[str, str]:
    """
    从问题中提取元数据过滤条件，返回
    {'owner': str, 'date_from': str, 'date_to': str}。
    均为空字符串表示未识别。
    """
    owner = _extract_owner(question)
    date_from, date_to = _parse_date_range(question)
    return {"owner": owner, "date_from": date_from, "date_to": date_to}


# ========== 多人评价/排名问题检测 ==========
# 此类问题需要逐人拉取明细活动记录，而非走聚合统计
_EVALUATION_PATTERNS = [
    r'(所有|全部|全体|每个|各个|所有的).{0,6}(销售|负责人|同事|员工).{0,10}(评价|排名|排行|考核|绩效|表现|工作|业绩)',
    r'(评价|排名|排行|考核|绩效|表现|业绩).{0,10}(所有|全部|全体|每个|各个).{0,6}(销售|负责人|同事|员工)',
    r'(销售|负责人|销售人员).{0,6}(排名|排行榜|绩效排名|评分|评级|综合.{0,4}评)',
    r'谁.{0,6}(最好|最差|最积极|最努力|排[第名]|第一|倒数)',
    r'对(所有|每个|各个|全部).{0,6}(销售|负责人).{0,10}(做|给|进行).{0,6}(评价|评估|分析|考核)',
]
_EVALUATION_RE = re.compile('|'.join(_EVALUATION_PATTERNS))


def _is_evaluation_question(question: str) -> bool:
    """判断是否为需要逐人拉取明细的多人评价/排名问题"""
    return bool(_EVALUATION_RE.search(question))


def _build_evaluation_context(question: str, per_owner_limit: int = 60) -> tuple:
    """
    为多人评价问题构建上下文：遍历所有负责人，分别拉取活动明细。
    返回 (context_str, sources)
    """
    owners = _get_owners()
    if not owners:
        return "（知识库中未找到任何负责人记录）", []

    logger.info(f"[evaluation] 开始逐一拉取 {len(owners)} 位负责人的活动记录")
    parts = []
    sources = set()

    for owner in owners:
        hits = query_by_metadata(owner=owner, limit=per_owner_limit)
        logger.info(f"[evaluation] {owner}: 共 {len(hits)} 条记录")
        if not hits:
            parts.append(f"\n### 负责人：{owner}\n（无活动记录）\n")
            continue

        sources.update(h["source"] for h in hits)
        # 按日期排序（新 → 旧）
        hits_sorted = sorted(hits, key=lambda h: h.get("date") or "", reverse=True)
        records = []
        for h in hits_sorted:
            meta = ""
            if h.get("date"):
                meta += f"[{h['date']}] "
            if h.get("company"):
                meta += f"客户:{h['company']}  "
            records.append(f"{meta}{h['text']}")
        parts.append(
            f"\n### 负责人：{owner}（共 {len(hits)} 条活动记录）\n"
            + "\n".join(records)
        )

    context = (
        "【各销售人员活动明细（逐人完整记录）】\n"
        "以下为知识库中所有负责人的跟进活动详情，请基于此进行综合评价与排名。\n"
        + "".join(parts)
    )
    return context, list(sources)


# ========== 活跃度/数量排名问题检测 ==========
# 检测"前N客户/负责人 按活动记录数量排名"类问题
_RANKING_PATTERNS = [
    r'活跃.{0,6}前\s*(\d+|[一二三四五六七八九十]+)',
    r'前\s*(\d+|[一二三四五六七八九十]+).{0,10}(活跃|活动|跟进|记录)',
    r'(活动|跟进|拜访).{0,6}(记录|次数).{0,10}(前|最多|排名|排行)',
    r'(最|更)活跃.{0,8}(客户|公司|负责人|销售)',
    r'(客户|公司|负责人|销售).{0,10}(活跃|活动次数|跟进次数).{0,6}(排[名行]|前\d)',
    r'(活动|记录).{0,4}(数量|次数).{0,4}(排[名行]|前\d|最多)',
    r'跟进(次数|记录数).{0,6}(前|最多|排)',
    r'(哪[些个家]).{0,6}客户.{0,10}(最|更)?(多|频繁|活跃)',
]
_RANKING_RE = re.compile('|'.join(_RANKING_PATTERNS))

# 识别排名字段：客户 or 负责人
_RANKING_FIELD_OWNER_RE = re.compile(r'(负责人|销售|员工|同事)')
_RANKING_FIELD_COMPANY_RE = re.compile(r'(客户|公司|企业)')


def _is_ranking_question(question: str) -> bool:
    """判断是否为按活动数量排名的问题"""
    return bool(_RANKING_RE.search(question))


def _extract_ranking_params(question: str) -> Dict[str, Any]:
    """
    从问题中提取排名参数：
    - field: 'company' 或 'owner'
    - top_n: 数字，默认 5
    """
    # 判断字段
    if _RANKING_FIELD_OWNER_RE.search(question) and not _RANKING_FIELD_COMPANY_RE.search(question):
        field = "owner"
    else:
        field = "company"

    # 中文数字映射
    _CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

    # 提取 N（支持阿拉伯数字和中文数字）
    n_match = re.search(r'前\s*(\d+|[一二三四五六七八九十]+)', question)
    if n_match:
        raw = n_match.group(1)
        if raw.isdigit():
            top_n = int(raw)
        else:
            top_n = _CN_NUM.get(raw, 5)
    else:
        top_n = 5

    return {"field": field, "top_n": top_n}


def _build_ranking_context(question: str) -> tuple:
    """为活跃度排名问题构建上下文：统计各实体的活动记录数量并排序"""
    params = _extract_ranking_params(question)
    field = params["field"]
    top_n = params["top_n"]

    counts = get_field_activity_counts(field, top_n=top_n * 3)  # 多拉一些做筛选余量

    if not counts:
        return "（知识库中未找到相关记录）", []

    field_label = "客户" if field == "company" else "负责人"
    lines = [
        f"【{field_label}活跃度排名（按活动记录数量，共统计 {len(counts)} 个{field_label}）】",
        f"以下为活动记录数量最多的前 {top_n} 个{field_label}：\n",
    ]
    for rank, item in enumerate(counts[:top_n], 1):
        lines.append(f"第 {rank} 名：{item['value']}（活动记录数：{item['count']} 条）")

    context = "\n".join(lines)
    return context, ["crm_activities_recent.md"]


# ========== 聚合问题关键词检测 ==========
# 聚合型问题使用正则匹配，兼容量词变体（个/家/位/条 等）
_AGGREGATE_PATTERNS = [
    r'多少[个家位条]?客户',
    r'多少[个家]?公司',
    r'[哪所全]+(些|有|部)?客户',      # 哪些客户 / 所有客户 / 全部客户
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
    """判断是否为需要全量统计的聚合型问题。
    若问题中含有具体负责人名，优先识别为元数据过滤而非聚合。
    """
    if not _AGGREGATE_RE.search(question):
        return False
    # 含业务分析/进展语义时，不走聚合清单路由
    if re.search(r'进展|跟进|推进|分析|总结|情况|动态|风险|机会|策略|建议|成单|转化|原因', question):
        return False
    # 如果问题里有具体人名，走元数据过滤而非聚合
    if _extract_owner(question):
        return False
    return True


def _build_aggregate_context(question: str) -> tuple:
    """为聚合型问题构建全量统计上下文，返回 (context_str, sources)"""
    stats = get_aggregate_stats()
    companies_str = "、".join(stats["companies"]) if stats["companies"] else "（无数据）"
    owners_str = "、".join(stats["owners"]) if stats["owners"] else "（无数据）"
    context = (
        f"【知识库全量统计】\n"
        f"总活动记录片段数：{stats['total_chunks']}\n"
        f"客户总数：{stats['company_count']}\n"
        f"客户列表：{companies_str}\n\n"
        f"负责人总数：{stats['owner_count']}\n"
        f"负责人列表：{owners_str}\n"
    )
    return context, ["crm_activities_recent.md"]


# ========== System Prompt ==========
SYSTEM_PROMPT = """你是一个专业的 CRM 业务助手，拥有公司近期客户跟进活动的完整知识库。
你的任务是根据提供的 CRM 活动记录，准确、深入地回答用户问题。

回答规范：
1. **信息提取**：直接基于检索到的上下文内容回答，不要编造不存在的信息
2. **推理分析**：可以基于活动记录进行合理的分析、总结和评价
   - 例如：根据跟进频率、客户数量、成交情况等推断工作表现
   - 例如：根据活动内容分析工作风格、优势和不足
3. **评价维度**：当用户要求用特定词汇或维度评价时，基于实际记录灵活运用
   - 理解用户的评价语境（如网络用语、俚语等）
   - 结合活动记录的客观数据给出合理评价
4. **缺失信息**：仅当完全没有相关记录时，才告知"知识库中未找到相关记录"
   - 如果有活动记录但缺少某些具体信息（如绩效指标），说明"记录中未包含该信息，但可基于活动情况分析..."
5. **表达方式**：
   - 涉及具体客户、日期、负责人时，精确引用原文
   - 进行分析评价时，清晰说明依据和推理过程
   - 回答使用中文，结构清晰，如有多个相关记录用编号列举
"""


def build_context(hits: List[Dict[str, Any]]) -> str:
    """将检索到的片段拼接成上下文字符串"""
    if not hits:
        return "（未检索到相关记录）"
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = ""
        if hit.get("date"):
            meta += f"日期：{hit['date']}  "
        if hit.get("company"):
            meta += f"客户：{hit['company']}  "
        if hit.get("owner"):
            meta += f"负责人：{hit['owner']}  "
        parts.append(
            f"【片段 {i}】{meta}相似度：{hit['score']}\n"
            f"{hit['text']}"
        )
    return "\n\n".join(parts)


def build_messages(question: str, context: str, sort_by: str = "relevance") -> List[Dict[str, str]]:
    """构建发送给 GPT 的消息列表。
    sort_by: 'relevance'（向量相似度） | 'date'（元数据日期降序）
    """
    if sort_by == "date":
        sort_note = "已按日期降序排列（最新记录在前）"
    else:
        sort_note = "已按相关性排序"
    user_content = f"""请根据以下 CRM 活动记录回答问题。

=== CRM 活动记录（{sort_note}）===
{context}

=== 用户问题 ===
{question}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def answer(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """普通问答（非流式），返回完整 JSON"""
    logger.info(f"[answer] 问题: {question}")

    # --- 路由优先级判断 ---
    if _is_ranking_question(question):
        # 活跃度/数量排名类：统计各实体activity记录数
        logger.info("[answer] 检测到活跃度排名问题，统计活动记录数量")
        context, sources = _build_ranking_context(question)
        hits = []
    elif _is_evaluation_question(question):
        # 多人评价排名类：逐人拉取活动明细
        logger.info("[answer] 检测到多人评价/排名问题，逐人拉取活动明细")
        context, sources = _build_evaluation_context(question)
        hits = []
    elif _is_aggregate_question(question):
        # 全量统计类：多少客户/负责人
        logger.info("[answer] 检测到聚合统计问题，使用全量元数据查询")
        context, sources = _build_aggregate_context(question)
        hits = []
    else:
        # 尝试提取元数据过滤器
        filters = extract_filters(question)
        has_filter = bool(filters["owner"] or filters["date_from"])

        if has_filter:
            # 元数据过滤类：指定负责人/日期，精确匹配
            logger.info(f"[answer] 元数据过滤检索: owner={filters['owner']!r}  "
                        f"date={filters['date_from']} ~ {filters['date_to']}")
            hits = query_by_metadata(
                owner=filters["owner"],
                date_from=filters["date_from"],
                date_to=filters["date_to"],
                limit=top_k * 8,  # 全量匹配，多取一些
            )
            logger.info(f"[answer] 元数据查询匹配 {len(hits)} 条")
            if not hits:
                context = (
                    f"未在知识库中找到"
                    f"{'负责人为' + filters['owner'] + '的' if filters['owner'] else ''}"
                    f"{filters['date_from'] + ' 至 ' + filters['date_to'] + '期间的' if filters['date_from'] else ''}"
                    f"活动记录。"
                )
            else:
                context = build_context(hits)
            sources = list({h["source"] for h in hits})
            sort_by = "date"
        else:
            # 语义检索（普通 or Advanced RAG）
            if ADVANCED_RAG_ENABLED:
                hits = advanced_retrieve(question, top_k=top_k)
                logger.info(f"[answer] Advanced RAG 检索到 {len(hits)} 条相关片段")
            else:
                hits = search(question, top_k=top_k)
                logger.info(f"[answer] 语义检索到 {len(hits)} 条相关片段")
            # 摘要模式：用摘要检索，用原文生成回答
            if SUMMARY_RAG_ENABLED and hits:
                hits = fetch_originals(hits)
                logger.info(f"[answer] 摘要模式：已替换为 {len(hits)} 条原始记录")
            context = build_context(hits)
            sources = list({h["source"] for h in hits})
            sort_by = "relevance"

    messages = build_messages(question, context, sort_by=sort_by)

    url = _chat_url()
    payload = _build_payload(messages, stream=False)

    logger.info(f"[answer] 调用 Chat API: {url}  model={OPENAI_CHAT_MODEL}  mode={CHAT_API_MODE}")
    try:
        resp = httpx.post(url, headers=_chat_headers(), json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"[answer] Chat API HTTP 错误 {e.response.status_code}: {e.response.text}")
        raise RuntimeError(f"Chat API 请求失败 {e.response.status_code}: {e.response.text[:200]}") from e

    # 兼容两种响应格式
    if CHAT_API_MODE == "responses":
        try:
            answer_text = data["output"][0]["content"][0]["text"]
        except (KeyError, IndexError):
            logger.error(f"[answer] Responses API 响应格式异常: {json.dumps(data, ensure_ascii=False)[:300]}")
            raise RuntimeError(f"Responses API 响应格式异常")
    else:
        answer_text = data["choices"][0]["message"]["content"]

    usage = data.get("usage", {})
    logger.info(f"[answer] 完成，usage={usage}")

    return {
        "answer": answer_text,
        "sources": sources,
        "hits": hits,
        "model": OPENAI_CHAT_MODEL,
        "usage": usage,
    }


def answer_stream(question: str, top_k: int = TOP_K) -> Iterator[str]:
    """
    流式问答，使用 httpx 直接处理 SSE，兼容第三方代理。
    格式：data: <json>\n\n
    """
    logger.info(f"[stream] 问题: {question}")

    # --- 路由判断 ---
    if _is_ranking_question(question):
        logger.info("[stream] 检测到活跃度排名问题，统计活动记录数量")
        try:
            context, sources = _build_ranking_context(question)
            hits = []
        except Exception as e:
            logger.error(f"[stream] 排名查询失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'排名查询失败: {e}'}, ensure_ascii=False)}\n\n"
            return
    elif _is_evaluation_question(question):
        # 多人评价排名类：逐人拉取活动明细
        logger.info("[stream] 检测到多人评价/排名问题，逐人拉取活动明细")
        try:
            context, sources = _build_evaluation_context(question)
            hits = []
        except Exception as e:
            logger.error(f"[stream] 多人评价查询失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'评价查询失败: {e}'}, ensure_ascii=False)}\n\n"
            return
    elif _is_aggregate_question(question):
        logger.info("[stream] 检测到聚合统计问题，使用全量元数据查询")
        try:
            context, sources = _build_aggregate_context(question)
            hits = []
        except Exception as e:
            logger.error(f"[stream] 聚合查询失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'聚合查询失败: {e}'}, ensure_ascii=False)}\n\n"
            return
    else:
        filters = extract_filters(question)
        has_filter = bool(filters["owner"] or filters["date_from"])

        if has_filter:
            logger.info(f"[stream] 元数据过滤检索: owner={filters['owner']!r}  "
                        f"date={filters['date_from']} ~ {filters['date_to']}")
            try:
                hits = query_by_metadata(
                    owner=filters["owner"],
                    date_from=filters["date_from"],
                    date_to=filters["date_to"],
                    limit=top_k * 8,
                )
            except Exception as e:
                logger.error(f"[stream] 元数据查询失败: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': f'元数据查询失败: {e}'}, ensure_ascii=False)}\n\n"
                return
            logger.info(f"[stream] 元数据查询匹配 {len(hits)} 条")
            if not hits:
                context = (
                    f"未在知识库中找到"
                    f"{'负责人为' + filters['owner'] + '的' if filters['owner'] else ''}"
                    f"{filters['date_from'] + ' 至 ' + filters['date_to'] + '期间的' if filters['date_from'] else ''}"
                    f"活动记录。"
                )
            else:
                context = build_context(hits)
            sources = list({h["source"] for h in hits})
            sort_by = "date"
        else:
            # --- 语义检索（普通 or Advanced RAG）---
            try:
                if ADVANCED_RAG_ENABLED:
                    hits = advanced_retrieve(question, top_k=top_k)
                    logger.info(f"[stream] Advanced RAG 检索到 {len(hits)} 条相关片段")
                else:
                    hits = search(question, top_k=top_k)
                    logger.info(f"[stream] 语义检索到 {len(hits)} 条相关片段")
            except Exception as e:
                logger.error(f"[stream] 检索失败: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': f'检索失败: {e}'}, ensure_ascii=False)}\n\n"
                return
            # 摘要模式：用摘要检索，用原文生成回答
            if SUMMARY_RAG_ENABLED and hits:
                hits = fetch_originals(hits)
                logger.info(f"[stream] 摘要模式：已替换为 {len(hits)} 条原始记录")
            sources = list({h["source"] for h in hits})
            context = build_context(hits)
            sort_by = "relevance"

    # --- 2. 先推送来源信息 ---
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'hits': hits}, ensure_ascii=False)}\n\n"

    # --- 3. 构建 Prompt ---
    messages = build_messages(question, context, sort_by=sort_by)
    url      = _chat_url()
    payload  = _build_payload(messages, stream=True)

    logger.info(f"[stream] 调用 Chat API: {url}  mode={CHAT_API_MODE}")

    # --- 4. 流式请求 ---
    token_count = 0
    try:
        with httpx.stream(
            "POST", url,
            headers=_chat_headers(),
            json=payload,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="ignore")
                logger.error(f"[stream] Chat API HTTP {resp.status_code}: {body[:300]}")
                yield f"data: {json.dumps({'type': 'error', 'content': f'Chat API 错误 {resp.status_code}: {body[:200]}'}, ensure_ascii=False)}\n\n"
                return

            buffer = ""
            for raw_line in resp.iter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    logger.info(f"[stream] [DONE] 收到，共 {token_count} 个 token")
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning(f"[stream] 无法解析行: {data_str[:100]}")
                    continue

                # 提取 token 内容，兼容两种格式
                content = None
                if CHAT_API_MODE == "responses":
                    chunk_type = chunk.get("type", "")
                    if chunk_type == "response.output_text.delta":
                        content = chunk.get("delta", "")
                    elif chunk_type in ("response.completed", "response.done"):
                        logger.info(f"[stream] {chunk_type}，共 {token_count} 个 token")
                        break
                    elif chunk_type == "error":
                        err_msg = chunk.get("message", str(chunk))
                        logger.error(f"[stream] API error 事件: {err_msg}")
                        yield f"data: {json.dumps({'type': 'error', 'content': err_msg}, ensure_ascii=False)}\n\n"
                        return
                else:
                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content")

                if content:
                    token_count += 1
                    yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

    except httpx.TimeoutException:
        logger.error("[stream] Chat API 请求超时")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Chat API 请求超时，请稍后重试'}, ensure_ascii=False)}\n\n"
        return
    except Exception as e:
        logger.error(f"[stream] 未知错误: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': f'请求异常: {e}'}, ensure_ascii=False)}\n\n"
        return

    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
