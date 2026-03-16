"""
Advanced RAG 检索增强模块 (11.1)

实现四项增强技术：
  1. 查询改写 (Query Rewriting)      — 用 LLM 生成多个查询变体，扩大召回
  2. 混合检索 (Hybrid Search)        — 向量检索 + BM25 关键词检索融合 (RRF)
  3. 重排序   (Reranking)            — 用 LLM Cross-Encoder 对候选片段重打分
  4. 父文档检索 (Parent Doc Retrieval)— 命中子块时自动扩展至完整父活动记录

所有特性均可通过 config.py 中的开关独立控制，降级为原始向量检索不影响主流程。
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Any, Optional

import httpx

from app.config import (
    CHAT_API_KEY, CHAT_BASE_URL, OPENAI_CHAT_MODEL, CHAT_API_MODE,
    ADVANCED_RAG_QUERY_REWRITE,
    ADVANCED_RAG_HYBRID_SEARCH,
    ADVANCED_RAG_RERANKER,
    ADVANCED_RAG_PARENT_DOC,
    ADVANCED_RAG_REWRITE_N,
    ADVANCED_RAG_EXPAND_FACTOR,
    ADVANCED_RAG_RERANK_TOP_N,
)
from app.vector_store import search, query_by_metadata

logger = logging.getLogger("crm_rag.advanced")

# ──────────────────────────────────────────────
# 内部工具：调用 Chat API（非流式，短 prompt）
# ──────────────────────────────────────────────

def _chat_url() -> str:
    base = CHAT_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + ("/responses" if CHAT_API_MODE == "responses" else "/chat/completions")


def _chat_headers() -> dict:
    return {"Authorization": f"Bearer {CHAT_API_KEY}", "Content-Type": "application/json"}


def _extract_responses_text(data: dict) -> str:
    """
    从 Responses API 返回中提取文本，兼容不同供应商的字段差异。
    常见结构：
    1) output[0].content[0].text
    2) output_text (字符串)
    3) output[*].content[*].text / output[*].content[*].output_text
    """
    if not isinstance(data, dict):
        return ""

    # 1) OpenAI 标准 / 常见代理结构
    try:
        output = data.get("output", [])
        if isinstance(output, list):
            texts: List[str] = []
            for item in output:
                content = item.get("content", []) if isinstance(item, dict) else []
                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        t = c.get("text") or c.get("output_text") or ""
                        if t:
                            texts.append(str(t))
            if texts:
                return "\n".join(texts).strip()
    except Exception:
        pass

    # 2) 部分代理直接返回 output_text
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # 3) 兜底：兼容非常规字段
    for k in ("text", "content"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _call_chat(system: str, user: str, max_tokens: int = 512) -> str:
    """
    调用 Chat API 一次，返回模型输出文本。
    兼容 completions / responses 两种模式。
    出错时返回空字符串（降级），不向上抛异常。
    """
    if CHAT_API_MODE == "responses":
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "instructions": system,
            "input": [{"role": "user", "content": user}],
            "temperature": 0.0,
            "max_output_tokens": max_tokens,
            "stream": False,
        }
    else:
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
    try:
        resp = httpx.post(_chat_url(), headers=_chat_headers(), json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if CHAT_API_MODE == "responses":
            text = _extract_responses_text(data)
            if text:
                return text
            raise RuntimeError(f"Responses API 返回中未找到文本字段: {json.dumps(data, ensure_ascii=False)[:220]}")
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"[advanced_rag] _call_chat 失败，降级: {e}")
        return ""


# ══════════════════════════════════════════════
# 1. 查询改写 (Query Rewriting)
# ══════════════════════════════════════════════

_REWRITE_SYSTEM = """你是一个 CRM 知识库检索助手。
用户会给你一个查询问题，你需要将它改写成 {n} 个不同的检索变体，以便从向量数据库中检索到更多相关内容。
要求：
1. 每个变体占一行，不要加编号或标点前缀
2. 变体应从不同角度表达同一查询意图（同义替换、缩略展开、侧重点调整等）
3. 只输出变体，不要输出其他任何内容
"""


def rewrite_query(question: str, n: int = ADVANCED_RAG_REWRITE_N) -> List[str]:
    """
    用 LLM 将原始查询改写为 n 个变体，返回 [原始查询] + 变体列表。
    若改写失败，仅返回 [原始查询]。
    """
    if not ADVANCED_RAG_QUERY_REWRITE:
        return [question]

    system = _REWRITE_SYSTEM.format(n=n)
    raw = _call_chat(system, question, max_tokens=256)
    if not raw.strip():
        logger.info("[advanced_rag] 查询改写返回空，跳过")
        return [question]

    variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    # 过滤与原始问题完全相同的变体
    variants = [v for v in variants if v != question][:n]
    all_queries = [question] + variants
    logger.info(f"[advanced_rag] 查询改写 {len(all_queries)-1} 个变体: {variants}")
    return all_queries


# ══════════════════════════════════════════════
# 2. 混合检索：多查询向量搜索 + BM25 重打分 (RRF 融合)
# ══════════════════════════════════════════════

def _bm25_scores(query: str, texts: List[str]) -> List[float]:
    """
    对 texts 用 BM25 打分。
    依赖 rank_bm25；若未安装则全部返回 0.0。
    """
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:
        logger.warning("[advanced_rag] rank_bm25 未安装，BM25 降级为 0.0 分")
        return [0.0] * len(texts)

    # 简单中文分词：按字拆分
    def tokenize(s: str) -> List[str]:
        return list(re.sub(r'\s+', '', s))

    corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus)
    q_tokens = tokenize(query)
    raw = bm25.get_scores(q_tokens)
    # 归一化到 [0, 1]
    max_score = max(raw) if max(raw) > 0 else 1.0
    return [float(s / max_score) for s in raw]


def _reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion (RRF)。
    rankings: 多个按相关性排序的 chunk_id 列表
    返回 {chunk_id: rrf_score}，分数越高越相关
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_retrieve(
    queries: List[str],
    top_k: int,
    expand_factor: int = ADVANCED_RAG_EXPAND_FACTOR,
) -> List[Dict[str, Any]]:
    """
    混合检索主函数：
    1. 对每个查询做向量检索（top_k * expand_factor 条候选）
    2. 合并去重所有候选
    3. 用第一个查询（原始问题）对候选做 BM25 打分
    4. RRF 融合向量排名 + BM25 排名
    5. 按 RRF 分数排序，返回 top_k 条
    """
    expanded = top_k * expand_factor  # 每个查询的初始召回量

    # --- Step 1: 多查询向量检索 ---
    all_hits: Dict[str, Dict[str, Any]] = {}   # chunk_id -> hit
    vector_rankings: List[List[str]] = []

    for q in queries:
        hits = search(q, top_k=expanded)
        ranking: List[str] = []
        for h in hits:
            cid = h.get("chunk_id") or h.get("text", "")[:32]
            if cid not in all_hits:
                all_hits[cid] = h
            ranking.append(cid)
        vector_rankings.append(ranking)
        logger.info(f"[advanced_rag] 向量检索 '{q[:30]}…' → {len(hits)} 条")

    unique_hits = list(all_hits.values())
    logger.info(f"[advanced_rag] 合并去重后候选 {len(unique_hits)} 条")

    if not ADVANCED_RAG_HYBRID_SEARCH or len(unique_hits) == 0:
        # 未启用 BM25 或无候选：仅按第一个查询的向量分排序
        unique_hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        return unique_hits[:top_k]

    # --- Step 2: BM25 打分 ---
    original_query = queries[0]
    texts = [h.get("text", "") for h in unique_hits]
    bm25_raw = _bm25_scores(original_query, texts)

    # BM25 排名（score 降序）
    bm25_order = sorted(range(len(unique_hits)), key=lambda i: bm25_raw[i], reverse=True)
    bm25_ranking = [
        (unique_hits[i].get("chunk_id") or unique_hits[i].get("text", "")[:32])
        for i in bm25_order
    ]

    # --- Step 3: RRF 融合 ---
    all_rankings = vector_rankings + [bm25_ranking]
    rrf = _reciprocal_rank_fusion(all_rankings)

    # 将 RRF 分数写回 hit，方便调试
    for h in unique_hits:
        cid = h.get("chunk_id") or h.get("text", "")[:32]
        h["rrf_score"] = round(rrf.get(cid, 0.0), 6)

    unique_hits.sort(key=lambda h: h.get("rrf_score", 0.0), reverse=True)
    result = unique_hits[:top_k]
    logger.info(f"[advanced_rag] RRF 融合后取 top {top_k}，最高 RRF={result[0]['rrf_score'] if result else 0}")
    return result


# ══════════════════════════════════════════════
# 3. LLM 重排序 (Reranking)
# ══════════════════════════════════════════════

_RERANK_SYSTEM = """你是一个文档相关性评分器。
给定一个用户问题和若干文档片段，请对每个片段的相关性打分（整数 1-10），10 分最相关。
只输出 JSON 数组，格式严格如下（不要输出其他任何内容）：
[{"index": 0, "score": 8}, {"index": 1, "score": 3}, ...]
"""


def llm_rerank(question: str, hits: List[Dict[str, Any]], top_n: int = ADVANCED_RAG_RERANK_TOP_N) -> List[Dict[str, Any]]:
    """
    使用 LLM 对检索结果重排序，取 top_n 条。
    若 LLM 调用失败，返回原始顺序的 top_n 条。
    """
    if not ADVANCED_RAG_RERANKER or not hits:
        return hits[:top_n]

    # 每个片段截到 400 字显示给 LLM，节省 token
    snippets = []
    for i, h in enumerate(hits):
        text_preview = h.get("text", "")[:400].replace("\n", " ")
        meta = ""
        if h.get("date"):
            meta += f"[{h['date']}] "
        if h.get("company"):
            meta += f"客户:{h['company']} "
        if h.get("owner"):
            meta += f"负责人:{h['owner']} "
        snippets.append(f"[{i}] {meta}{text_preview}")

    user_content = f"问题：{question}\n\n文档片段：\n" + "\n\n".join(snippets)

    raw = _call_chat(_RERANK_SYSTEM, user_content, max_tokens=256)
    if not raw.strip():
        logger.info("[advanced_rag] 重排序返回空，使用原始顺序")
        return hits[:top_n]

    try:
        # 提取 JSON 数组（允许 LLM 前后有多余文字）
        json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not json_match:
            raise ValueError("无法解析 JSON")
        scored = json.loads(json_match.group(0))
        # 按 score 降序排列
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        reranked = []
        for item in scored:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(hits):
                h = dict(hits[idx])
                h["rerank_score"] = item.get("score", 0)
                reranked.append(h)
        if reranked:
            logger.info(f"[advanced_rag] 重排序完成，top_n={top_n}，最高分={reranked[0]['rerank_score']}")
            return reranked[:top_n]
    except Exception as e:
        logger.warning(f"[advanced_rag] 重排序解析失败: {e}，使用原始顺序")

    return hits[:top_n]


# ══════════════════════════════════════════════
# 4. 父文档检索 (Parent Document Retrieval)
# ══════════════════════════════════════════════

def expand_to_parent(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对 chunk_type=activity_part 的子块，尝试从 Milvus 查找同一活动记录
    的所有子块并拼合为完整父文档，替换原条目。
    其余类型的块保持不变。
    """
    if not ADVANCED_RAG_PARENT_DOC:
        return hits

    expanded: List[Dict[str, Any]] = []
    seen_parent_ids: set = set()

    for h in hits:
        if h.get("chunk_type") != "activity_part":
            expanded.append(h)
            continue

        # chunk_id 格式为 "{base}_{sub}"，base 即父块标识
        raw_cid = str(h.get("chunk_id", ""))
        parts = raw_cid.rsplit("_", 1)
        parent_id = parts[0] if len(parts) == 2 else raw_cid

        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        # 查询同 source + chunk_id LIKE "{parent_id}_%"
        try:
            source = h.get("source", "")
            siblings = query_by_metadata(owner="", date_from="", date_to="", limit=20)
            # 过滤出同 source 且 chunk_id 以 parent_id 开头的子块
            siblings = [
                s for s in siblings
                if s.get("source") == source
                and str(s.get("chunk_id", "")).startswith(parent_id)
            ]
        except Exception as e:
            logger.warning(f"[advanced_rag] 父文档查询失败 chunk_id={raw_cid}: {e}")
            expanded.append(h)
            continue

        if not siblings:
            expanded.append(h)
            continue

        # 按 chunk_id 排序，拼合所有子块文本
        siblings.sort(key=lambda s: str(s.get("chunk_id", "")))
        full_text = "\n\n".join(s.get("text", "") for s in siblings)
        parent_hit = dict(h)
        parent_hit["text"] = full_text
        parent_hit["chunk_type"] = "activity"   # 标记为已展开
        parent_hit["_expanded"] = True
        parent_hit["_child_count"] = len(siblings)
        logger.info(f"[advanced_rag] 父文档展开: chunk_id={parent_id}，合并 {len(siblings)} 个子块")
        expanded.append(parent_hit)

    return expanded


# ══════════════════════════════════════════════
# 总入口：advanced_retrieve
# ══════════════════════════════════════════════

def advanced_retrieve(
    question: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Advanced RAG 检索总入口，串联四项增强：

    原始查询
        │
        ▼ [1] 查询改写 (可选)
    多个查询变体
        │
        ▼ [2] 混合检索：多查询向量 + BM25 RRF 融合 (可选)
    候选命中列表
        │
        ▼ [3] LLM 重排序 (可选)
    精排后列表
        │
        ▼ [4] 父文档展开 (可选)
    最终命中列表
    """
    logger.info(f"[advanced_rag] 开始增强检索，question={question[:50]!r}")

    # Step 1: 查询改写
    queries = rewrite_query(question)

    # Step 2: 混合检索
    hits = hybrid_retrieve(queries, top_k=top_k)

    # Step 3: LLM 重排序（在候选中取更多，重排后裁至 top_k）
    rerank_pool = hits[:ADVANCED_RAG_RERANK_TOP_N] if ADVANCED_RAG_RERANKER else hits
    if ADVANCED_RAG_RERANKER and rerank_pool:
        rerank_pool = llm_rerank(question, rerank_pool, top_n=top_k)
    else:
        rerank_pool = rerank_pool[:top_k]

    # Step 4: 父文档展开
    final_hits = expand_to_parent(rerank_pool)

    logger.info(f"[advanced_rag] 增强检索完成，最终 {len(final_hits)} 条")
    return final_hits
