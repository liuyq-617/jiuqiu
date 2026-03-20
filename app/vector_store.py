"""
向量存储模块 - 基于 Milvus
负责：文本向量化（OpenAI Embedding）+ Milvus 集合管理 + 相似度检索
"""
from typing import List, Dict, Any, Optional
import hashlib
import sqlite3
import struct
import httpx
import json
from openai import OpenAI
from pymilvus import (
    connections, utility, Collection, CollectionSchema,
    FieldSchema, DataType, MilvusException
)

from app.config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION, MILVUS_HOST, MILVUS_PORT,
    MILVUS_COLLECTION, TOP_K, SCORE_THRESHOLD, BASE_DIR
)

# ========== Embedding 磁盘缓存 ==========
_CACHE_PATH = BASE_DIR / "data" / "embedding_cache.db"


def _cache_key(text: str) -> str:
    """以 SHA256(text + model) 作为缓存键，模型变更自动失效"""
    raw = f"{OPENAI_EMBEDDING_MODEL}::{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_cache_conn() -> sqlite3.Connection:
    _CACHE_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_PATH))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings "
        "(key TEXT PRIMARY KEY, vector BLOB)"
    )
    conn.commit()
    return conn


def _read_cache(keys: List[str]) -> Dict[str, List[float]]:
    """批量读缓存，返回 {key: vector}"""
    if not keys:
        return {}
    conn = _get_cache_conn()
    placeholders = ",".join("?" * len(keys))
    rows = conn.execute(
        f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", keys
    ).fetchall()
    conn.close()
    result = {}
    for key, blob in rows:
        n = len(blob) // 8
        result[key] = list(struct.unpack(f"{n}d", blob))
    return result


def _write_cache(items: Dict[str, List[float]]):
    """批量写缓存"""
    if not items:
        return
    conn = _get_cache_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
        [
            (key, struct.pack(f"{len(vec)}d", *vec))
            for key, vec in items.items()
        ],
    )
    conn.commit()
    conn.close()


# ========== Embedding 专用客户端 ==========
_embedding_client: Optional[OpenAI] = None


def get_embedding_client() -> OpenAI:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    return _embedding_client

# 兼容旧调用
get_openai_client = get_embedding_client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    批量向量化文本。优先读磁盘缓存（SQLite），仅对未命中的文本调用 API。
    缓存以 SHA256(model+text) 为键，模型变更自动失效。
    """
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    keys    = [_cache_key(t) for t in cleaned]

    # --- 读缓存 ---
    cached  = _read_cache(keys)
    miss_indices = [i for i, k in enumerate(keys) if k not in cached]

    hit_count  = len(keys) - len(miss_indices)
    miss_count = len(miss_indices)
    print(f"  [缓存] 命中 {hit_count} 条，需请求 API {miss_count} 条")

    # --- 对未命中批量调用 API ---
    if miss_indices:
        base = EMBEDDING_BASE_URL.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        url = base + "/embeddings"
        headers = {
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        }

        batch_size = 100
        miss_texts = [cleaned[i] for i in miss_indices]
        new_cache: Dict[str, List[float]] = {}

        for b in range(0, len(miss_texts), batch_size):
            batch = miss_texts[b: b + batch_size]
            payload = {"model": OPENAI_EMBEDDING_MODEL, "input": batch}

            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Embedding API 请求失败 HTTP {e.response.status_code}: {e.response.text}"
                ) from e

            if "data" not in data:
                raise RuntimeError(
                    f"Embedding API 返回格式异常: {json.dumps(data, ensure_ascii=False)[:300]}"
                )

            sorted_items = sorted(data["data"], key=lambda x: x["index"])
            for j, item in enumerate(sorted_items):
                global_idx = miss_indices[b + j]
                key = keys[global_idx]
                vec = item["embedding"]
                cached[key]     = vec
                new_cache[key]  = vec

            done = min(b + batch_size, len(miss_texts))
            print(f"  [Embedding] API 已处理 {done}/{miss_count} 条")

        # --- 写入缓存 ---
        _write_cache(new_cache)
        print(f"  [缓存] 新增写入 {len(new_cache)} 条到 {_CACHE_PATH.name}")

    # --- 按原始顺序返回 ---
    return [cached[k] for k in keys]


# ========== Milvus 连接管理 ==========
def connect_milvus():
    """连接 Milvus"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            timeout=10,
        )
        print(f"[Milvus] 已连接 {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        raise ConnectionError(f"Milvus 连接失败: {e}")


def disconnect_milvus():
    try:
        connections.disconnect("default")
    except Exception:
        pass


def ensure_collection() -> Collection:
    """确保 Milvus Collection 存在，不存在则创建"""
    if utility.has_collection(MILVUS_COLLECTION):
        col = Collection(MILVUS_COLLECTION)
        col.load()
        return col

    # 定义 Schema
    fields = [
        FieldSchema(name="id",         dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name="text",        dtype=DataType.VARCHAR,       max_length=8192),
        FieldSchema(name="source",      dtype=DataType.VARCHAR,       max_length=256),
        FieldSchema(name="chunk_id",    dtype=DataType.VARCHAR,       max_length=64),
        FieldSchema(name="chunk_type",  dtype=DataType.VARCHAR,       max_length=32),
        FieldSchema(name="date",        dtype=DataType.VARCHAR,       max_length=16),
        FieldSchema(name="company",     dtype=DataType.VARCHAR,       max_length=256),
        FieldSchema(name="owner",       dtype=DataType.VARCHAR,       max_length=128),
        FieldSchema(name="embedding",   dtype=DataType.FLOAT_VECTOR,  dim=EMBEDDING_DIMENSION),
    ]
    schema = CollectionSchema(fields, description="CRM 知识库")
    col = Collection(name=MILVUS_COLLECTION, schema=schema)

    # 创建 IVF_FLAT 索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    col.create_index(field_name="embedding", index_params=index_params)
    col.load()
    print(f"[Milvus] 创建集合 '{MILVUS_COLLECTION}' 成功")
    return col


def build_index(chunks: List[Dict[str, Any]]) -> int:
    """
    将文档片段向量化并写入 Milvus
    如果集合已存在，先删除重建（全量更新）
    返回写入的条数
    """
    connect_milvus()

    # 删除旧集合（全量重建）
    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)
        print(f"[Milvus] 已删除旧集合 '{MILVUS_COLLECTION}'")

    col = ensure_collection()

    MAX_BYTES = 8000  # UTF-8 字节数上限，留余量低于 Schema max_length=8192

    def truncate_to_bytes(s: str, max_bytes: int) -> str:
        """按 UTF-8 字节数截断，避免截断多字节字符"""
        encoded = s.encode("utf-8")
        if len(encoded) <= max_bytes:
            return s
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    texts       = [truncate_to_bytes(c["text"], MAX_BYTES)       for c in chunks]
    sources     = [c.get("source", "")                           for c in chunks]
    chunk_ids   = [str(c.get("chunk_id", i))                     for i, c in enumerate(chunks)]
    chunk_types = [c.get("type", "activity")                     for c in chunks]
    dates       = [c.get("date", "")                             for c in chunks]
    companies   = [c.get("company", "")                          for c in chunks]
    owners      = [c.get("owner", "")                            for c in chunks]

    # 统计截断情况
    truncated = sum(1 for c in chunks if len(c["text"].encode("utf-8")) > MAX_BYTES)
    if truncated:
        print(f"  [警告] {truncated} 个片段超过 {MAX_BYTES} 字节，已截断")

    print(f"[向量化] 开始向量化 {len(texts)} 个片段...")
    embeddings = embed_texts(texts)

    # 写入 Milvus
    col.insert([texts, sources, chunk_ids, chunk_types, dates, companies, owners, embeddings])
    col.flush()
    count = col.num_entities
    print(f"[Milvus] 写入完成，集合共 {count} 条记录")
    return count


def get_distinct_values(field: str) -> List[str]:
    """
    从 Milvus 全量遍历，返回某个字段的所有去重非空值。
    适用于 company / owner 等元数据字段的统计查询。
    """
    connect_milvus()
    col = ensure_collection()
    total = col.num_entities
    if total == 0:
        return []

    PAGE = 1000  # 每次分页拉取数量（Milvus query limit 上限 16384，保守取 1000）
    seen: set = set()
    offset = 0

    while offset < total:
        rows = col.query(
            expr=f"{field} != ''",
            output_fields=[field],
            limit=PAGE,
            offset=offset,
        )
        if not rows:
            break
        for row in rows:
            val = row.get(field, "").strip()
            if val:
                seen.add(val)
        offset += PAGE

    return sorted(seen)


def get_aggregate_stats() -> Dict[str, Any]:
    """返回知识库聚合统计：客户列表、负责人列表、总条数"""
    companies = get_distinct_values("company")
    owners = get_distinct_values("owner")
    connect_milvus()
    col = ensure_collection()
    return {
        "total_chunks": col.num_entities,
        "company_count": len(companies),
        "companies": companies,
        "owner_count": len(owners),
        "owners": owners,
    }


def get_field_activity_counts(field: str, top_n: int = 0) -> List[Dict[str, Any]]:
    """
    统计每个字段值的活动记录数量，返回按数量降序排列的列表。
    适用于 "活跃度前N客户" 等排名问题。

    field  : 要统计的元数据字段，通常为 'company' 或 'owner'
    top_n  : 返回前 N 条，0 表示返回全部
    返回：[{'value': str, 'count': int}, ...]，已按 count 降序
    """
    connect_milvus()
    col = ensure_collection()
    total = col.num_entities
    if total == 0:
        return []

    PAGE = 1000
    count_map: Dict[str, int] = {}
    offset = 0

    while offset < total:
        rows = col.query(
            expr=f"{field} != ''",
            output_fields=[field],
            limit=PAGE,
            offset=offset,
        )
        if not rows:
            break
        for row in rows:
            val = row.get(field, "").strip()
            if val:
                count_map[val] = count_map.get(val, 0) + 1
        offset += PAGE

    sorted_list = sorted(count_map.items(), key=lambda x: x[1], reverse=True)
    result = [{"value": v, "count": c} for v, c in sorted_list]
    if top_n > 0:
        result = result[:top_n]
    return result


def query_by_metadata(
    owner: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    纯元数据查询（不走向量检索），直接用 Milvus expr 精确过滤。
    适合「某人上周/本月的活动记录」这类需要返回所有匹配记录的问题。

    owner    : 负责人名，支持部分匹配（使用 like '%%名字%%'）
    date_from: 起始日期，格式 YYYY-MM-DD
    date_to  : 截止日期，格式 YYYY-MM-DD
    limit    : 最多返回条数（默认 200）
    """
    connect_milvus()
    col = ensure_collection()

    exprs = []
    if owner:
        # Milvus like: 单个 % 即通配
        escaped = owner.replace("'", "\'")
        exprs.append(f"owner like '%{escaped}%'")
    if date_from:
        exprs.append(f"date >= '{date_from}'")
    if date_to:
        exprs.append(f"date <= '{date_to}'")

    expr = " and ".join(exprs) if exprs else "date != ''"

    rows = col.query(
        expr=expr,
        output_fields=["text", "source", "chunk_id", "chunk_type", "date", "company", "owner"],
        limit=limit,
    )

    # 按日期降序排列（最新的在前）
    rows.sort(key=lambda r: r.get("date", ""), reverse=True)

    return [
        {
            "text":       r.get("text", ""),
            "source":     r.get("source", ""),
            "chunk_id":   r.get("chunk_id", ""),
            "chunk_type": r.get("chunk_type", ""),
            "date":       r.get("date", ""),
            "company":    r.get("company", ""),
            "owner":      r.get("owner", ""),
            "score":      1.0,  # 精确匹配，满分
        }
        for r in rows
    ]


def query_by_company_keyword(
    keyword: str,
    owner: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    按 company 字段关键词模糊匹配查询，适用于"某行业所有客户记录"检索。
    例如：keyword="烟" 可匹配所有卷烟厂/烟草公司的记录。
    优先返回 chunk_type=activity 的完整活动记录，去除重复父块。
    """
    connect_milvus()
    col = ensure_collection()

    escaped = keyword.replace("'", "\\'")
    exprs = [f"company like '%{escaped}%'"]
    if owner:
        escaped_owner = owner.replace("'", "\\'")
        exprs.append(f"owner like '%{escaped_owner}%'")
    if date_from:
        exprs.append(f"date >= '{date_from}'")
    if date_to:
        exprs.append(f"date <= '{date_to}'")

    expr = " and ".join(exprs)

    rows = col.query(
        expr=expr,
        output_fields=["text", "source", "chunk_id", "chunk_type", "date", "company", "owner"],
        limit=limit,
    )

    # 优先保留 chunk_type=activity 的完整块；如果同一 chunk_id 有父子块，去重只保留父块
    seen_chunk_ids: set = set()
    result = []
    # 先遍历父块
    for r in sorted(rows, key=lambda x: x.get("date", ""), reverse=True):
        ctype = r.get("chunk_type", "")
        cid = r.get("chunk_id", "")
        if ctype == "activity":
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                result.append({
                    "text":       r.get("text", ""),
                    "source":     r.get("source", ""),
                    "chunk_id":   cid,
                    "chunk_type": ctype,
                    "date":       r.get("date", ""),
                    "company":    r.get("company", ""),
                    "owner":      r.get("owner", ""),
                    "score":      1.0,
                })
    # 补充无父块的子块（chunk_type=activity_part，父 chunk_id 未被收录的）
    for r in sorted(rows, key=lambda x: x.get("date", ""), reverse=True):
        ctype = r.get("chunk_type", "")
        cid = r.get("chunk_id", "")
        if ctype == "activity_part":
            parent_id = cid.split("_sub")[0] if "_sub" in cid else ""
            if parent_id not in seen_chunk_ids and cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                result.append({
                    "text":       r.get("text", ""),
                    "source":     r.get("source", ""),
                    "chunk_id":   cid,
                    "chunk_type": ctype,
                    "date":       r.get("date", ""),
                    "company":    r.get("company", ""),
                    "owner":      r.get("owner", ""),
                    "score":      1.0,
                })

    return result


def search(query: str, top_k: int = TOP_K, expr: str = "") -> List[Dict[str, Any]]:
    """
    相似度检索：将 query 向量化，在 Milvus 中检索最相关片段。
    可选传入 expr 实现向量检索 + 元数据过滤的混合查询。
    返回 [{text, source, score, ...}, ...]
    """
    connect_milvus()
    col = ensure_collection()

    query_embedding = embed_texts([query])[0]

    results = col.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        expr=expr if expr else None,
        output_fields=["text", "source", "chunk_id", "chunk_type", "date", "company", "owner"],
    )

    hits = []
    for hit in results[0]:
        score = hit.score  # COSINE: 1.0 最相似
        if score >= SCORE_THRESHOLD:
            hits.append({
                "text":       hit.entity.get("text"),
                "source":     hit.entity.get("source"),
                "chunk_id":   hit.entity.get("chunk_id"),
                "chunk_type": hit.entity.get("chunk_type"),
                "date":       hit.entity.get("date"),
                "company":    hit.entity.get("company"),
                "owner":      hit.entity.get("owner"),
                "score":      round(float(score), 4),
            })

    return hits
