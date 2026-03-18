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
    MILVUS_COLLECTION, TOP_K, SCORE_THRESHOLD, BASE_DIR,
    SUMMARY_RAG_ENABLED,
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


# 固定 schema 字段（不含 embedding）
_SCHEMA_FIELDS = {"text", "source", "chunk_id", "chunk_type", "date", "company", "owner", "title", "tags"}


def ensure_collection() -> Collection:
    """确保 Milvus Collection 存在，不存在则创建。
    开启 enable_dynamic_field=True，支持写入任意额外字段。
    """
    if utility.has_collection(MILVUS_COLLECTION):
        col = Collection(MILVUS_COLLECTION)
        col.load()
        return col

    # 定义 Schema（固定字段 + dynamic field 支持额外自定义字段）
    fields = [
        FieldSchema(name="id",         dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name="text",        dtype=DataType.VARCHAR,       max_length=8192),
        FieldSchema(name="source",      dtype=DataType.VARCHAR,       max_length=256),
        FieldSchema(name="chunk_id",    dtype=DataType.VARCHAR,       max_length=64),
        FieldSchema(name="chunk_type",  dtype=DataType.VARCHAR,       max_length=32),
        FieldSchema(name="date",        dtype=DataType.VARCHAR,       max_length=64),
        FieldSchema(name="company",     dtype=DataType.VARCHAR,       max_length=256),
        FieldSchema(name="owner",       dtype=DataType.VARCHAR,       max_length=128),
        FieldSchema(name="title",       dtype=DataType.VARCHAR,       max_length=512),
        FieldSchema(name="tags",        dtype=DataType.VARCHAR,       max_length=512),
        FieldSchema(name="embedding",   dtype=DataType.FLOAT_VECTOR,  dim=EMBEDDING_DIMENSION),
    ]
    schema = CollectionSchema(
        fields,
        description="CRM 知识库",
        enable_dynamic_field=True,  # 允许写入任意额外字段
    )
    col = Collection(name=MILVUS_COLLECTION, schema=schema)

    # 创建 IVF_FLAT 索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    col.create_index(field_name="embedding", index_params=index_params)
    col.load()
    print(f"[Milvus] 创建集合 '{MILVUS_COLLECTION}'（enable_dynamic_field=True）成功")
    return col


def clear_index() -> bool:
    """清空向量库：删除 Milvus Collection，不重建。
    返回 True 表示确实删除了集合，False 表示集合本就不存在。
    """
    connect_milvus()
    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)
        print(f"[Milvus] 已清空集合 '{MILVUS_COLLECTION}'")
        return True
    print(f"[Milvus] 集合 '{MILVUS_COLLECTION}' 不存在，无需清空")
    return False


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

    truncated = sum(1 for c in chunks if len(c["text"].encode("utf-8")) > MAX_BYTES)
    if truncated:
        print(f"  [警告] {truncated} 个片段超过 {MAX_BYTES} 字节，已截断")

    print(f"[向量化] 开始向量化 {len(chunks)} 个片段...")
    texts = [truncate_to_bytes(c["text"], MAX_BYTES) for c in chunks]
    embeddings = embed_texts(texts)

    # 构建行列表（每行为 dict，dynamic field 自动处理额外字段）
    rows = []
    for i, c in enumerate(chunks):
        row: Dict[str, Any] = {
            "text":       truncate_to_bytes(c["text"], MAX_BYTES),
            "source":     str(c.get("source", ""))[:256],
            "chunk_id":   str(c.get("chunk_id", i))[:64],
            "chunk_type": str(c.get("type", "activity"))[:32],
            "date":       str(c.get("date", ""))[:64],
            "company":    str(c.get("company", ""))[:256],
            "owner":      str(c.get("owner", ""))[:128],
            "title":      str(c.get("title", ""))[:512],
            "tags":       str(c.get("tags", ""))[:512],
            "embedding":  embeddings[i],
        }
        # 额外自定义字段（非固定 schema 字段）写入 dynamic field
        for k, v in c.items():
            if k not in _SCHEMA_FIELDS and k not in ("text", "type", "embedding"):
                row[k] = str(v)[:512] if v is not None else ""
        rows.append(row)

    # 分批写入 Milvus（避免 gRPC 64MB 消息上限）
    INSERT_BATCH = 500
    for b in range(0, len(rows), INSERT_BATCH):
        col.insert(rows[b: b + INSERT_BATCH])
        done = min(b + INSERT_BATCH, len(rows))
        print(f"  [Milvus] 已写入 {done}/{len(rows)} 条")
    col.flush()

    # 摘要向量检索：生成摘要并追加写入
    if SUMMARY_RAG_ENABLED:
        from app.document_loader import generate_summaries
        summary_chunks = generate_summaries(chunks)
        if summary_chunks:
            s_texts = [truncate_to_bytes(c["text"], MAX_BYTES) for c in summary_chunks]
            print(f"[向量化] 开始向量化 {len(s_texts)} 条摘要...")
            s_embeddings = embed_texts(s_texts)
            s_rows = []
            for j, c in enumerate(summary_chunks):
                s_row: Dict[str, Any] = {
                    "text":       truncate_to_bytes(c["text"], MAX_BYTES),
                    "source":     str(c.get("source", ""))[:256],
                    "chunk_id":   str(c.get("chunk_id", ""))[:64],
                    "chunk_type": str(c.get("type", "summary"))[:32],
                    "date":       str(c.get("date", ""))[:64],
                    "company":    str(c.get("company", ""))[:256],
                    "owner":      str(c.get("owner", ""))[:128],
                    "title":      str(c.get("title", ""))[:512],
                    "tags":       str(c.get("tags", ""))[:512],
                    "embedding":  s_embeddings[j],
                }
                for k, v in c.items():
                    if k not in _SCHEMA_FIELDS and k not in ("text", "type", "embedding"):
                        s_row[k] = str(v)[:512] if v is not None else ""
                s_rows.append(s_row)
            for b in range(0, len(s_rows), INSERT_BATCH):
                col.insert(s_rows[b: b + INSERT_BATCH])
            col.flush()
            print(f"[Milvus] 追加写入 {len(s_rows)} 条摘要记录")

    count = col.num_entities
    print(f"[Milvus] 写入完成，集合共 {count} 条记录")
    return count


def insert_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    增量插入 chunks 到 Milvus（不清空现有数据）。
    支持固定 schema 字段（date/company/owner/title/tags）及任意自定义字段（via dynamic field）。
    返回实际写入条数。
    """
    connect_milvus()
    col = ensure_collection()

    # ── 去重：查询 Milvus 中已存在的 chunk_id，跳过重复，节省 embedding token ──
    all_ids = [str(c.get("chunk_id", "")) for c in chunks]
    existing_ids: set = set()
    QUERY_BATCH = 200
    for bi in range(0, len(all_ids), QUERY_BATCH):
        batch = [cid for cid in all_ids[bi: bi + QUERY_BATCH] if cid]
        if not batch:
            continue
        expr = "chunk_id in [" + ", ".join(f'"{cid}"' for cid in batch) + "]"
        try:
            rows = col.query(expr=expr, output_fields=["chunk_id"], limit=len(batch))
            for r in rows:
                existing_ids.add(r["chunk_id"])
        except Exception as e:
            print(f"  [去重] 查询失败（跳过去重）: {e}")

    new_chunks = [c for c in chunks if str(c.get("chunk_id", "")) not in existing_ids]
    skipped = len(chunks) - len(new_chunks)
    if skipped:
        print(f"  [去重] 跳过 {skipped} 条已存在，新增 {len(new_chunks)} 条")
    if not new_chunks:
        print("[insert_chunks] 全部重复，跳过写入")
        return 0
    chunks = new_chunks
    # ─────────────────────────────────────────────────────────────────────────

    MAX_BYTES = 8000

    def truncate_to_bytes(s: str, max_bytes: int) -> str:
        encoded = s.encode("utf-8")
        if len(encoded) <= max_bytes:
            return s
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    truncated = sum(1 for c in chunks if len(c["text"].encode("utf-8")) > MAX_BYTES)
    if truncated:
        print(f"  [警告] {truncated} 个片段超过 {MAX_BYTES} 字节，已截断")

    print(f"[向量化] 增量写入 {len(chunks)} 个片段...")
    texts = [truncate_to_bytes(c["text"], MAX_BYTES) for c in chunks]
    embeddings = embed_texts(texts)

    rows = []
    for i, c in enumerate(chunks):
        row: Dict[str, Any] = {
            "text":       truncate_to_bytes(c["text"], MAX_BYTES),
            "source":     str(c.get("source", ""))[:256],
            "chunk_id":   str(c.get("chunk_id", i))[:64],
            "chunk_type": str(c.get("type", "uploaded"))[:32],
            "date":       str(c.get("date", ""))[:64],
            "company":    str(c.get("company", ""))[:256],
            "owner":      str(c.get("owner", ""))[:128],
            "title":      str(c.get("title", ""))[:512],
            "tags":       str(c.get("tags", ""))[:512],
            "embedding":  embeddings[i],
        }
        # 额外自定义字段写入 dynamic field
        for k, v in c.items():
            if k not in _SCHEMA_FIELDS and k not in ("text", "type", "embedding"):
                row[k] = str(v)[:512] if v is not None else ""
        rows.append(row)

    # 分批写入，避免 gRPC 64MB 消息上限
    INSERT_BATCH = 500
    for b in range(0, len(rows), INSERT_BATCH):
        col.insert(rows[b: b + INSERT_BATCH])
        done = min(b + INSERT_BATCH, len(rows))
        print(f"  [Milvus] 已写入 {done}/{len(rows)} 条")
    col.flush()
    print(f"[Milvus] 增量写入完成，共 {len(rows)} 条")
    return len(rows)


def get_distinct_values(field: str) -> List[str]:
    """
    从 Milvus 全量遍历，返回某个字段的所有去重非空值。
    使用 query_iterator 避免 Milvus offset+limit ≤ 16384 的限制。
    """
    connect_milvus()
    col = ensure_collection()
    if col.num_entities == 0:
        return []

    seen: set = set()
    iterator = col.query_iterator(
        expr=f"{field} != ''",
        output_fields=[field],
        batch_size=1000,
    )
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        for row in batch:
            val = row.get(field, "").strip()
            if val:
                seen.add(val)

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

    count_map: Dict[str, int] = {}
    iterator = col.query_iterator(
        expr=f"{field} != ''",
        output_fields=[field],
        batch_size=1000,
    )
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        for row in batch:
            val = row.get(field, "").strip()
            if val:
                count_map[val] = count_map.get(val, 0) + 1

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


def search(query: str, top_k: int = TOP_K, expr: str = "") -> List[Dict[str, Any]]:
    """
    相似度检索：将 query 向量化，在 Milvus 中检索最相关片段。
    可选传入 expr 实现向量检索 + 元数据过滤的混合查询。
    当 SUMMARY_RAG_ENABLED 时，自动只检索 summary 类型的 chunk。
    返回 [{text, source, score, ...}, ...]
    """
    connect_milvus()
    col = ensure_collection()

    # 摘要模式：只检索 summary chunk
    if SUMMARY_RAG_ENABLED:
        summary_filter = 'chunk_type == "summary"'
        expr = f"({expr}) and {summary_filter}" if expr else summary_filter

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


def fetch_originals(summary_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    根据 summary 检索命中结果，查回对应的原始 activity 记录。
    通过 chunk_id + source 关联，chunk_type 为 activity 或 activity_part。
    返回的列表保持与 summary_hits 相同的顺序，每条包含原始文本和元数据。
    """
    if not summary_hits:
        return []

    connect_milvus()
    col = ensure_collection()

    originals = []
    for hit in summary_hits:
        chunk_id = hit.get("chunk_id", "")
        source = hit.get("source", "")
        expr = (
            f'chunk_id == "{chunk_id}" '
            f'and source == "{source}" '
            f'and (chunk_type == "activity" or chunk_type == "activity_part")'
        )
        rows = col.query(
            expr=expr,
            output_fields=["text", "source", "chunk_id", "chunk_type", "date", "company", "owner"],
            limit=10,
        )
        if rows:
            # 拼合同一活动的所有 part
            rows.sort(key=lambda r: r.get("chunk_id", ""))
            full_text = "\n\n".join(r.get("text", "") for r in rows)
            originals.append({
                "text":       full_text,
                "source":     hit.get("source", ""),
                "chunk_id":   hit.get("chunk_id", ""),
                "chunk_type": "activity",
                "date":       hit.get("date", ""),
                "company":    hit.get("company", ""),
                "owner":      hit.get("owner", ""),
                "score":      hit.get("score", 0),
            })
        else:
            # 找不到原文，回退使用摘要本身
            originals.append(hit)

    return originals
