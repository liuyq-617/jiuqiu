"""
回答质量评价体系
===============
包含三个子系统：

1. SQLite 持久化存储
   - 每次问答自动写入 data/feedback.db
   - 支持用户点赞/踩（thumbs +1/-1）
   - 记录响应耗时

2. LLM-as-Judge 自动评分（后台线程）
   - 相关性 / 完整性 / 准确性，1~5 分
   - 用户提交反馈后异步触发，不阻塞响应

3. 统计看板数据
   - 总问答数、好评率、平均分、平均响应时间
   - 近 30 条记录明细
"""
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from app.config import (
    CHAT_API_KEY, CHAT_BASE_URL, OPENAI_CHAT_MODEL, CHAT_API_MODE, DATA_DIR
)

logger = logging.getLogger("crm_feedback")

# ========== 数据库初始化 ==========
_DB_PATH = DATA_DIR / "feedback.db"
_db_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表（首次启动时调用）"""
    DATA_DIR.mkdir(exist_ok=True)
    with _db_lock, _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS qa_log (
                id              TEXT PRIMARY KEY,
                question        TEXT NOT NULL,
                answer          TEXT NOT NULL,
                sources_json    TEXT DEFAULT '[]',
                response_ms     INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL,
                thumbs          INTEGER DEFAULT NULL,   -- 1 赞 / -1 踩 / NULL 未评价
                llm_relevance   REAL DEFAULT NULL,      -- LLM 自动评分 1.0~5.0
                llm_completeness REAL DEFAULT NULL,
                llm_accuracy    REAL DEFAULT NULL,
                llm_comment     TEXT DEFAULT NULL,
                llm_judged_at   TEXT DEFAULT NULL,
                manual_relevance    REAL DEFAULT NULL,  -- 用户手动打分 1~5
                manual_completeness REAL DEFAULT NULL,
                manual_accuracy     REAL DEFAULT NULL,
                manual_comment      TEXT DEFAULT NULL,
                manual_scored_at    TEXT DEFAULT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_qa_created ON qa_log(created_at DESC);

            CREATE TABLE IF NOT EXISTS prompt_candidates (
                id               TEXT PRIMARY KEY,
                route            TEXT NOT NULL,
                suggestion       TEXT NOT NULL,
                sample_ids       TEXT DEFAULT '[]',
                sample_count     INTEGER DEFAULT 0,
                avg_score_before REAL DEFAULT NULL,
                avg_score_after  REAL DEFAULT NULL,
                status           TEXT DEFAULT 'pending',
                created_at       TEXT NOT NULL,
                reviewed_at      TEXT DEFAULT NULL,
                review_note      TEXT DEFAULT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pc_status
                ON prompt_candidates(status, created_at DESC);
        """)
        # 兼容旧库：按需 ALTER TABLE 追加手动打分列
        existing = {row[1] for row in conn.execute("PRAGMA table_info(qa_log)").fetchall()}
        for col_def in [
            ("manual_relevance",    "REAL DEFAULT NULL"),
            ("manual_completeness", "REAL DEFAULT NULL"),
            ("manual_accuracy",     "REAL DEFAULT NULL"),
            ("manual_comment",      "TEXT DEFAULT NULL"),
            ("manual_scored_at",    "TEXT DEFAULT NULL"),
        ]:
            if col_def[0] not in existing:
                conn.execute(f"ALTER TABLE qa_log ADD COLUMN {col_def[0]} {col_def[1]}")
    logger.info("[feedback] 数据库初始化完成: %s", _DB_PATH)


# ========== 写入 / 更新 ==========

def save_qa(question: str, answer: str, sources: list, response_ms: int) -> str:
    """保存一条问答记录，返回 answer_id（UUID）"""
    aid = str(uuid.uuid4())
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with _db_lock, _get_conn() as conn:
        conn.execute(
            """INSERT INTO qa_log (id, question, answer, sources_json, response_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (aid, question, answer, json.dumps(sources, ensure_ascii=False), response_ms, created_at),
        )
    logger.debug("[feedback] 已保存问答 id=%s q=%s", aid, question[:30])
    return aid


def save_thumbs(answer_id: str, thumbs: int) -> bool:
    """更新用户反馈（thumbs=1 赞 / -1 踩），返回是否找到记录"""
    if thumbs not in (1, -1):
        return False
    with _db_lock, _get_conn() as conn:
        cur = conn.execute(
            "UPDATE qa_log SET thumbs=? WHERE id=?", (thumbs, answer_id)
        )
        return cur.rowcount > 0


def save_manual_scores(
    answer_id: str,
    relevance: float,
    completeness: float,
    accuracy: float,
    comment: str = "",
) -> bool:
    """
    保存用户手动打分（1~5 分，支持整数）。
    同时将 thumbs 自动更新：平均分 >= 3.5 → 赞，< 3.5 → 踩。
    返回是否找到对应记录。
    """
    for v in (relevance, completeness, accuracy):
        if not (1 <= v <= 5):
            return False

    avg = (relevance + completeness + accuracy) / 3
    auto_thumbs = 1 if avg >= 3.5 else -1
    scored_at = time.strftime("%Y-%m-%d %H:%M:%S")

    with _db_lock, _get_conn() as conn:
        cur = conn.execute(
            """UPDATE qa_log SET
                manual_relevance=?, manual_completeness=?, manual_accuracy=?,
                manual_comment=?, manual_scored_at=?,
                thumbs=COALESCE(thumbs, ?)
               WHERE id=?""",
            (relevance, completeness, accuracy, comment, scored_at, auto_thumbs, answer_id),
        )
        return cur.rowcount > 0


def _fetch_qa(answer_id: str) -> Optional[dict]:
    with _db_lock, _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM qa_log WHERE id=?", (answer_id,)
        ).fetchone()
    return dict(row) if row else None


# ========== LLM-as-Judge ==========

_JUDGE_SYSTEM = """你是一个严格的 RAG 系统质量评估专家。
请根据用户问题和 AI 回答，对回答从以下三个维度打分（1~5 分，支持小数），并给出简短评语：
- relevance（相关性）：回答是否切题，是否基于问题作答
- completeness（完整性）：是否完整覆盖了问题各个方面
- accuracy（准确性）：回答内容是否准确、引用是否有据可查

以 JSON 格式回复，格式如下（不要加代码块标记）：
{"relevance": 4.5, "completeness": 3.0, "accuracy": 4.0, "comment": "..."}
"""


def _call_judge_llm(question: str, answer: str) -> Optional[dict]:
    """同步调用 LLM 进行评分，返回 {relevance, completeness, accuracy, comment}"""
    base = CHAT_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"

    if CHAT_API_MODE == "responses":
        url = base + "/responses"
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "instructions": _JUDGE_SYSTEM,
            "input": [{"role": "user", "content": f"问题：{question}\n\n回答：{answer}"}],
            "temperature": 0.1,
            "max_output_tokens": 256,
            "stream": False,
        }
    else:
        url = base + "/chat/completions"
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": f"问题：{question}\n\n回答：{answer}"},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
            "stream": False,
        }

    headers = {
        "Authorization": f"Bearer {CHAT_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, headers=headers, json=payload)
            data = resp.json()

        if CHAT_API_MODE == "responses":
            text = data["output"][0]["content"][0]["text"]
        else:
            text = data["choices"][0]["message"]["content"]

        result = json.loads(text.strip())
        # 数值范围校验
        for key in ("relevance", "completeness", "accuracy"):
            result[key] = max(1.0, min(5.0, float(result[key])))
        return result
    except Exception as e:
        logger.warning("[feedback] LLM 评分失败: %s", e)
        return None


def _do_judge(answer_id: str) -> None:
    """后台线程：拉取问答记录 → 调用 LLM 评分 → 写回数据库"""
    qa = _fetch_qa(answer_id)
    if not qa:
        return
    if qa["llm_judged_at"]:
        return  # 已评过，跳过

    logger.info("[feedback] 开始 LLM 评分 id=%s", answer_id)
    scores = _call_judge_llm(qa["question"], qa["answer"])
    if not scores:
        return

    judged_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with _db_lock, _get_conn() as conn:
        conn.execute(
            """UPDATE qa_log SET
                llm_relevance=?, llm_completeness=?, llm_accuracy=?,
                llm_comment=?, llm_judged_at=?
               WHERE id=?""",
            (
                scores["relevance"],
                scores["completeness"],
                scores["accuracy"],
                scores.get("comment", ""),
                judged_at,
                answer_id,
            ),
        )
    logger.info(
        "[feedback] LLM 评分完成 id=%s rel=%.1f comp=%.1f acc=%.1f",
        answer_id, scores["relevance"], scores["completeness"], scores["accuracy"],
    )


def trigger_judge(answer_id: str) -> None:
    """异步触发 LLM-as-Judge（不阻塞请求）"""
    t = threading.Thread(target=_do_judge, args=(answer_id,), daemon=True, name=f"judge-{answer_id[:8]}")
    t.start()


# ========== 统计看板 ==========

def get_stats() -> dict:
    """返回全量统计数据 + 近 30 条明细"""
    with _db_lock, _get_conn() as conn:
        # 总体指标
        total = conn.execute("SELECT COUNT(*) FROM qa_log").fetchone()[0]
        thumbs_up = conn.execute("SELECT COUNT(*) FROM qa_log WHERE thumbs=1").fetchone()[0]
        thumbs_down = conn.execute("SELECT COUNT(*) FROM qa_log WHERE thumbs=-1").fetchone()[0]
        avg_ms = conn.execute("SELECT AVG(response_ms) FROM qa_log WHERE response_ms>0").fetchone()[0]
        avg_rel = conn.execute("SELECT AVG(llm_relevance) FROM qa_log WHERE llm_relevance IS NOT NULL").fetchone()[0]
        avg_comp = conn.execute("SELECT AVG(llm_completeness) FROM qa_log WHERE llm_completeness IS NOT NULL").fetchone()[0]
        avg_acc = conn.execute("SELECT AVG(llm_accuracy) FROM qa_log WHERE llm_accuracy IS NOT NULL").fetchone()[0]
        judged_count = conn.execute("SELECT COUNT(*) FROM qa_log WHERE llm_judged_at IS NOT NULL").fetchone()[0]

        # 手动打分均值
        m_rel  = conn.execute("SELECT AVG(manual_relevance)    FROM qa_log WHERE manual_relevance IS NOT NULL").fetchone()[0]
        m_comp = conn.execute("SELECT AVG(manual_completeness) FROM qa_log WHERE manual_completeness IS NOT NULL").fetchone()[0]
        m_acc  = conn.execute("SELECT AVG(manual_accuracy)     FROM qa_log WHERE manual_accuracy IS NOT NULL").fetchone()[0]
        manual_count = conn.execute("SELECT COUNT(*) FROM qa_log WHERE manual_scored_at IS NOT NULL").fetchone()[0]

        # 近 30 条明细
        rows = conn.execute(
            """SELECT id, question, answer, response_ms, created_at,
                      thumbs, llm_relevance, llm_completeness, llm_accuracy, llm_comment, llm_judged_at,
                      manual_relevance, manual_completeness, manual_accuracy, manual_comment, manual_scored_at
               FROM qa_log ORDER BY created_at DESC LIMIT 30"""
        ).fetchall()

    rated = thumbs_up + thumbs_down
    return {
        "total": total,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "thumbs_up_rate": round(thumbs_up / rated, 3) if rated > 0 else None,
        "avg_response_ms": round(avg_ms) if avg_ms else None,
        "judged_count": judged_count,
        "avg_llm_relevance":    round(avg_rel, 2)  if avg_rel  else None,
        "avg_llm_completeness": round(avg_comp, 2) if avg_comp else None,
        "avg_llm_accuracy":     round(avg_acc, 2)  if avg_acc  else None,
        "manual_count": manual_count,
        "avg_manual_relevance":    round(m_rel, 2)  if m_rel  else None,
        "avg_manual_completeness": round(m_comp, 2) if m_comp else None,
        "avg_manual_accuracy":     round(m_acc, 2)  if m_acc  else None,
        "recent": [
            {
                "id": r["id"],
                "question": r["question"][:60] + ("…" if len(r["question"]) > 60 else ""),
                "answer_preview": r["answer"][:80] + ("…" if len(r["answer"]) > 80 else ""),
                "response_ms": r["response_ms"],
                "created_at": r["created_at"],
                "thumbs": r["thumbs"],
                "llm_relevance": r["llm_relevance"],
                "llm_completeness": r["llm_completeness"],
                "llm_accuracy": r["llm_accuracy"],
                "llm_comment": r["llm_comment"],
                "llm_judged_at": r["llm_judged_at"],
                "manual_relevance": r["manual_relevance"],
                "manual_completeness": r["manual_completeness"],
                "manual_accuracy": r["manual_accuracy"],
                "manual_comment": r["manual_comment"],
                "manual_scored_at": r["manual_scored_at"],
            }
            for r in rows
        ],
    }


# ========== Prompt 候选管理 ==========

def list_prompt_candidates(status: Optional[str] = None) -> list:
    """
    返回 prompt_candidates 表中的候选记录。
    status 不传则返回全部；传 'pending'/'approved'/'rejected' 则按 status 过滤。
    按 created_at DESC 排序。
    """
    with _db_lock, _get_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM prompt_candidates WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM prompt_candidates ORDER BY created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def save_prompt_candidate(
    route: str,
    suggestion: str,
    sample_ids: list,
    avg_score_before: Optional[float] = None,
) -> str:
    """
    插入一条新的 prompt_candidate 记录（status='pending'），返回 UUID。
    route: 路由类型（semantic/metadata_filter/aggregate/ranking/evaluation/industry_scenario）
    suggestion: Meta-LLM 生成的完整候选 SYSTEM_PROMPT 文本
    sample_ids: 触发本次优化的低分 QA 记录 ID 列表
    avg_score_before: 这批样本的平均评分（用于对比优化效果）
    """
    cid = str(uuid.uuid4())
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with _db_lock, _get_conn() as conn:
        conn.execute(
            """INSERT INTO prompt_candidates
               (id, route, suggestion, sample_ids, sample_count,
                avg_score_before, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (
                cid,
                route,
                suggestion,
                json.dumps(sample_ids, ensure_ascii=False),
                len(sample_ids),
                avg_score_before,
                created_at,
            ),
        )
    logger.info("[feedback] 新 prompt 候选已保存 id=%s route=%s samples=%d", cid, route, len(sample_ids))
    return cid


def review_prompt_candidate(
    candidate_id: str,
    action: str,       # 'approved' 或 'rejected'
    note: str = "",
) -> bool:
    """
    审核 Prompt 候选：设置 status='approved'/'rejected'，记录 reviewed_at 和备注。
    返回 True 表示找到并更新了记录，False 表示记录不存在。
    """
    if action not in ("approved", "rejected"):
        return False
    reviewed_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with _db_lock, _get_conn() as conn:
        cur = conn.execute(
            """UPDATE prompt_candidates
               SET status=?, reviewed_at=?, review_note=?
               WHERE id=?""",
            (action, reviewed_at, note, candidate_id),
        )
        return cur.rowcount > 0
