#!/usr/bin/env python3
"""
Prompt 自动优化脚本
===================
从 feedback.db 中提取近 N 天的低分 QA 样本，按路由类型分组，
调用 Meta-LLM 分析共性缺陷并生成候选 Prompt 改进建议，
写入 prompt_candidates 表（status='pending'），等待人工审核后上线。

用法：
  cd crm_kb
  python3 scripts/prompt_optimizer.py                      # 默认：近7天 / min_samples=5
  python3 scripts/prompt_optimizer.py --min-samples 3 --days 14
  python3 scripts/prompt_optimizer.py --dry-run            # 仅报告，不写库、不调用LLM

退出码：0 = 正常完成，1 = 无低分记录或执行异常
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── sys.path（与 benchmark.py 保持一致）────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent   # crm_kb/scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent              # crm_kb/
sys.path.insert(0, str(_PROJECT_ROOT))
# ───────────────────────────────────────────────────────────────────────────

import httpx

from app.config import (
    CHAT_API_KEY, CHAT_BASE_URL, OPENAI_CHAT_MODEL, CHAT_API_MODE,
)
from app.feedback import _get_conn, _db_lock, save_prompt_candidate
from app.rag import (
    SYSTEM_PROMPT,
    _is_ranking_question,
    _is_evaluation_question,
    _is_aggregate_question,
    _is_industry_scenario_question,
    extract_filters,
)

logger = logging.getLogger("prompt_optimizer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s][%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── 路由常量（与 rag.py answer() 优先级顺序一致）───────────────────────────
ROUTE_RANKING         = "ranking"
ROUTE_EVALUATION      = "evaluation"
ROUTE_AGGREGATE       = "aggregate"
ROUTE_INDUSTRY        = "industry_scenario"
ROUTE_METADATA_FILTER = "metadata_filter"
ROUTE_SEMANTIC        = "semantic"
# ───────────────────────────────────────────────────────────────────────────


def detect_route(question: str) -> str:
    """
    对问题重跑 rag.py 的路由检测逻辑，返回路由类型常量。
    优先级镜像 rag.py answer()：ranking → evaluation → aggregate → industry → metadata_filter → semantic
    """
    try:
        if _is_ranking_question(question):
            return ROUTE_RANKING
        if _is_evaluation_question(question):
            return ROUTE_EVALUATION
        if _is_aggregate_question(question):
            return ROUTE_AGGREGATE
        if _is_industry_scenario_question(question):
            return ROUTE_INDUSTRY
        filters = extract_filters(question)
        if filters.get("owner") or filters.get("date_from"):
            return ROUTE_METADATA_FILTER
    except Exception as e:
        logger.warning("[optimizer] 路由检测异常，降级为 semantic: %s", e)
    return ROUTE_SEMANTIC


def fetch_low_score_records(days: int) -> List[Dict]:
    """
    查询 qa_log 中最近 days 天内 LLM 已评分且平均分 < 3.5 的记录。
    返回字段：id, question, answer, llm_relevance, llm_completeness, llm_accuracy,
              llm_comment, created_at
    """
    cutoff = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(time.time() - days * 86400),
    )
    with _db_lock, _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, question, answer,
                   llm_relevance, llm_completeness, llm_accuracy, llm_comment,
                   created_at
            FROM qa_log
            WHERE llm_judged_at IS NOT NULL
              AND created_at >= ?
              AND (llm_relevance + llm_completeness + llm_accuracy) / 3.0 < 3.5
            ORDER BY created_at DESC
            """,
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def group_by_route(records: List[Dict]) -> Dict[str, List[Dict]]:
    """将记录列表按路由类型分组，返回 {route: [record, ...]}"""
    groups: Dict[str, List[Dict]] = {}
    for rec in records:
        route = detect_route(rec["question"])
        groups.setdefault(route, []).append(rec)
    return groups


def _avg_score(records: List[Dict]) -> Optional[float]:
    """计算这批记录各自 avg(rel,comp,acc) 的总平均，无记录返回 None"""
    if not records:
        return None
    total = sum(
        (r["llm_relevance"] + r["llm_completeness"] + r["llm_accuracy"]) / 3.0
        for r in records
    )
    return round(total / len(records), 3)


def _call_meta_llm(
    route: str,
    current_prompt: str,
    samples: List[Dict],
) -> Optional[str]:
    """
    调用 Meta-LLM，输入当前 SYSTEM_PROMPT + 低分样本，要求生成改进后的完整 SYSTEM_PROMPT。
    HTTP 调用模式与 feedback.py 的 _call_judge_llm() 完全一致。
    返回改进后的 prompt 字符串，失败返回 None。
    """
    sample_text = "\n\n".join(
        f"--- 样本 {i + 1} ---\n"
        f"问题: {s['question']}\n"
        f"回答（前400字）: {s['answer'][:400]}{'...' if len(s['answer']) > 400 else ''}\n"
        f"Judge评语: {s.get('llm_comment', '（无）')}\n"
        f"评分: 相关性={s['llm_relevance']:.1f}  完整性={s['llm_completeness']:.1f}  "
        f"准确性={s['llm_accuracy']:.1f}  "
        f"均分={((s['llm_relevance'] + s['llm_completeness'] + s['llm_accuracy']) / 3):.2f}"
        for i, s in enumerate(samples[:10])
    )

    meta_system = (
        "你是一个 RAG 系统的 Prompt 优化专家。\n"
        "你会收到：① 当前 SYSTEM_PROMPT；② 一批低分 QA 样本（含 LLM 评委评语）；③ 这批问题所属的路由类型。\n"
        "你的任务：\n"
        "1. 分析这批低分样本的共性缺陷（结合评委评语）\n"
        "2. 针对该路由类型的典型问题，给出有针对性的改进\n"
        "3. 返回改进后的**完整 SYSTEM_PROMPT**（保留原有优秀指令，仅修改/补充有问题的部分）\n\n"
        "要求：\n"
        "- 直接返回完整的新 SYSTEM_PROMPT 文本\n"
        "- 不要包含任何解释、前言或代码块标记\n"
        "- 改进必须针对低分样本暴露的具体问题，不要泛泛而谈\n"
    )

    user_content = (
        f"【路由类型】{route}\n\n"
        f"【当前 SYSTEM_PROMPT】\n{current_prompt}\n\n"
        f"【低分 QA 样本（共 {len(samples[:10])} 条）】\n{sample_text}\n\n"
        "请给出改进后的完整 SYSTEM_PROMPT："
    )

    base = CHAT_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"

    if CHAT_API_MODE == "responses":
        url = base + "/responses"
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "instructions": meta_system,
            "input": [{"role": "user", "content": user_content}],
            "temperature": 0.4,
            "max_output_tokens": 2048,
            "stream": False,
        }
    else:
        url = base + "/chat/completions"
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": meta_system},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.4,
            "max_tokens": 2048,
            "stream": False,
        }

    headers = {
        "Authorization": f"Bearer {CHAT_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
            data = resp.json()

        if CHAT_API_MODE == "responses":
            text = data["output"][0]["content"][0]["text"]
        else:
            text = data["choices"][0]["message"]["content"]

        text = text.strip()
        if not text:
            logger.warning("[optimizer] Meta-LLM 返回空文本 route=%s", route)
            return None
        return text

    except Exception as e:
        logger.warning("[optimizer] Meta-LLM 调用失败 route=%s err=%s", route, e)
        return None


def run_optimizer(
    min_samples: int = 5,
    days: int = 7,
    dry_run: bool = False,
) -> Dict:
    """
    主入口，可被 main.py 调度器直接调用，也可作为 CLI 脚本运行。

    返回汇总字典：
    {
        "total_low_score": int,       # 查到的低分记录总数
        "routes_qualified": int,      # 达到阈值的路由数
        "candidates_written": int,    # 实际写入 DB 的候选数
        "skipped_routes": {route: count},
        "written_routes": [route, ...],
    }
    """
    logger.info(
        "[optimizer] 开始运行  min_samples=%d  days=%d  dry_run=%s",
        min_samples, days, dry_run,
    )

    records = fetch_low_score_records(days)
    logger.info("[optimizer] 查到 %d 条低分记录（近 %d 天，均分 < 3.5）", len(records), days)

    if not records:
        _print_summary(days, records, {}, [], 0, dry_run)
        return {
            "total_low_score": 0,
            "routes_qualified": 0,
            "candidates_written": 0,
            "skipped_routes": {},
            "written_routes": [],
        }

    groups = group_by_route(records)

    skipped: Dict[str, int] = {}
    written: List[str] = []
    candidates_written = 0

    for route, route_records in groups.items():
        count = len(route_records)

        if count < min_samples:
            logger.info(
                "[optimizer] 路由 %-20s  %d 条 < 阈值 %d，跳过",
                route, count, min_samples,
            )
            skipped[route] = count
            continue

        avg_before = _avg_score(route_records)
        logger.info(
            "[optimizer] 路由 %-20s  %d 条  avg_before=%.2f  → 触发优化",
            route, count, avg_before or 0,
        )

        if dry_run:
            logger.info("[optimizer] [DRY-RUN] 跳过 LLM 调用，路由 %s 标记为 written", route)
            written.append(route)
            continue

        suggestion = _call_meta_llm(
            route=route,
            current_prompt=SYSTEM_PROMPT,
            samples=route_records[:10],
        )
        if not suggestion:
            logger.warning("[optimizer] 路由 %s LLM 优化调用失败，跳过", route)
            skipped[route] = count
            continue

        sample_ids = [r["id"] for r in route_records]
        save_prompt_candidate(
            route=route,
            suggestion=suggestion,
            sample_ids=sample_ids,
            avg_score_before=avg_before,
        )
        logger.info("[optimizer] 路由 %s 候选已写入 prompt_candidates 表", route)
        written.append(route)
        candidates_written += 1

    _print_summary(days, records, skipped, written, candidates_written, dry_run)

    return {
        "total_low_score": len(records),
        "routes_qualified": len(written),
        "candidates_written": candidates_written,
        "skipped_routes": skipped,
        "written_routes": written,
    }


def _print_summary(
    days: int,
    records: List[Dict],
    skipped: Dict[str, int],
    written: List[str],
    candidates_written: int,
    dry_run: bool,
) -> None:
    """打印人类可读的汇总报告"""
    print("\n" + "=" * 60)
    print("  Prompt Optimizer 汇总报告")
    print("=" * 60)
    print(f"  扫描期间   : 最近 {days} 天")
    print(f"  低分记录数 : {len(records)}  （均分 < 3.5）")
    print(f"  路由分组数 : {len(skipped) + len(written)} 个")
    if written:
        print(f"  达标路由   : {written}")
    if skipped:
        print(f"  跳过路由   : {skipped}")
    print(f"  写入候选数 : {candidates_written}")
    if dry_run:
        print("  ⚠️  DRY-RUN 模式，未实际调用 LLM 或写入数据库")
    print("=" * 60 + "\n")


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CRM 知识库 Prompt 自动优化脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--min-samples", type=int, default=5,
        help="每个路由至少需要多少条低分记录才触发优化",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="回溯天数",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印报告，不调用 Meta-LLM，不写入数据库",
    )
    args = parser.parse_args()

    try:
        result = run_optimizer(
            min_samples=args.min_samples,
            days=args.days,
            dry_run=args.dry_run,
        )
    except Exception:
        logging.exception("Prompt optimizer 执行异常")
        sys.exit(1)
    # 按约定：0 = 正常完成（有低分记录），1 = 无低分记录或执行异常
    sys.exit(1 if result["total_low_score"] == 0 else 0)
