#!/usr/bin/env python3
"""
基准测试集 — 回答质量回归测试
===============================
维护一组标准 Q&A 问题，每次执行时：
  1. 调用 /api/chat 接口获取回答
  2. 调用 LLM-as-Judge 自动评分（相关性 / 完整性 / 准确性）
  3. 汇总打印测试报告，并将结果保存到 data/benchmark_results.jsonl

用法：
  cd /root/crm/crm_kb
  python3 scripts/benchmark.py                  # 测试所有用例
  python3 scripts/benchmark.py --case 3         # 只测试第 3 条
  python3 scripts/benchmark.py --url http://localhost:8000  # 指定服务地址
  python3 scripts/benchmark.py --no-judge       # 跳过 LLM 评分（仅测试接口可用性）

退出码：
  0 = 全部通过（均分 >= 3.0），1 = 存在低分项
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# ========== 基准测试集 ==========
# 格式：{"question": str, "expected_keywords": [...], "category": str}
# expected_keywords 用于简单关键词命中检验（至少命中 1 个则 PASS）
BENCHMARK_CASES = [
    # ── 聚合类 ──────────────────────────────────────────
    {
        "category": "聚合",
        "question": "知识库里共有多少家客户？",
        "expected_keywords": ["客户", "家", "个", "公司"],
    },
    {
        "category": "聚合",
        "question": "系统里有多少位负责人？",
        "expected_keywords": ["负责人", "位", "名", "人"],
    },
    {
        "category": "聚合",
        "question": "所有负责人的名字是什么？",
        "expected_keywords": ["负责人", "姓名", "名单"],
    },
    # ── 元数据过滤类 ────────────────────────────────────
    {
        "category": "元数据过滤",
        "question": "蒯歆越上个月跟进了哪些客户？",
        "expected_keywords": ["蒯歆越", "客户", "跟进", "月"],
    },
    {
        "category": "元数据过滤",
        "question": "2月份有哪些销售活动？",
        "expected_keywords": ["2月", "活动", "跟进", "拜访"],
    },
    {
        "category": "元数据过滤",
        "question": "本月的活动记录有哪些？",
        "expected_keywords": ["活动", "记录", "跟进"],
    },
    # ── 语义检索类 ──────────────────────────────────────
    {
        "category": "语义检索",
        "question": "最近有哪些正在等待回款的客户？",
        "expected_keywords": ["回款", "客户", "等待"],
    },
    {
        "category": "语义检索",
        "question": "目前有哪些正在推进方案的机会？",
        "expected_keywords": ["方案", "推进", "机会", "商机"],
    },
    {
        "category": "语义检索",
        "question": "哪些客户最近有需求变化或重新接触？",
        "expected_keywords": ["需求", "客户", "接触"],
    },
    # ── 绩效评分类 ──────────────────────────────────────
    {
        "category": "绩效评分",
        "question": "请评估魏明慧的工作情况，如果100分满分，请给出对他工作评估的分数",
        # 期望：给出具体数字分数 + 分析依据，不得纯拒绝
        "expected_keywords": ["分", "跟进", "魏明慧", "客户"],
    },
    {
        "category": "绩效评分",
        "question": "蒯歆越最近的工作表现如何？请打分（满分100分）",
        "expected_keywords": ["分", "蒯歆越", "跟进", "客户"],
    },
    {
        "category": "绩效评分",
        "question": "查看一下郭浩最近的表现，然后用夯、人上人、NPC、拉，进行评价",
        # 期望：识别用户档位体系，给出「夯/人上人/NPC/拉」之一的结论 + 依据
        "expected_keywords": ["郭浩", "夯", "人上人", "NPC", "拉"],
    },
    # ── 边界/鲁棒性类 ────────────────────────────────────
    {
        "category": "鲁棒性",
        "question": "张三李四王五最近怎么样",     # 不存在的人名
        "expected_keywords": ["未找到", "没有", "不存在", "知识库"],
    },
    {
        "category": "鲁棒性",
        "question": "今天天气怎么样？",            # 与 CRM 无关
        "expected_keywords": ["知识库", "无法", "不", "抱歉"],
    },
]  # END_BENCHMARK_CASES — 此注释为 API 插入锚点，请勿删除或移动


# ========== LLM-as-Judge（直接调用，无需启动服务）==========
_JUDGE_SYSTEM = """你是一个严格的 RAG 系统质量评估专家。
请根据用户问题和 AI 回答，打分（1~5 分，支持小数）并给出简短评语。
返回 JSON，不要加代码块标记：
{"relevance": 4.5, "completeness": 3.0, "accuracy": 4.0, "comment": "..."}
"""

_COLOR_GREEN  = "\033[92m"
_COLOR_YELLOW = "\033[93m"
_COLOR_RED    = "\033[91m"
_COLOR_RESET  = "\033[0m"
_COLOR_BOLD   = "\033[1m"


def _color(text: str, score: float) -> str:
    if score >= 4.0:
        return f"{_COLOR_GREEN}{text}{_COLOR_RESET}"
    if score >= 3.0:
        return f"{_COLOR_YELLOW}{text}{_COLOR_RESET}"
    return f"{_COLOR_RED}{text}{_COLOR_RESET}"


def judge_answer(question: str, answer: str, base_url: str) -> dict:
    """使用服务端 /api/chat 间接触发评分（省去重复 LLM 配置）——
    实际上直接用 httpx 调用同一个 LLM 端点。
    """
    # 读取服务的 LLM 配置
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.config import CHAT_API_KEY, CHAT_BASE_URL, OPENAI_CHAT_MODEL, CHAT_API_MODE

    api_base = CHAT_BASE_URL.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"

    if CHAT_API_MODE == "responses":
        url = api_base + "/responses"
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "instructions": _JUDGE_SYSTEM,
            "input": [{"role": "user", "content": f"问题：{question}\n\n回答：{answer}"}],
            "temperature": 0.1,
            "max_output_tokens": 256,
            "stream": False,
        }
    else:
        url = api_base + "/chat/completions"
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": f"问题：{question}\n\n回答：{answer}"},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
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
        for k in ("relevance", "completeness", "accuracy"):
            result[k] = max(1.0, min(5.0, float(result[k])))
        return result
    except Exception as e:
        return {"relevance": None, "completeness": None, "accuracy": None, "comment": f"评分失败: {e}"}


def run_benchmark(base_url: str = "http://localhost:8000",
                  case_index: int = None,
                  skip_judge: bool = False) -> int:
    cases = BENCHMARK_CASES
    if case_index is not None:
        if case_index < 1 or case_index > len(cases):
            print(f"❌ case 编号超出范围（1~{len(cases)}）")
            return 1
        cases = [cases[case_index - 1]]

    results = []
    fail_count = 0

    print(f"\n{_COLOR_BOLD}CRM 知识库问答 — 基准测试{_COLOR_RESET}")
    print(f"服务地址: {base_url}  用例数: {len(cases)}  LLM评分: {'关闭' if skip_judge else '开启'}")
    print("=" * 70)

    for idx, case in enumerate(cases, 1):
        q = case["question"]
        keywords = case["expected_keywords"]
        category = case["category"]

        print(f"\n[{idx}/{len(cases)}] [{category}] {q}")
        t0 = time.time()
        try:
            resp = httpx.post(
                f"{base_url}/api/chat",
                json={"question": q},
                timeout=60,
            )
            elapsed = time.time() - t0
            data = resp.json()
        except Exception as e:
            print(f"  ❌ 接口调用失败: {e}")
            results.append({"question": q, "category": category, "error": str(e)})
            fail_count += 1
            continue

        if not data.get("success"):
            print(f"  ❌ 接口返回失败: {data.get('detail')}")
            results.append({"question": q, "category": category, "error": data.get("detail")})
            fail_count += 1
            continue

        answer = data["data"].get("answer", "")
        answer_id = data["data"].get("answer_id", "")
        answer_preview = answer[:100].replace("\n", " ")

        # 关键词命中检验
        keyword_hit = any(k in answer for k in keywords)
        kw_flag = "✅" if keyword_hit else "⚠️ "

        print(f"  {kw_flag} 响应: {elapsed:.1f}s  |  answer_id: {answer_id or '—'}")
        print(f"  回答预览: {answer_preview}{'…' if len(answer) > 100 else ''}")

        scores = {}
        if not skip_judge and answer:
            print("  ⏳ LLM 评分中…", end="", flush=True)
            scores = judge_answer(q, answer, base_url)
            rel  = scores.get("relevance")
            comp = scores.get("completeness")
            acc  = scores.get("accuracy")
            avg_score = (rel + comp + acc) / 3 if all(v is not None for v in [rel, comp, acc]) else None
            score_str = (
                f"rel={_color(f'{rel:.1f}', rel)}  "
                f"comp={_color(f'{comp:.1f}', comp)}  "
                f"acc={_color(f'{acc:.1f}', acc)}  "
                f"avg={_color(f'{avg_score:.2f}', avg_score)}"
            ) if avg_score is not None else "评分失败"
            print(f"\r  {'✅' if avg_score and avg_score >= 3.0 else '⚠️ '} {score_str}")
            print(f"  评语: {scores.get('comment', '')}")
            if avg_score is not None and avg_score < 3.0:
                fail_count += 1
        elif not keyword_hit:
            fail_count += 1

        results.append({
            "question": q,
            "category": category,
            "answer_preview": answer_preview,
            "answer_id": answer_id,
            "response_ms": int(elapsed * 1000),
            "keyword_hit": keyword_hit,
            **scores,
        })

    # ── 汇总报告 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{_COLOR_BOLD}汇总报告{_COLOR_RESET}  总计: {len(results)}  ", end="")

    if not skip_judge:
        scored = [r for r in results if r.get("relevance") is not None]
        if scored:
            avgs = {k: sum(r[k] for r in scored) / len(scored)
                    for k in ("relevance", "completeness", "accuracy")}
            overall = sum(avgs.values()) / 3
            print(
                f"平均分: {_color(f'{overall:.2f}', overall)}  "
                f"(rel={avgs['relevance']:.2f} / comp={avgs['completeness']:.2f} / acc={avgs['accuracy']:.2f})"
            )
        else:
            print()
    else:
        print()

    kw_pass = sum(1 for r in results if r.get("keyword_hit"))
    print(f"关键词命中: {kw_pass}/{len(results)}  失败: {fail_count}")

    # ── 保存 JSONL ────────────────────────────────────
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "benchmark_results.jsonl"
    run_meta = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": base_url,
        "total": len(results),
        "fail_count": fail_count,
        "results": results,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_meta, ensure_ascii=False) + "\n")
    print(f"结果已保存: {out_path}")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRM 知识库问答基准测试")
    parser.add_argument("--url", default="http://localhost:8000", help="服务地址")
    parser.add_argument("--case", type=int, default=None, help="只运行第 N 条用例（1-based）")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 评分")
    args = parser.parse_args()

    exit_code = run_benchmark(
        base_url=args.url,
        case_index=args.case,
        skip_judge=args.no_judge,
    )
    sys.exit(exit_code)
