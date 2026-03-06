#!/usr/bin/env python3
"""
分块预览脚本 - 将文档分块结果导出为可读文件，方便检查分块效果
用法：
  python scripts/preview_chunks.py              # 输出到 data/chunks_preview.txt
  python scripts/preview_chunks.py --json       # 同时输出 JSON 格式
  python scripts/preview_chunks.py --limit 20   # 只显示前 N 个片段
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.document_loader import load_and_split
from app.config import BASE_DIR, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="CRM 分块预览工具")
    parser.add_argument("--json", action="store_true", help="同时导出 JSON 文件")
    parser.add_argument("--limit", type=int, default=0, help="只预览前 N 个片段（0=全部）")
    args = parser.parse_args()

    # 加载并分块（自动查找文档位置）
    data_path = DATA_DIR if any(DATA_DIR.glob("**/*.md")) else BASE_DIR.parent
    print(f"[文档目录] {data_path}\n")
    chunks = load_and_split(data_path)

    if not chunks:
        print("[错误] 未找到任何文档片段")
        sys.exit(1)

    display_chunks = chunks[:args.limit] if args.limit > 0 else chunks

    # ===== 统计信息 =====
    lengths = [len(c["text"]) for c in chunks]
    print(f"\n{'='*60}")
    print(f"  分块统计")
    print(f"{'='*60}")
    print(f"  总片段数  : {len(chunks)}")
    print(f"  最短片段  : {min(lengths)} 字符")
    print(f"  最长片段  : {max(lengths)} 字符")
    print(f"  平均长度  : {sum(lengths)//len(lengths)} 字符")
    over_1000 = sum(1 for l in lengths if l > 1000)
    print(f"  >1000字符 : {over_1000} 个（可能包含多条记录，建议检查）")
    print(f"{'='*60}\n")

    # ===== 导出 TXT =====
    out_txt = BASE_DIR / "data" / "chunks_preview.txt"
    out_txt.parent.mkdir(exist_ok=True)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"CRM 知识库分块预览\n")
        f.write(f"总片段数: {len(chunks)}  |  预览数: {len(display_chunks)}\n")
        f.write("=" * 60 + "\n\n")

        for i, chunk in enumerate(display_chunks, 1):
            f.write(f"【片段 {i:04d}】")
            f.write(f"  来源: {chunk['source']}")
            f.write(f"  | chunk_id: {chunk['chunk_id']}")
            f.write(f"  | 类型: {chunk['type']}")
            f.write(f"  | 长度: {len(chunk['text'])} 字符\n")
            f.write("-" * 60 + "\n")
            f.write(chunk["text"])
            f.write("\n\n" + "=" * 60 + "\n\n")

    print(f"[✓] TXT 预览已保存：{out_txt}")

    # ===== 导出 JSON（可选）=====
    if args.json:
        out_json = BASE_DIR / "data" / "chunks_preview.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                [{"index": i, **c} for i, c in enumerate(display_chunks, 1)],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[✓] JSON 预览已保存：{out_json}")

    # ===== 控制台显示前 5 个片段 =====
    print(f"\n{'='*60}")
    print(f"  前 {min(5, len(display_chunks))} 个片段预览（完整内容见上方文件）")
    print(f"{'='*60}")
    for i, chunk in enumerate(display_chunks[:5], 1):
        print(f"\n▶ 片段 {i}  [{chunk['source']} | chunk_id={chunk['chunk_id']} | {len(chunk['text'])}字符]")
        print("-" * 50)
        # 控制台只显示前 200 字符
        preview = chunk["text"][:200]
        if len(chunk["text"]) > 200:
            preview += " …"
        print(preview)


if __name__ == "__main__":
    main()
