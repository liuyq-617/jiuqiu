#!/usr/bin/env python3
"""
一键构建知识库索引脚本
用法：python scripts/build_index.py
"""
import sys
from pathlib import Path

# 将 crm_kb 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.document_loader import load_and_split
from app.vector_store import build_index, disconnect_milvus
from app.config import DATA_DIR, BASE_DIR

def main():
    print("=" * 55)
    print("  CRM 知识库索引构建工具")
    print("=" * 55)

    # 优先从 data/ 目录加载，若为空则从上级 crm/ 目录加载
    data_path = DATA_DIR
    if not any(data_path.glob("**/*.md")):
        # 尝试从父目录找 md 文件
        parent_md = list(BASE_DIR.parent.glob("*.md"))
        if parent_md:
            print(f"\n[提示] data/ 目录为空，将使用上级目录的 markdown 文件：")
            for f in parent_md:
                print(f"  - {f.name}")
            data_path = BASE_DIR.parent
        else:
            print("\n[提示] 暂无 markdown 文件，跳过索引构建")
            print("  请将文档放入 crm_kb/data/ 目录，或访问 http://localhost:8000/upload 上传")
            print("  上传后执行 python3 scripts/build_index.py 重建索引")
            sys.exit(0)

    print(f"\n[步骤 1/3] 加载并分割文档...")
    chunks = load_and_split(data_path)
    if not chunks:
        print("[错误] 没有可索引的文档片段")
        sys.exit(1)

    print(f"\n[步骤 2/3] 向量化并写入 Milvus...")
    try:
        count = build_index(chunks)
    except ConnectionError as e:
        print(f"\n[错误] {e}")
        print("  请先启动 Milvus：docker compose up -d")
        sys.exit(1)

    print(f"\n[步骤 3/3] 完成！")
    print(f"  共写入 {count} 条向量记录")
    print("\n[下一步] 启动 Web 服务：")
    print("  cd crm_kb && uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("=" * 55)

    disconnect_milvus()


if __name__ == "__main__":
    main()
