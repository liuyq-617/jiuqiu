"""
文档加载与分块模块
支持将 CRM markdown 文件按活动记录分割成语义完整的块
"""
import re
from pathlib import Path
from typing import List, Dict, Any
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


def load_markdown_files(data_dir: Path = DATA_DIR) -> List[Dict[str, Any]]:
    """加载指定目录下所有 markdown 文件"""
    docs = []
    md_files = list(data_dir.glob("**/*.md"))
    if not md_files:
        # 向上一级查找，兼容直接放在 crm/ 目录的文件
        parent_dir = data_dir.parent.parent
        md_files = list(parent_dir.glob("*.md"))

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            docs.append({
                "source": str(file_path.name),
                "content": content,
                "path": str(file_path),
            })
            print(f"  [加载] {file_path.name} ({len(content)} 字符)")
        except Exception as e:
            print(f"  [警告] 无法读取 {file_path}: {e}")
    return docs


def extract_metadata(text: str) -> Dict[str, str]:
    """
    从活动记录文本中提取结构化元数据：
      - date   : 活动日期，如 2026-03-05
      - company: 客户公司名，如 上海朋熙半导体有限公司
      - owner  : 负责人姓名，如 蒯歆越（Xinyue Kuai）
    """
    meta = {"date": "", "company": "", "owner": ""}

    # 从标题行提取日期和公司：### 2026-03-05 15:38  |  公司名
    title_match = re.search(
        r'^###\s+(\d{4}-\d{2}-\d{2})[^|]*\|\s*(.+?)\s*$',
        text, re.MULTILINE
    )
    if title_match:
        meta["date"]    = title_match.group(1).strip()
        meta["company"] = title_match.group(2).strip()

    # 从表格行提取负责人：| 负责人 | 姓名 |
    owner_match = re.search(
        r'\|\s*负责人\s*\|\s*(.+?)\s*\|',
        text
    )
    if owner_match:
        meta["owner"] = owner_match.group(1).strip()

    return meta


def split_by_activity(content: str, source: str) -> List[Dict[str, Any]]:
    """
    按照 CRM 活动记录的 markdown 格式拆分：
    每个 '---' 横线作为分隔符，前面有 '### 日期 | 公司' 作为标题
    """
    chunks = []
    # 按 '---' 分割，保留活动块
    sections = re.split(r'\n---\n', content)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        meta = extract_metadata(section)

        # 进一步拆分超长块
        if len(section) <= CHUNK_SIZE:
            chunks.append({
                "text": section,
                "source": source,
                "chunk_id": i,
                "type": "activity",
                **meta,
            })
        else:
            # 超长块按段落继续拆分（元数据继承自原始块）
            sub_chunks = split_long_text(section, source, i)
            for sub in sub_chunks:
                sub.update(meta)
            chunks.extend(sub_chunks)

    return chunks


def split_long_text(text: str, source: str, base_idx: int) -> List[Dict[str, Any]]:
    """对超长文本按段落进行拆分，保留滑动窗口重叠"""
    chunks = []
    paragraphs = text.split('\n\n')
    current = []
    current_len = 0
    sub_idx = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > CHUNK_SIZE and current:
            chunk_text = '\n\n'.join(current)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_id": f"{base_idx}_{sub_idx}",
                "type": "activity_part",
            })
            sub_idx += 1
            # 保留重叠
            overlap_text = chunk_text[-CHUNK_OVERLAP:]
            current = [overlap_text]
            current_len = len(overlap_text)
        current.append(para)
        current_len += para_len + 2

    if current:
        chunks.append({
            "text": '\n\n'.join(current),
            "source": source,
            "chunk_id": f"{base_idx}_{sub_idx}",
            "type": "activity_part",
        })
    return chunks


def load_and_split(data_dir: Path = DATA_DIR) -> List[Dict[str, Any]]:
    """加载所有文档并分块，返回所有片段列表"""
    docs = load_markdown_files(data_dir)
    all_chunks = []
    for doc in docs:
        chunks = split_by_activity(doc["content"], doc["source"])
        all_chunks.extend(chunks)
    print(f"\n[分块完成] 共 {len(docs)} 个文件，{len(all_chunks)} 个片段")
    return all_chunks
