"""
文档加载与分块模块
支持将 CRM markdown 文件按活动记录分割成语义完整的块
支持通过 MCP 协议从外部数据源加载数据
"""
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR, BASE_DIR

logger = logging.getLogger(__name__)


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


def load_and_split(data_dir: Path = DATA_DIR, enable_mcp: bool = True) -> List[Dict[str, Any]]:
    """
    加载所有文档并分块，返回所有片段列表

    Args:
        data_dir: 本地数据目录
        enable_mcp: 是否启用 MCP 数据源
    """
    # 1. 加载本地 markdown 文件
    docs = load_markdown_files(data_dir)

    # 2. 加载 MCP 数据源（如果启用）
    mcp_docs = []
    if enable_mcp:
        try:
            from app.mcp_loader import MCPDataLoader
            mcp_config_path = BASE_DIR / "mcp_config.json"
            if mcp_config_path.exists():
                print(f"\n[MCP] 加载配置: {mcp_config_path}")
                mcp_loader = MCPDataLoader(mcp_config_path)
                mcp_docs = mcp_loader.fetch_all()
                if mcp_docs:
                    print(f"[MCP] 从 MCP 数据源获取 {len(mcp_docs)} 个文档")
            else:
                print(f"[MCP] 配置文件不存在，跳过 MCP 数据源: {mcp_config_path}")
        except Exception as e:
            print(f"[MCP] 加载 MCP 数据源失败: {e}")

    # 3. 分块处理
    all_chunks = []

    # 处理本地 markdown 文件（使用 CRM 格式分块）
    for doc in docs:
        chunks = split_by_activity(doc["content"], doc["source"])
        all_chunks.extend(chunks)

    # 处理 MCP 数据源（使用通用分块策略）
    for doc in mcp_docs:
        chunks = split_mcp_document(doc)
        all_chunks.extend(chunks)

    print(f"\n[分块完成] 共 {len(docs) + len(mcp_docs)} 个文件，{len(all_chunks)} 个片段")
    return all_chunks


def split_mcp_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    对 MCP 数据源文档进行分块

    策略：
    1. 如果文档已包含元数据（date, company, owner），视为单个活动记录，不拆分
    2. 如果文档较短（<= CHUNK_SIZE），保持完整
    3. 如果文档较长，按段落拆分，保留重叠
    """
    content = doc.get("content", "")
    source = doc.get("source", "unknown")
    metadata = doc.get("metadata", {})

    # 提取元数据
    date = metadata.get("date", "")
    company = metadata.get("company", "")
    owner = metadata.get("owner", "")

    # 如果文档较短或已有完整元数据，不拆分
    if len(content) <= CHUNK_SIZE or (date and company and owner):
        return [{
            "text": content,
            "source": source,
            "chunk_id": 0,
            "type": "mcp_doc",
            "date": date,
            "company": company,
            "owner": owner,
        }]

    # 文档较长，按段落拆分
    chunks = []
    paragraphs = content.split('\n\n')
    current = []
    current_len = 0
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        # 如果当前段落加入后超过限制，先保存当前块
        if current_len + para_len > CHUNK_SIZE and current:
            chunk_text = '\n\n'.join(current)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_id": chunk_idx,
                "type": "mcp_doc_part",
                "date": date,
                "company": company,
                "owner": owner,
            })
            chunk_idx += 1

            # 保留重叠部分
            if CHUNK_OVERLAP > 0 and len(chunk_text) > CHUNK_OVERLAP:
                overlap_text = chunk_text[-CHUNK_OVERLAP:]
                current = [overlap_text, para]
                current_len = len(overlap_text) + para_len + 2
            else:
                current = [para]
                current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2  # +2 for '\n\n'

    # 保存最后一块
    if current:
        chunks.append({
            "text": '\n\n'.join(current),
            "source": source,
            "chunk_id": chunk_idx,
            "type": "mcp_doc_part",
            "date": date,
            "company": company,
            "owner": owner,
        })

    return chunks if chunks else [{
        "text": content,
        "source": source,
        "chunk_id": 0,
        "type": "mcp_doc",
        "date": date,
        "company": company,
        "owner": owner,
    }]


# ========== 摘要生成（Summary RAG） ==========

SUMMARY_PROMPT = """请将以下 CRM 活动记录总结为简洁的结构化摘要，包含：
- 客户名称
- 沟通日期
- 核心结论（1-2句话）
- 关键待办事项

只输出摘要，不要解释。

活动记录：
{text}"""


def generate_summary(text: str, client, model: str) -> str:
    """调用 LLM 为单条活动记录生成摘要"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(text=text)}],
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[摘要生成] 失败，使用原文前200字: {e}")
        return text[:200]


def generate_summaries(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    为 activity 类型的 chunks 批量生成摘要，返回 summary chunks 列表。
    摘要 chunk 与原始 chunk 共享 chunk_id/source/date/company/owner，chunk_type="summary"。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from app.config import (
        CHAT_API_KEY, CHAT_BASE_URL,
        SUMMARY_LLM_MODEL, SUMMARY_MAX_CONCURRENCY,
    )

    activity_chunks = [c for c in chunks if c.get("type") == "activity"]
    if not activity_chunks:
        logger.info("[摘要生成] 没有 activity 类型的 chunk，跳过")
        return []

    client = OpenAI(api_key=CHAT_API_KEY, base_url=CHAT_BASE_URL)
    summary_chunks = []
    total = len(activity_chunks)
    logger.info(f"[摘要生成] 开始为 {total} 条活动记录生成摘要 (model={SUMMARY_LLM_MODEL}, concurrency={SUMMARY_MAX_CONCURRENCY})")

    def _process(chunk):
        summary_text = generate_summary(chunk["text"], client, SUMMARY_LLM_MODEL)
        return {
            "text": summary_text,
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "type": "summary",
            "date": chunk.get("date", ""),
            "company": chunk.get("company", ""),
            "owner": chunk.get("owner", ""),
        }

    with ThreadPoolExecutor(max_workers=SUMMARY_MAX_CONCURRENCY) as pool:
        futures = {pool.submit(_process, c): i for i, c in enumerate(activity_chunks)}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                summary_chunks.append(future.result())
            except Exception as e:
                idx = futures[future]
                logger.error(f"[摘要生成] chunk {idx} 失败: {e}")
            if done % 50 == 0 or done == total:
                logger.info(f"[摘要生成] 进度 {done}/{total}")

    logger.info(f"[摘要生成] 完成，生成 {len(summary_chunks)} 条摘要")
    return summary_chunks


# ========== 文件解析（上传功能） ==========

def parse_uploaded_file(filename: str, content: bytes) -> str:
    """
    解析上传的文件内容，支持 .md, .txt, .pdf
    返回纯文本内容
    """
    ext = filename.lower().split('.')[-1]

    if ext in ('md', 'txt'):
        # Markdown 和文本文件直接解码
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            return content.decode('gbk', errors='ignore')

    elif ext == 'pdf':
        # PDF 文件解析
        try:
            from pypdf import PdfReader
            from io import BytesIO

            pdf_file = BytesIO(content)
            reader = PdfReader(pdf_file)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return '\n\n'.join(text_parts)
        except Exception as e:
            logger.error(f"[文件解析] PDF 解析失败 {filename}: {e}")
            raise ValueError(f"PDF 解析失败: {e}")

    else:
        raise ValueError(f"不支持的文件格式: .{ext}")


def process_uploaded_file(filename: str, content: bytes) -> List[Dict[str, Any]]:
    """
    处理上传的文件：解析 -> 分块 -> 返回 chunks
    """
    logger.info(f"[文件上传] 开始处理: {filename}")

    # 1. 解析文件内容
    text = parse_uploaded_file(filename, content)
    if not text.strip():
        raise ValueError("文件内容为空")

    logger.info(f"[文件上传] 解析完成，共 {len(text)} 字符")

    # 2. 判断文件类型并分块
    # 如果是 CRM 格式的 markdown，按活动记录分块
    if filename.lower().endswith('.md') and '---' in text and '###' in text:
        chunks = split_by_activity(text, filename)
        logger.info(f"[文件上传] CRM 格式分块，共 {len(chunks)} 个活动记录")
    else:
        # 通用文档，按段落分块
        chunks = split_mcp_document({
            'content': text,
            'source': filename,
            'metadata': {}
        })
        logger.info(f"[文件上传] 通用格式分块，共 {len(chunks)} 个片段")

    return chunks
