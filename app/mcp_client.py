"""
标准 MCP (Model Context Protocol) 客户端
支持服务发现、资源列表、数据拉取等标准 MCP 操作
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger("crm_mcp_client")


class MCPClient:
    """MCP 协议客户端"""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        初始化 MCP 客户端

        Args:
            base_url: MCP 服务器地址，例如 http://192.168.127.190:8000
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def discover_resources(self) -> List[Dict[str, Any]]:
        """
        发现 MCP 服务器提供的资源列表

        Returns:
            资源列表，每个资源包含：
            - uri: 资源标识符
            - name: 资源名称
            - description: 资源描述
            - mimeType: 数据类型
            - metadata: 元数据信息
        """
        try:
            # MCP 标准：GET /resources 获取资源列表
            resp = self.client.get(f"{self.base_url}/resources")
            resp.raise_for_status()
            data = resp.json()

            resources = data.get("resources", [])
            logger.info(f"[MCP] 发现 {len(resources)} 个资源")
            return resources
        except Exception as e:
            logger.error(f"[MCP] 资源发现失败: {e}")
            return []

    def read_resource(
        self,
        uri: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        读取指定资源的数据

        Args:
            uri: 资源 URI
            filters: 过滤条件，例如 {"date_from": "2026-01-01", "date_to": "2026-03-01"}

        Returns:
            资源数据，包含：
            - contents: 数据内容列表
            - metadata: 元数据
        """
        try:
            # MCP 标准：POST /resources/read
            payload = {"uri": uri}
            if filters:
                payload["filters"] = filters

            resp = self.client.post(
                f"{self.base_url}/resources/read",
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()

            logger.info(f"[MCP] 读取资源 {uri}，获取 {len(data.get('contents', []))} 条数据")
            return data
        except Exception as e:
            logger.error(f"[MCP] 读取资源失败 {uri}: {e}")
            return {"contents": [], "metadata": {}}

    def get_resource_schema(self, uri: str) -> Dict[str, Any]:
        """
        获取资源的数据结构（字段定义）

        Args:
            uri: 资源 URI

        Returns:
            Schema 定义，包含字段列表和类型信息
        """
        try:
            resp = self.client.get(
                f"{self.base_url}/resources/schema",
                params={"uri": uri}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[MCP] 获取 Schema 失败 {uri}: {e}")
            return {}

    def suggest_chunking(
        self,
        sample_data: List[Dict[str, Any]],
        resource_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        基于样本数据和 Schema，建议分块策略

        Args:
            sample_data: 样本数据
            resource_schema: 资源 Schema

        Returns:
            分块建议，包含：
            - strategy: 分块策略（"no_split", "by_size", "by_field"）
            - chunk_size: 建议的块大小
            - split_field: 如果按字段拆分，使用的字段名
            - metadata_fields: 建议提取的元数据字段
        """
        # 分析数据特征
        if not sample_data:
            return {
                "strategy": "no_split",
                "chunk_size": 1500,
                "metadata_fields": []
            }

        # 计算平均文本长度
        text_lengths = []
        for item in sample_data[:10]:  # 只分析前10条
            content = str(item.get("content", ""))
            text_lengths.append(len(content))

        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

        # 识别元数据字段
        metadata_fields = []
        common_meta_fields = ["date", "created_at", "updated_at", "company", "customer", "owner", "user", "author"]

        if sample_data:
            first_item = sample_data[0]
            for field in common_meta_fields:
                if field in first_item:
                    metadata_fields.append(field)

        # 决定分块策略
        if avg_length <= 1500:
            strategy = "no_split"
            chunk_size = 1500
        else:
            strategy = "by_size"
            chunk_size = 1500

        return {
            "strategy": strategy,
            "chunk_size": chunk_size,
            "metadata_fields": metadata_fields,
            "avg_content_length": int(avg_length),
            "sample_count": len(sample_data)
        }

    def close(self):
        """关闭客户端连接"""
        self.client.close()


def fetch_mcp_data(
    base_url: str,
    resource_uri: str,
    filters: Optional[Dict[str, Any]] = None,
    metadata_mapping: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    从 MCP 服务器获取数据的便捷函数

    Args:
        base_url: MCP 服务器地址
        resource_uri: 资源 URI
        filters: 过滤条件
        metadata_mapping: 元数据字段映射

    Returns:
        标准化的文档列表
    """
    client = MCPClient(base_url)

    try:
        # 读取资源数据
        result = client.read_resource(resource_uri, filters)
        contents = result.get("contents", [])

        docs = []
        for i, item in enumerate(contents):
            # 提取文本内容
            content = item.get("text", item.get("content", str(item)))

            # 提取元数据
            metadata = {}
            if metadata_mapping:
                for target_field, source_field in metadata_mapping.items():
                    if source_field in item:
                        metadata[target_field] = item[source_field]
            else:
                # 自动提取常见字段
                for field in ["date", "company", "owner", "created_at", "customer", "user"]:
                    if field in item:
                        metadata[field] = item[field]

            docs.append({
                "content": content,
                "source": f"mcp:{resource_uri}:item_{i}",
                "metadata": metadata
            })

        logger.info(f"[MCP] 从 {resource_uri} 获取 {len(docs)} 条数据")
        return docs

    finally:
        client.close()
