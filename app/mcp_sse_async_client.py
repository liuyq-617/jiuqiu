"""
MCP-over-SSE 异步客户端（使用官方 SDK）
支持通过 SSE 传输层与 MCP 服务器通信
"""
import json
import logging
from typing import List, Dict, Any, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger("crm_mcp_sse")


class MCPSSEAsyncClient:
    """MCP SSE 协议异步客户端 - 使用官方 SDK"""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        初始化 MCP SSE 客户端

        Args:
            base_url: MCP 服务器地址，例如 http://192.168.127.98:8001
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.sse_context = None
        self.session_context = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()

    async def _initialize(self):
        """异步初始化会话"""
        try:
            logger.info(f"[MCP SSE] 正在连接到 {self.base_url}/sse")

            # 建立 SSE 连接
            self.sse_context = sse_client(f"{self.base_url}/sse")
            read_stream, write_stream = await self.sse_context.__aenter__()

            # 创建客户端会话
            self.session_context = ClientSession(read_stream, write_stream)
            self.session = await self.session_context.__aenter__()

            # 初始化会话
            await self.session.initialize()
            logger.info(f"[MCP SSE] 会话初始化成功")

        except Exception as e:
            logger.error(f"[MCP SSE] 初始化失败: {e}")
            raise

    async def discover_resources(self) -> List[Dict[str, Any]]:
        """
        发现 MCP 服务器提供的资源列表

        Returns:
            资源列表
        """
        try:
            result = await self.session.list_resources()
            resources = [
                {
                    "uri": r.uri,
                    "name": r.name,
                    "description": r.description,
                    "mimeType": r.mimeType
                }
                for r in result.resources
            ]
            logger.info(f"[MCP SSE] 发现 {len(resources)} 个资源")
            return resources
        except Exception as e:
            logger.error(f"[MCP SSE] 资源发现失败: {e}")
            return []

    async def read_resource(
        self,
        uri: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        读取指定资源的数据

        Args:
            uri: 资源 URI
            filters: 过滤条件

        Returns:
            资源数据
        """
        try:
            result = await self.session.read_resource(uri)
            contents = [
                {
                    "uri": c.uri,
                    "mimeType": c.mimeType,
                    "text": c.text if hasattr(c, 'text') else None,
                    "blob": c.blob if hasattr(c, 'blob') else None
                }
                for c in result.contents
            ]

            logger.info(f"[MCP SSE] 读取资源 {uri}，获取 {len(contents)} 条数据")
            return {"contents": contents}
        except Exception as e:
            logger.error(f"[MCP SSE] 读取资源失败 {uri}: {e}")
            return {"contents": []}

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出服务器提供的工具

        Returns:
            工具列表
        """
        try:
            result = await self.session.list_tools()
            tools = []
            for t in result.tools:
                # 处理 inputSchema - 可能是 dict 或 pydantic model
                if hasattr(t.inputSchema, 'model_dump'):
                    schema = t.inputSchema.model_dump()
                elif isinstance(t.inputSchema, dict):
                    schema = t.inputSchema
                else:
                    schema = {}

                tools.append({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": schema
                })

            logger.info(f"[MCP SSE] 发现 {len(tools)} 个工具")
            return tools
        except Exception as e:
            logger.error(f"[MCP SSE] 工具列表获取失败: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        调用服务器工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        try:
            result = await self.session.call_tool(tool_name, arguments or {})

            # 提取内容
            data = []
            for content in result.content:
                if hasattr(content, 'text'):
                    data.append({"type": "text", "text": content.text})
                elif hasattr(content, 'data'):
                    data.append({"type": "resource", "data": content.data})

            logger.info(f"[MCP SSE] 调用工具 {tool_name} 成功")
            return {"content": data, "isError": result.isError if hasattr(result, 'isError') else False}
        except Exception as e:
            logger.error(f"[MCP SSE] 工具调用失败 {tool_name}: {e}")
            raise

    def suggest_chunking(
        self,
        sample_data: List[Dict[str, Any]],
        resource_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        基于样本数据建议分块策略

        Args:
            sample_data: 样本数据
            resource_schema: 资源 Schema（可选）

        Returns:
            分块建议
        """
        if not sample_data:
            return {
                "strategy": "no_split",
                "chunk_size": 1500,
                "metadata_fields": []
            }

        # 计算平均文本长度
        text_lengths = []
        for item in sample_data[:10]:
            content = ""
            if isinstance(item, dict):
                content = item.get("text", item.get("content", str(item)))
            else:
                content = str(item)
            text_lengths.append(len(content))

        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

        # 识别元数据字段
        metadata_fields = []
        common_meta_fields = ["date", "created_at", "updated_at", "company", "customer", "owner", "user", "author", "timestamp"]

        if sample_data and isinstance(sample_data[0], dict):
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

    async def close(self):
        """关闭客户端连接"""
        try:
            if self.session_context:
                await self.session_context.__aexit__(None, None, None)
            if self.sse_context:
                await self.sse_context.__aexit__(None, None, None)
            logger.debug(f"[MCP SSE] 客户端已关闭")
        except Exception as e:
            logger.error(f"[MCP SSE] 关闭连接失败: {e}")
