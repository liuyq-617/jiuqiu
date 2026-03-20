"""
MCP (Model Context Protocol) 数据加载器
支持通过 MCP 协议从外部数据源接入数据到知识库
"""
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("crm_mcp")


class MCPDataSource:
    """MCP 数据源基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    def fetch_data(self) -> List[Dict[str, Any]]:
        """
        从数据源获取数据，返回标准化的文档列表
        每个文档包含：
        - content: 文本内容
        - source: 数据源标识
        - metadata: 元数据（可选）
        """
        raise NotImplementedError("子类必须实现 fetch_data 方法")


class FileSystemMCPSource(MCPDataSource):
    """文件系统 MCP 数据源（示例实现）"""

    def fetch_data(self) -> List[Dict[str, Any]]:
        """从指定目录加载文件"""
        path = Path(self.config.get("path", ""))
        if not path.exists():
            logger.warning(f"[MCP:{self.name}] 路径不存在: {path}")
            return []

        docs = []
        pattern = self.config.get("pattern", "**/*.md")

        for file_path in path.glob(pattern):
            try:
                content = file_path.read_text(encoding="utf-8")
                docs.append({
                    "content": content,
                    "source": f"mcp:{self.name}:{file_path.name}",
                    "metadata": {
                        "path": str(file_path),
                        "mcp_source": self.name,
                    }
                })
                logger.info(f"[MCP:{self.name}] 加载文件: {file_path.name}")
            except Exception as e:
                logger.error(f"[MCP:{self.name}] 读取文件失败 {file_path}: {e}")

        return docs


class MCPProtocolSource(MCPDataSource):
    """标准 MCP 协议数据源"""

    def fetch_data(self) -> List[Dict[str, Any]]:
        """从 MCP 服务器获取数据"""
        from app.mcp_client import fetch_mcp_data

        base_url = self.config.get("base_url", "")
        resource_uri = self.config.get("resource_uri", "")
        filters = self.config.get("filters", {})
        metadata_mapping = self.config.get("metadata_mapping", {})

        if not base_url or not resource_uri:
            logger.error(f"[MCP:{self.name}] 缺少 base_url 或 resource_uri")
            return []

        try:
            docs = fetch_mcp_data(base_url, resource_uri, filters, metadata_mapping)
            logger.info(f"[MCP:{self.name}] 从 MCP 服务器获取 {len(docs)} 条数据")
            return docs
        except Exception as e:
            logger.error(f"[MCP:{self.name}] MCP 协议获取数据失败: {e}")
            return []


class HTTPMCPSource(MCPDataSource):
    """HTTP API MCP 数据源"""

    def fetch_data(self) -> List[Dict[str, Any]]:
        """从 HTTP API 获取数据"""
        import httpx

        url = self.config.get("url", "")
        method = self.config.get("method", "GET").upper()
        headers = self.config.get("headers", {})
        params = self.config.get("params", {})

        try:
            with httpx.Client(timeout=30) as client:
                if method == "GET":
                    resp = client.get(url, headers=headers, params=params)
                elif method == "POST":
                    resp = client.post(url, headers=headers, json=params)
                else:
                    logger.error(f"[MCP:{self.name}] 不支持的 HTTP 方法: {method}")
                    return []

                resp.raise_for_status()
                data = resp.json()

                # 根据配置的数据路径提取内容
                content_path = self.config.get("content_path", "data")
                items = self._extract_by_path(data, content_path)

                docs = []
                for i, item in enumerate(items):
                    # 提取文本内容
                    text_field = self.config.get("text_field", "content")
                    content = item.get(text_field, str(item))

                    # 提取元数据字段（如果配置了映射）
                    metadata = {}
                    metadata_mapping = self.config.get("metadata_mapping", {})
                    if metadata_mapping:
                        # 支持字段映射，例如 {"date": "created_at", "company": "customer_name"}
                        for target_field, source_field in metadata_mapping.items():
                            if source_field in item:
                                metadata[target_field] = item[source_field]
                    else:
                        # 默认尝试提取常见字段
                        for field in ["date", "company", "owner", "created_at", "customer", "user"]:
                            if field in item:
                                metadata[field] = item[field]

                    docs.append({
                        "content": content,
                        "source": f"mcp:{self.name}:item_{i}",
                        "metadata": metadata,
                    })

                logger.info(f"[MCP:{self.name}] 从 API 获取 {len(docs)} 条数据")
                return docs

        except Exception as e:
            logger.error(f"[MCP:{self.name}] HTTP 请求失败: {e}")
            return []

    def _extract_by_path(self, data: Any, path: str) -> List:
        """根据路径提取数据（支持 data.items 这样的路径）"""
        if not path:
            return [data] if not isinstance(data, list) else data

        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, [])
            else:
                return []

        return current if isinstance(current, list) else [current]


class DatabaseMCPSource(MCPDataSource):
    """数据库 MCP 数据源（支持 MySQL, PostgreSQL 等）"""

    def fetch_data(self) -> List[Dict[str, Any]]:
        """从数据库查询数据"""
        db_type = self.config.get("type", "mysql")
        query = self.config.get("query", "")

        if not query:
            logger.error(f"[MCP:{self.name}] 未配置查询语句")
            return []

        try:
            if db_type == "mysql":
                return self._fetch_from_mysql(query)
            elif db_type == "postgresql":
                return self._fetch_from_postgresql(query)
            else:
                logger.error(f"[MCP:{self.name}] 不支持的数据库类型: {db_type}")
                return []
        except Exception as e:
            logger.error(f"[MCP:{self.name}] 数据库查询失败: {e}")
            return []

    def _fetch_from_mysql(self, query: str) -> List[Dict[str, Any]]:
        """从 MySQL 查询数据"""
        try:
            import pymysql
        except ImportError:
            logger.error(f"[MCP:{self.name}] 请安装 pymysql: pip install pymysql")
            return []

        conn = pymysql.connect(
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 3306),
            user=self.config.get("user", ""),
            password=self.config.get("password", ""),
            database=self.config.get("database", ""),
        )

        docs = []
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

                text_field = self.config.get("text_field", "content")

                # 元数据字段映射
                metadata_mapping = self.config.get("metadata_mapping", {})

                for i, row in enumerate(rows):
                    content = row.get(text_field, str(row))

                    # 提取元数据
                    metadata = {}
                    if metadata_mapping:
                        for target_field, source_field in metadata_mapping.items():
                            if source_field in row:
                                metadata[target_field] = str(row[source_field])
                    else:
                        # 默认提取常见字段
                        for field in ["date", "company", "owner", "created_at", "customer", "user"]:
                            if field in row:
                                metadata[field] = str(row[field])

                    docs.append({
                        "content": content,
                        "source": f"mcp:{self.name}:row_{i}",
                        "metadata": metadata,
                    })

                logger.info(f"[MCP:{self.name}] 从 MySQL 查询到 {len(docs)} 条数据")
        finally:
            conn.close()

        return docs

    def _fetch_from_postgresql(self, query: str) -> List[Dict[str, Any]]:
        """从 PostgreSQL 查询数据"""
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            logger.error(f"[MCP:{self.name}] 请安装 psycopg2: pip install psycopg2-binary")
            return []

        conn = psycopg2.connect(
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 5432),
            user=self.config.get("user", ""),
            password=self.config.get("password", ""),
            database=self.config.get("database", ""),
        )

        docs = []
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

                text_field = self.config.get("text_field", "content")

                # 元数据字段映射
                metadata_mapping = self.config.get("metadata_mapping", {})

                for i, row in enumerate(rows):
                    content = row.get(text_field, str(dict(row)))

                    # 提取元数据
                    metadata = {}
                    if metadata_mapping:
                        for target_field, source_field in metadata_mapping.items():
                            if source_field in row:
                                metadata[target_field] = str(row[source_field])
                    else:
                        # 默认提取常见字段
                        for field in ["date", "company", "owner", "created_at", "customer", "user"]:
                            if field in row:
                                metadata[field] = str(row[field])

                    docs.append({
                        "content": content,
                        "source": f"mcp:{self.name}:row_{i}",
                        "metadata": metadata,
                    })

                logger.info(f"[MCP:{self.name}] 从 PostgreSQL 查询到 {len(docs)} 条数据")
        finally:
            conn.close()

        return docs


# MCP 数据源注册表
MCP_SOURCE_TYPES = {
    "filesystem": FileSystemMCPSource,
    "mcp": MCPProtocolSource,
    "http": HTTPMCPSource,
    "database": DatabaseMCPSource,
}


class MCPDataLoader:
    """MCP 数据加载器管理器"""

    def __init__(self, config_path: Optional[Path] = None):
        self.sources: List[MCPDataSource] = []
        if config_path and config_path.exists():
            self.load_config(config_path)

    def load_config(self, config_path: Path):
        """从配置文件加载 MCP 数据源"""
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            sources_config = config.get("mcp_sources", [])

            for src_config in sources_config:
                source_type = src_config.get("type", "")
                source_name = src_config.get("name", "")
                enabled = src_config.get("enabled", True)

                if not enabled:
                    logger.info(f"[MCP] 跳过已禁用的数据源: {source_name}")
                    continue

                source_class = MCP_SOURCE_TYPES.get(source_type)
                if not source_class:
                    logger.warning(f"[MCP] 未知的数据源类型: {source_type}")
                    continue

                source = source_class(source_name, src_config)
                self.sources.append(source)
                logger.info(f"[MCP] 注册数据源: {source_name} ({source_type})")

        except Exception as e:
            logger.error(f"[MCP] 加载配置文件失败: {e}")

    def add_source(self, source: MCPDataSource):
        """手动添加数据源"""
        self.sources.append(source)
        logger.info(f"[MCP] 添加数据源: {source.name}")

    def fetch_all(self) -> List[Dict[str, Any]]:
        """从所有数据源获取数据"""
        all_docs = []
        for source in self.sources:
            try:
                docs = source.fetch_data()
                all_docs.extend(docs)
                logger.info(f"[MCP] {source.name} 获取 {len(docs)} 条数据")
            except Exception as e:
                logger.error(f"[MCP] {source.name} 获取数据失败: {e}", exc_info=True)

        logger.info(f"[MCP] 总共获取 {len(all_docs)} 条数据")
        return all_docs