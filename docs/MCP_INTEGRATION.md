# MCP 数据源集成指南

## 概述

本系统支持通过 MCP (Model Context Protocol) 接口从外部数据源动态接入数据到知识库。

## 支持的数据源类型

### 1. 文件系统 (filesystem)
从本地或网络文件系统加载文档。

**配置示例：**
```json
{
  "name": "local_docs",
  "type": "filesystem",
  "enabled": true,
  "path": "/path/to/documents",
  "pattern": "**/*.md",
  "description": "本地文件系统数据源"
}
```

**参数说明：**
- `path`: 文件目录路径
- `pattern`: 文件匹配模式（支持 glob 语法）

### 2. HTTP API (http)
从 REST API 获取数据。

**配置示例：**
```json
{
  "name": "crm_api",
  "type": "http",
  "enabled": true,
  "url": "https://api.example.com/crm/activities",
  "method": "GET",
  "headers": {
    "Authorization": "Bearer YOUR_TOKEN_HERE",
    "Content-Type": "application/json"
  },
  "params": {
    "limit": 100,
    "status": "active"
  },
  "content_path": "data",
  "text_field": "content",
  "description": "CRM API 数据源"
}
```

**参数说明：**
- `url`: API 端点地址
- `method`: HTTP 方法（GET/POST）
- `headers`: 请求头（如认证 token）
- `params`: 请求参数（GET 为 query params，POST 为 JSON body）
- `content_path`: 数据在响应中的路径（如 "data.items"）
- `text_field`: 文本内容字段名

**API 响应格式示例：**
```json
{
  "data": [
    {
      "id": 1,
      "content": "客户跟进记录内容...",
      "created_at": "2026-03-10",
      "customer_name": "上海朋熙半导体",
      "sales_person": "张三"
    }
  ]
}
```

**元数据映射配置：**

如果 API 返回的字段名与系统标准字段（date, company, owner）不同，可以配置 `metadata_mapping` 进行映射：

```json
{
  "name": "crm_api",
  "type": "http",
  "enabled": true,
  "url": "https://api.example.com/crm/activities",
  "method": "GET",
  "headers": {
    "Authorization": "Bearer YOUR_TOKEN"
  },
  "content_path": "data",
  "text_field": "content",
  "metadata_mapping": {
    "date": "created_at",
    "company": "customer_name",
    "owner": "sales_person"
  }
}
```

这样系统会自动将 `created_at` 映射为 `date`，`customer_name` 映射为 `company`，`sales_person` 映射为 `owner`。

### 3. 数据库 (database)
从 MySQL 或 PostgreSQL 数据库查询数据。

**MySQL 配置示例：**
```json
{
  "name": "crm_mysql",
  "type": "database",
  "enabled": true,
  "db_type": "mysql",
  "host": "localhost",
  "port": 3306,
  "user": "root",
  "password": "password",
  "database": "crm",
  "query": "SELECT * FROM activities WHERE created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)",
  "text_field": "content",
  "description": "CRM MySQL 数据库"
}
```

**PostgreSQL 配置示例：**
```json
{
  "name": "crm_postgres",
  "type": "database",
  "enabled": true,
  "db_type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "user": "postgres",
  "password": "password",
  "database": "crm",
  "query": "SELECT * FROM activities WHERE created_at > NOW() - INTERVAL '30 days'",
  "text_field": "content",
  "description": "CRM PostgreSQL 数据库"
}
```

**参数说明：**
- `db_type`: 数据库类型（mysql/postgresql）
- `host`: 数据库主机
- `port`: 数据库端口
- `user`: 数据库用户名
- `password`: 数据库密码
- `database`: 数据库名
- `query`: SQL 查询语句
- `text_field`: 文本内容字段名

## 配置文件

配置文件位于项目根目录：`mcp_config.json`

**完整配置示例：**
```json
{
  "mcp_sources": [
    {
      "name": "local_docs",
      "type": "filesystem",
      "enabled": false,
      "path": "/path/to/documents",
      "pattern": "**/*.md"
    },
    {
      "name": "crm_api",
      "type": "http",
      "enabled": true,
      "url": "https://api.example.com/crm/activities",
      "method": "GET",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      },
      "content_path": "data",
      "text_field": "content"
    }
  ]
}
```

## 使用方法

### 1. 配置数据源

编辑 `mcp_config.json`，添加或修改数据源配置：
- 设置 `enabled: true` 启用数据源
- 填写必要的连接信息和认证凭据

### 2. 安装依赖

根据使用的数据源类型安装相应依赖：

```bash
# HTTP 数据源（已包含在 requirements.txt）
pip install httpx

# MySQL 数据源
pip install pymysql

# PostgreSQL 数据源
pip install psycopg2-binary
```

### 3. 重建索引

通过 API 触发知识库重建，系统会自动从所有启用的 MCP 数据源加载数据：

```bash
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"confirm": true}'
```

或通过前端界面点击"重建索引"按钮。

### 4. 验证数据

重建完成后，可以通过以下方式验证：

```bash
# 查看知识库统计
curl http://localhost:8000/api/stats

# 测试查询
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "最近的客户跟进情况"}'
```

## 数据格式要求

MCP 数据源返回的文档需要包含以下字段：

```python
{
    "content": "文档文本内容（必需）",
    "source": "数据源标识（必需）",
    "metadata": {  # 可选
        "date": "2026-03-10",
        "company": "客户公司名",
        "owner": "负责人"
    }
}
```

系统会自动对内容进行分块和向量化处理。

## 安全建议

1. **敏感信息保护**
   - 不要在配置文件中硬编码密码和 token
   - 使用环境变量存储敏感信息
   - 将 `mcp_config.json` 添加到 `.gitignore`

2. **访问控制**
   - 数据库账号使用只读权限
   - API token 设置适当的权限范围
   - 定期轮换认证凭据

3. **数据验证**
   - 验证从外部数据源获取的数据格式
   - 过滤敏感或不适当的内容
   - 设置数据量限制，避免过载

## 故障排查

### 问题：MCP 数据源未加载

**检查步骤：**
1. 确认 `mcp_config.json` 文件存在且格式正确
2. 检查数据源的 `enabled` 字段是否为 `true`
3. 查看应用日志中的 `[MCP]` 相关信息

### 问题：数据库连接失败

**检查步骤：**
1. 确认数据库服务正在运行
2. 验证连接信息（host、port、user、password）
3. 检查防火墙和网络连接
4. 确认已安装相应的数据库驱动（pymysql/psycopg2）

### 问题：HTTP API 请求失败

**检查步骤：**
1. 验证 API URL 是否正确
2. 检查认证 token 是否有效
3. 确认 API 返回的数据格式符合预期
4. 查看应用日志中的详细错误信息

## 扩展开发

### 自定义数据源

可以通过继承 `MCPDataSource` 基类实现自定义数据源：

```python
from app.mcp_loader import MCPDataSource, MCP_SOURCE_TYPES

class CustomMCPSource(MCPDataSource):
    def fetch_data(self) -> List[Dict[str, Any]]:
        # 实现自定义数据获取逻辑
        docs = []
        # ... 获取数据 ...
        return docs

# 注册自定义数据源
MCP_SOURCE_TYPES["custom"] = CustomMCPSource
```

然后在 `mcp_config.json` 中配置：

```json
{
  "name": "my_custom_source",
  "type": "custom",
  "enabled": true,
  "custom_param": "value"
}
```

## 性能优化

1. **增量更新**：对于大型数据源，考虑实现增量更新机制，只获取新增或修改的数据
2. **并行加载**：多个数据源可以并行加载以提高速度
3. **缓存策略**：对于更新频率低的数据源，可以实现本地缓存
4. **分页查询**：对于大量数据，使用分页查询避免内存溢出

## 相关文件

- `app/mcp_loader.py` - MCP 数据加载器实现
- `app/document_loader.py` - 文档加载和分块逻辑
- `mcp_config.json` - MCP 数据源配置文件
