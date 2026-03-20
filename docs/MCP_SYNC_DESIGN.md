# MCP 数据同步系统设计

## 概述

通用的 MCP 数据同步系统，完全符合 MCP 协议规范，支持 Resources 和 Tools 两种数据获取模式。

## 核心特性

### 1. 协议支持
- **MCP Resources** - 静态数据资源（`resources/list`, `resources/read`）
- **MCP Tools** - 动态工具调用（`tools/list`, `tools/call`）
- **传输层** - SSE 和 HTTP REST 双模式自动切换

### 2. 同步模式

#### Resource 模式
```json
{
  "type": "resource",
  "server_url": "http://192.168.127.98:8001",
  "resource_uri": "crm://activities",
  "filters": {
    "date_from": "2026-01-01",
    "date_to": "2026-03-31"
  }
}
```

#### Tool 模式
```json
{
  "type": "tool",
  "server_url": "http://192.168.127.98:8001",
  "tool_name": "get_activity_records",
  "arguments": {
    "limit": 1000,
    "date_from": "2026-01-01"
  }
}
```

### 3. 同步策略

- **手动触发** - 用户点击按钮立即同步
- **定时同步** - Cron 表达式配置（每天/每周/每小时）
- **增量同步** - 基于时间戳只同步新数据
- **全量同步** - 重新获取所有数据

### 4. 数据处理流程

```
MCP Server → 数据获取 → 数据转换 → 文本分块 → 向量化 → 索引更新
```

## 系统架构

### 后端组件

1. **MCP 同步管理器** (`app/mcp_sync.py`)
   - 同步任务配置管理
   - 任务调度和执行
   - 同步历史记录

2. **MCP 通用客户端** (`app/mcp_sse_client.py`)
   - 支持 Resources 和 Tools
   - SSE/HTTP 自动切换
   - 连接池和重试机制

3. **数据转换器** (`app/mcp_transformer.py`)
   - 统一数据格式转换
   - 元数据提取
   - 文本清洗和分块

4. **任务调度器** (集成到 `app/main.py`)
   - APScheduler 定时任务
   - 任务状态监控
   - 错误处理和重试

### 前端界面

1. **同步任务配置页面** (`static/mcp-sync.html`)
   - 连接 MCP 服务器
   - 选择 Resource 或 Tool
   - 配置同步参数
   - 设置定时策略

2. **同步任务列表** (集成到 `static/mcp.html`)
   - 查看所有同步任务
   - 手动触发同步
   - 查看同步历史
   - 任务启用/禁用

3. **同步历史记录** (`static/mcp-history.html`)
   - 同步时间和状态
   - 数据量统计
   - 错误日志查看

## API 设计

### 同步任务管理

```
POST   /api/mcp/sync/tasks          创建同步任务
GET    /api/mcp/sync/tasks          获取任务列表
PUT    /api/mcp/sync/tasks/{id}     更新任务配置
DELETE /api/mcp/sync/tasks/{id}     删除任务
POST   /api/mcp/sync/tasks/{id}/run 手动触发同步
GET    /api/mcp/sync/tasks/{id}/history 查看同步历史
```

### 服务发现

```
POST   /api/mcp/discover/resources  发现 Resources
POST   /api/mcp/discover/tools      发现 Tools
POST   /api/mcp/test                测试连接和预览数据
```

## 配置文件格式

### sync_tasks.json

```json
{
  "tasks": [
    {
      "id": "task_001",
      "name": "CRM 活动记录同步",
      "enabled": true,
      "type": "tool",
      "server_url": "http://192.168.127.98:8001",
      "tool_name": "get_activity_records",
      "arguments": {
        "limit": 1000
      },
      "schedule": {
        "type": "cron",
        "expression": "0 2 * * *",
        "timezone": "Asia/Shanghai"
      },
      "data_mapping": {
        "text_field": "content",
        "metadata_fields": ["date", "owner", "company"]
      },
      "last_sync": "2026-03-13T10:30:00Z",
      "last_status": "success"
    }
  ]
}
```

## 实现优先级

### Phase 1: 核心功能（当前）
- [x] MCP SSE 客户端
- [x] 服务发现 API
- [ ] Tool 调用支持
- [ ] 同步任务配置界面

### Phase 2: 调度和自动化
- [ ] 定时任务调度器
- [ ] 增量同步逻辑
- [ ] 同步历史记录

### Phase 3: 监控和优化
- [ ] 同步状态监控
- [ ] 错误告警
- [ ] 性能优化

## 使用场景

### 场景 1: CRM 活动记录定时同步
```
用户配置 → 选择 Tool: get_activity_records → 设置每天凌晨 2 点同步 → 自动执行
```

### 场景 2: 外部知识库资源接入
```
用户配置 → 选择 Resource: kb://documents → 设置每周同步 → 自动执行
```

### 场景 3: 手动触发全量同步
```
用户点击"立即同步" → 调用 MCP Tool → 更新索引 → 显示结果
```

## 技术栈

- **后端**: FastAPI + APScheduler + httpx
- **前端**: 原生 JavaScript + Fetch API
- **存储**: JSON 文件配置 + SQLite 历史记录
- **协议**: MCP-over-SSE / HTTP REST
