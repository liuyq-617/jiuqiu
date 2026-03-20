# CRM 知识库问答系统 — 待办事项

## 🚀 高优先级

### 1. 检索增强 (Advanced RAG)
**目标**: 提升召回率和排序质量

- [x] **查询改写 (Query Rewriting)**
  - 用 LLM 生成查询变体（同义词、不同表述）
  - 扩大召回范围，减少表述差异导致的漏检

- [x] **混合检索 (Hybrid Search)**
  - 向量检索 + BM25 关键词检索
  - Reciprocal Rank Fusion (RRF) 融合排名
  - 平衡语义理解和精确匹配

- [x] **LLM 重排序 (Reranking)**
  - 用 LLM 对候选结果打相关性分数 (1-10)
  - 提升排序质量，优先展示最有用的片段

- [x] **父文档展开 (Parent Doc Retrieval)**
  - 检索到子块时自动拼合完整父文档
  - 保留更丰富的上下文信息

- [x] **BM25 中文分词优化**（`app/advanced_rag.py` `_bm25_scores`）
  - ~~当前：按单字拆分~~，已升级为 `jieba` 词级分词
  - 停用词过滤（118词）：通用虚词 + 时间泛化词（最近/上周…）+ CRM高频词（客户/记录…）
  - TDengine 领域词典（62条）：超级表、子表、taosKeeper、taosAdapter、集群部署等专有术语，防止被拆碎
  - 服务启动时预热（`main.py` lifespan），避免首次检索卡顿
  - 降级兜底：jieba 未安装时自动回退为单字分词，不影响主流程
  - 依赖：`jieba>=0.42.1`（已加入 `requirements.txt`）

**实现文件**: `app/advanced_rag.py`（已有）、`app/rag.py`、`app/config.py`

---

### 2. 回答质量评价体系
**目标**: 持续量化和改进回答质量

- [ ] **用户反馈收集**
  - 前端添加"👍/👎"按钮
  - 将问答对 + 评价存入数据库
  - 支持用户补充文字反馈

- [x] **LLM-as-Judge 自动评分**
  - 用 GPT-4o 评估回答质量（相关性/完整性/准确性）
  - 每次问答结束后自动触发（`/api/chat` 和流式接口均已接入）
  - 用户提交反馈后幂等跳过（已评过则不重复计费）

- [ ] **基准测试集**
  - 维护标准 Q&A 测试集（50-100 条）
  - 每次修改后自动回归测试
  - 防止质量下降

- [ ] **指标看板**
  - 跟踪准确率、响应时间、用户满意度趋势
  - 可视化展示评价分布

**实现文件**: `app/feedback.py`（已有）、`scripts/benchmark.py`（已有）、`static/index.html`

---

### 3. MCP 数据源集成完善
**目标**: 支持更多外部数据源

- [ ] **数据源实现**
  - [x] 文件系统数据源 (FileSystemMCPSource)
  - [x] HTTP API 数据源 (HTTPMCPSource)
  - [ ] MySQL 数据源 (MySQLMCPSource)
  - [ ] PostgreSQL 数据源 (PostgreSQLMCPSource)
  - [ ] 飞书文档数据源 (FeishuDocsMCPSource)

- [ ] **配置管理**
  - [ ] Web UI 配置界面（无需手动编辑 JSON）
  - [ ] 数据源连接测试功能
  - [ ] 增量同步策略（定时/手动触发）

- [ ] **安全加固**
  - [ ] 敏感信息环境变量化
  - [ ] 数据源权限最小化（只读账号）
  - [ ] 凭据定期轮换提醒

**实现文件**: `app/mcp_loader.py`（新增）、`app/document_loader.py`、`mcp_config.json`

---

---

### 3.5 回答质量闭环优化（本次对话新增）
**目标**: 将 LLM-as-Judge 分数转化为可执行的 Prompt 改进行动

#### 3.5.1 Prompt 自动优化闭环
- [x] **低分样本分析脚本** `scripts/prompt_optimizer.py`
  - 从 `feedback.db` 提取近 7 天平均分 < 3.5 的样本，按路由类型分组（ranking / evaluation / aggregate / metadata_filter / semantic）
  - 调用 Meta-LLM 分析共性缺陷，生成候选 Prompt 改进建议
  - 写入 `prompt_candidates` 表（status=pending），待人工审核后上线

- [x] **`prompt_candidates` 数据表**（在 `feedback.db` 中新增）
  - 字段：`route` / `suggestion` / `sample_count` / `status` / `avg_score_before` / `avg_score_after`
  - status 流转：`pending` → `approved` / `rejected`

- [x] **`rag.py` 热加载已审核 Prompt**
  - `_load_active_prompt()`：优先读取 `prompt_candidates` 表中最新 approved 记录
  - 60s 缓存，无需重启服务；无审核记录时降级使用默认 `SYSTEM_PROMPT`

- [x] **每日定时触发**（`main.py` lifespan 后台线程）
  - 凌晨 03:00 自动运行 `run_optimizer(min_samples=5)`
  - API：`GET /api/prompt-candidates`、`POST /…/approve`、`POST /…/reject`

**实现文件**: `scripts/prompt_optimizer.py`（新增）、`app/rag.py`、`app/feedback.py`、`app/main.py`

---

#### 3.5.2 飞书机器人集成评价机制
**目标**: 飞书用户的每次问答也进入评价闭环，与 Web 端数据统一

- [ ] **`_process_question` 接入 feedback**（`feishu_bot.py`，5行代码，最高优先）
  - RAG 结束后调用 `save_qa()` 生成 `answer_id`
  - 同步触发 `trigger_judge(answer_id)` 异步 LLM 评分
  - 记录 `start_time` 计算 `response_ms`

- [ ] **卡片末尾添加 👍/👎/⭐ 交互按钮**
  - 新增 `_close_streaming_card_with_feedback(card_id, seq, final_text, answer_id)`
  - 按钮 `value` 字段携带 `{"action": "thumbs_up/thumbs_down/rate_detail", "answer_id": "..."}`
  - 用全量更新接口替换旧的 `_close_streaming_card`

- [ ] **注册卡片按钮回调** `_on_card_action()`
  - 解析 `action_value`，路由到对应处理逻辑
  - `thumbs_up/thumbs_down` → `save_thumbs()` + 卡片更新为"感谢反馈 ✅"
  - `rate_detail` → `_send_rating_form()` 私信发详细评分卡片
  - 在 `start_ws_client()` 中注册：`.register_p2_card_action_trigger_v1(_on_card_action)`

- [ ] **私信详细评分表单** `_send_rating_form(open_id, answer_id)`
  - 1-5 分按钮卡片，用户点击后调用 `save_manual_scores()`
  - 在群聊场景中避免暴露评分流程

**实现文件**: `app/feishu_bot.py`

---

## 🔧 中优先级

### 4. 性能优化

- [ ] **缓存策略**
  - [ ] Redis 缓存热门查询结果（TTL 5分钟）
  - [ ] 向量检索结果缓存（相同查询直接返回）

- [ ] **批量处理优化**
  - [ ] 向量化批量大小动态调整（根据 GPU 内存）
  - [ ] 并行处理多个查询变体

- [ ] **索引优化**
  - [ ] Milvus 索引参数调优（nlist/nprobe）
  - [ ] 定期索引重建（compact + optimize）

---

### 5. 监控与运维

- [ ] **日志增强**
  - [ ] 结构化日志（JSON 格式）
  - [ ] 慢查询日志（>2s 的请求）
  - [ ] 错误告警（钉钉/飞书/邮件）

- [ ] **健康检查**
  - [ ] `/health` 端点（检查 Milvus/LLM API 连通性）
  - [ ] Prometheus metrics 导出
  - [ ] Grafana 监控面板

- [ ] **备份恢复**
  - [ ] Milvus 数据定期备份
  - [ ] 配置文件版本管理
  - [ ] 一键恢复脚本

---

### 6. 用户体验优化

- [ ] **前端增强**
  - [ ] 搜索历史记录（本地存储）
  - [ ] 快捷问题模板（常见查询一键发送）
  - [ ] 深色模式支持
  - [ ] 移动端适配

- [ ] **交互优化**
  - [ ] 打字中断（用户输入时停止流式输出）
  - [ ] 多轮对话上下文（记住前 3 轮对话）
  - [ ] 引用溯源（点击参考来源跳转原文）

- [ ] **导出功能**
  - [ ] 对话导出为 Markdown/PDF
  - [ ] 批量查询结果导出为 Excel

---

## 📚 低优先级

### 7. 多语言支持

- [ ] 英文界面（i18n）
- [ ] 多语言文档检索（中英混合）
- [ ] 自动语言检测

---

### 8. 高级功能

- [ ] **多模态支持**
  - [ ] 图片上传识别（OCR）
  - [ ] PDF 表格提取
  - [ ] 语音输入（ASR）

- [ ] **协作功能**
  - [ ] 多用户权限管理
  - [ ] 问答分享链接
  - [ ] 团队知识库（多租户）

---

## ✅ 已完成

- [x] Milvus VARCHAR 字节溢出修复
- [x] Chat API 流式响应兼容
- [x] 日志系统增强
- [x] OpenAPI 规范化（OpenClaw 接入）
- [x] 聚合查询修复
- [x] 元数据精确过滤路由
- [x] 聚合关键词量词变体正则修复
- [x] 开源前脱敏处理
- [x] 飞书 WebSocket 长连接机器人
- [x] 回答质量评价体系（基础框架）
- [x] MCP 数据源集成（基础框架）
- [x] 文件上传入库（4步向导）
- [x] AI 分块分析流式输出
- [x] Milvus gRPC 64MB 批量写入修复
- [x] 内容 Hash 去重（跨文件重叠处理）
- [x] 上传配置 AI 助手（多轮对话）
- [x] hashlib 缺失导入修复

---

## 📝 备注

- 优先级根据业务价值和技术复杂度综合评估
- 每个任务完成后更新状态并移至"已完成"区域
- 新需求随时补充到对应优先级分类
