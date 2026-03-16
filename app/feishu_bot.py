"""
飞书机器人长连接模块（WebSocket 持久连接 + Card Kit 流式回复）
==============================================
使用飞书官方 SDK lark-oapi 的 WebSocket 长连接模式：
  - 无需公网域名 / HTTPS
  - 无需配置 Encrypt Key 或 Verification Token
  - 应用启动时自动建立出站 WebSocket 连接，断线自动重连

流式回复流程（Card Kit Streaming）：
  1. 收到用户消息
  2. 调用 Card Kit API 创建流式卡片，立即回复（用户看到“思考中…”）
  3. 运行 RAG 流式查询，每 150ms 节流更新一次卡片（满足飞书约 10 次/秒限制）
  4. 查询完成，附上来源引用并 streaming_finish=true 定稿卡片

环境变量（在 .env 中配置，仅需两项）：
  FEISHU_APP_ID       飞书应用 App ID
  FEISHU_APP_SECRET   飞书应用 App Secret
"""
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
from lark_oapi.ws.client import Client as _LarkWSClient
from lark_oapi.ws.model import ClientConfig

from app.config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_BOT_ENABLED
from app.rag import answer_stream


# ========== 快速重连 WebSocket 客户端 ==========
# 覆盖 lark.ws.Client._configure：
#   _reconnect_nonce  默认 30s 随机抖动 → 0（断线立即重连，消除 0~30s 随机等待）
#   _reconnect_interval 默认 120s → 3s（每次尝试间隔缩短）
#   _ping_interval    默认 120s → 20s（更快检测静默断线）
class _FastWSClient(_LarkWSClient):
    def _configure(self, conf: ClientConfig) -> None:
        super()._configure(conf)
        self._reconnect_nonce = 0    # 禁用随机等待抖动，断线后立即重连
        self._reconnect_interval = 3 # 重连失败后最多等 3s 再试
        self._ping_interval = 600    # 心跳 10 分钟一次，节省飞书 API 调用次数

logger = logging.getLogger("crm_feishu")
logger.setLevel(logging.DEBUG)  # 开启 DEBUG 以便排查飞书 API 请求/响应

# 线程池：执行阻塞型 RAG 查询，最多 4 个并发
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="feishu_rag")

_api_client: lark.Client | None = None
_ws_thread: threading.Thread | None = None

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"

# tenant_access_token 缓存
_token_cache: dict = {"token": "", "expire_at": 0.0}

# 机器人自身 open_id 缓存
_bot_open_id: str = ""

# Card Kit 流式更新节流间隔（飞书元素内容接口限制约 10 次/秒）
_UPDATE_INTERVAL = 0.15  # 秒


def _get_api_client() -> lark.Client:
    global _api_client
    if _api_client is None:
        _api_client = (
            lark.Client.builder()
            .app_id(FEISHU_APP_ID)
            .app_secret(FEISHU_APP_SECRET)
            .build()
        )
    return _api_client


# ========== tenant_access_token（带缓存）==========

def _get_token() -> str:
    """获取 tenant_access_token，提前 5 分钟自动刷新"""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expire_at"]:
        return _token_cache["token"]
    url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
    with httpx.Client(timeout=10) as client:
        resp = client.post(url, json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET})
        data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"获取 token 失败: {data}")
    _token_cache["token"] = data["tenant_access_token"]
    _token_cache["expire_at"] = now + data.get("expire", 7200) - 300
    logger.info("[feishu] tenant_access_token 已刷新")
    return _token_cache["token"]


def _get_bot_open_id() -> str:
    """获取机器人自身的 open_id（带缓存，仅首次调用时请求 API）"""
    global _bot_open_id
    if _bot_open_id:
        return _bot_open_id
    url = f"{FEISHU_API_BASE}/bot/v3/info"
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, headers=_headers())
        data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"获取机器人信息失败: {data}")
    _bot_open_id = data["bot"]["open_id"]
    logger.info(f"[feishu] 机器人 open_id={_bot_open_id}")
    return _bot_open_id


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }


# ========== Card Kit 流式卡片（JSON 2.0）==========
# 参考：https://github.com/openclaw/openclaw/tree/main/extensions/feishu/src/streaming-card.ts

def _create_streaming_card() -> str:
    """
    调用 Card Kit API 创建流式卡片，返回 card_id。
    卡片 JSON 2.0，body 内 markdown 组件必须带 element_id="content"，
    后续更新通过元素接口按此 element_id 精准更新。
    """
    card_json = {
        "schema": "2.0",
        "config": {
            "streaming_mode": True,
            "update_multi": True,
            "summary": {"content": "生成中…"},
            "streaming_config": {
                "print_frequency_ms": {"default": 50},
                "print_step": {"default": 1},
            },
        },
        "body": {
            "elements": [
                {"tag": "markdown", "content": "⏳ 思考中…", "element_id": "content"}
            ]
        },
    }
    url = f"{FEISHU_API_BASE}/cardkit/v1/cards"
    payload = {"type": "card_json", "data": json.dumps(card_json, ensure_ascii=False)}
    logger.debug(f"[feishu] 创建流式卡片 payload={payload}")
    with httpx.Client(timeout=15) as client:
        resp = client.post(url, headers=_headers(), json=payload)
        data = resp.json()
    logger.debug(f"[feishu] 创建流式卡片响应: {data}")
    if data.get("code") != 0:
        raise RuntimeError(f"创建流式卡片失败: {data}")
    card_id: str = data["data"]["card_id"]
    logger.info(f"[feishu] 已创建流式卡片 card_id={card_id}")
    return card_id


def _reply_with_card(message_id: str, card_id: str) -> None:
    """
    用流式卡片回复指定消息。
    content 字段格式为 {"type":"card","data":{"card_id":"..."}}
    """
    url = f"{FEISHU_API_BASE}/im/v1/messages/{message_id}/reply"
    content_str = json.dumps({"type": "card", "data": {"card_id": card_id}}, ensure_ascii=False)
    payload = {"msg_type": "interactive", "content": content_str}
    logger.debug(f"[feishu] 卡片回复 content={content_str}")
    with httpx.Client(timeout=15) as client:
        resp = client.post(url, headers=_headers(), json=payload)
        data = resp.json()
    logger.debug(f"[feishu] 卡片回复响应: {data}")
    if data.get("code") != 0:
        raise RuntimeError(f"卡片回复失败: {data}")
    logger.info(f"[feishu] 卡片回复成功 message_id={message_id} card_id={card_id}")


def _update_card_content(card_id: str, sequence: int, text: str) -> None:
    """
    通过 Card Kit 元素接口更新 element_id=content 的 markdown 组件内容。
    使用专用元素内容接口而非全量更新卡片，速率限制更宽松。
    """
    url = f"{FEISHU_API_BASE}/cardkit/v1/cards/{card_id}/elements/content/content"
    payload = {
        "content": text,
        "sequence": sequence,
        "uuid": f"s_{card_id}_{sequence}",
    }
    with httpx.Client(timeout=10) as client:
        resp = client.put(url, headers=_headers(), json=payload)
        data = resp.json()
    if data.get("code") != 0:
        logger.warning(f"[feishu] 更新卡片内容失败 seq={sequence}: {data}")
    else:
        logger.debug(f"[feishu] 卡片内容更新 seq={sequence}")


def _close_streaming_card(card_id: str, sequence: int, final_text: str) -> None:
    """
    通过 PATCH settings 接口关闭流式模式，并写入最终摘要。
    """
    # 先确保最终内容已更新
    _update_card_content(card_id, sequence, final_text)
    sequence += 1

    summary = final_text[:50] + ("…" if len(final_text) > 50 else "")
    url = f"{FEISHU_API_BASE}/cardkit/v1/cards/{card_id}/settings"
    settings = json.dumps(
        {"config": {"streaming_mode": False, "summary": {"content": summary}}},
        ensure_ascii=False,
    )
    payload = {
        "settings": settings,
        "sequence": sequence,
        "uuid": f"c_{card_id}_{sequence}",
    }
    with httpx.Client(timeout=15) as client:
        resp = client.patch(url, headers=_headers(), json=payload)
        data = resp.json()
    if data.get("code") != 0:
        logger.warning(f"[feishu] 关闭流式失败: {data}")
    else:
        logger.info(f"[feishu] 流式卡片已定稿 card_id={card_id}")


def _reply_text(message_id: str, text: str) -> None:
    """纯文本回复（卡片失败时的兜底方案）"""
    url = f"{FEISHU_API_BASE}/im/v1/messages/{message_id}/reply"
    payload = {
        "msg_type": "text",
        "content": json.dumps({"text": text}, ensure_ascii=False),
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_headers(), json=payload)
        data = resp.json()
    if data.get("code") != 0:
        logger.error(f"[feishu] 文本回复失败: {data}")
    else:
        logger.info(f"[feishu] 文本回复成功 message_id={message_id}")


# ========== RAG 查询 + 流式卡片更新 ==========

def _process_question(question: str, message_id: str) -> None:
    """
    流式 RAG 问答 + Card Kit 流式卡片回复。

    并行优化：卡片创建/回复 与 RAG 流式查询 同时启动。
    时序：
      t=0   同时启动：① _create_streaming_card + _reply_with_card
                      ② answer_stream（RAG + LLM，tokens 先缓冲）
      t≈1s  ① 卡片 ready → 把已积累的 tokens 立即 flush
      t>1s  每 150ms 节流推送后续 tokens（用户实时看到内容增长）
      t=N   RAG 完成 → 含来源引用定稿
    """
    logger.info(f"[feishu] 开始流式 RAG 查询: {question!r}")

    # 预热 token，避免卡片创建时串行等待 token 刷新
    try:
        _get_token()
    except Exception:
        pass

    # ① 卡片创建 + 回复在独立线程中并行进行
    card_ready = threading.Event()
    card_id_holder: list[str] = []
    card_error: list[Exception] = []

    def _setup_card() -> None:
        try:
            cid = _create_streaming_card()
            _reply_with_card(message_id, cid)
            card_id_holder.append(cid)
        except Exception as e:
            card_error.append(e)
        finally:
            card_ready.set()

    threading.Thread(target=_setup_card, daemon=True).start()

    # ② RAG 流式消费，与①并行
    tokens: list[str] = []
    hits: list = []
    rag_done = threading.Event()
    rag_error: list[Exception] = []

    def _run_rag() -> None:
        nonlocal hits
        try:
            for raw in answer_stream(question):
                data_str = raw.removeprefix("data:").strip()
                if not data_str:
                    continue
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                chunk_type = chunk.get("type", "")
                if chunk_type == "token":
                    tokens.append(chunk.get("content", ""))
                elif chunk_type == "sources":
                    hits = chunk.get("hits", [])
                elif chunk_type == "error":
                    raise RuntimeError(chunk.get("content", "流式接口返回错误"))
        except Exception as e:
            rag_error.append(e)
            logger.error(f"[feishu] RAG 查询失败: {e}", exc_info=True)
        finally:
            rag_done.set()

    rag_thread = threading.Thread(target=_run_rag, daemon=True)
    rag_thread.start()

    # 等待卡片就绪（一般 ~1s，RAG 也在跑）
    card_ready.wait(timeout=15)

    if card_error:
        logger.error(f"[feishu] 卡片创建/回复失败: {card_error[0]}", exc_info=True)
        rag_done.wait(timeout=60)
        _reply_text(message_id, "".join(tokens) or f"⚠️ 服务暂时不可用：{card_error[0]}")
        return

    card_id = card_id_holder[0]
    seq = 2  # 创建时已消耗 sequence=1
    last_update = time.time()

    def flush() -> None:
        nonlocal seq, last_update
        _update_card_content(card_id, seq, "".join(tokens))
        seq += 1
        last_update = time.time()

    # 节流推送：每 _UPDATE_INTERVAL 把已积累 tokens 推送到卡片
    while not rag_done.is_set():
        if tokens and time.time() - last_update >= _UPDATE_INTERVAL:
            flush()
        time.sleep(0.02)  # 20ms 轮询，响应粒度足够细

    # RAG 完成，最后 flush 一次确保内容完整
    if rag_error:
        _close_streaming_card(card_id, seq, f"⚠️ 查询失败：{rag_error[0]}")
        return

    final_text = "".join(tokens).strip() or "⚠️ 未能获取回答，请稍后重试。"
    if hits:
        lines = ["\n\n📎 **参考来源：**"]
        for s in hits[:3]:
            lines.append(
                f"- [{s.get('date', '')}] {s.get('owner', '')} / "
                f"{s.get('company', '')}（相关度 {s.get('score', 0):.2f}）"
            )
        final_text += "\n".join(lines)

    try:
        _close_streaming_card(card_id, seq, final_text)
    except Exception as e:
        logger.error(f"[feishu] 定稿卡片失败: {e}", exc_info=True)


# ========== 飞书事件处理器 ==========

def _on_message(data: P2ImMessageReceiveV1) -> None:
    """接收飞书消息事件（由 lark_oapi SDK 在 WS 线程中回调）"""
    msg = data.event.message
    message_id: str = msg.message_id
    msg_type: str = msg.message_type
    chat_type: str = msg.chat_type  # "p2p" | "group"

    # 仅处理文本消息
    if msg_type != "text":
        logger.debug(f"[feishu] 忽略非文本消息 type={msg_type}")
        return

    # 群聊中检查是否被 @ 机器人本身，否则忽略
    if chat_type == "group":
        mentions = msg.mentions if hasattr(msg, "mentions") else []
        if not mentions:
            logger.debug(f"[feishu] 群聊消息未 @ 任何人，忽略 message_id={message_id}")
            return
        try:
            bot_open_id = _get_bot_open_id()
            mentioned_ids = [
                (m.id.open_id if hasattr(m, "id") and m.id else "")
                for m in mentions
            ]
            if bot_open_id not in mentioned_ids:
                logger.debug(f"[feishu] 群聊消息未 @ 机器人，忽略 message_id={message_id}")
                return
        except Exception as e:
            logger.warning(f"[feishu] 无法获取机器人 open_id，跳过 @ 检查: {e}")
        logger.debug(f"[feishu] 群聊消息已 @ 机器人，处理中")

    # 解析 content（飞书文本消息 content 是 JSON 字符串：{"text":"..."}）
    try:
        content_obj = json.loads(msg.content)
        question = content_obj.get("text", "").strip()
    except Exception:
        question = str(msg.content).strip()

    if not question:
        return

    # 群聊中去掉 @xxx 前缀（飞书会在 text 中插入 @机器人昵称）
    if chat_type == "group":
        question = re.sub(r"@\S+\s*", "", question).strip()
        if not question:
            return

    logger.info(f"[feishu] 收到问题 [{chat_type}] message_id={message_id}: {question!r}")

    # 提交到线程池异步处理，立即返回（不阻塞 WS 事件循环）
    _executor.submit(_process_question, question, message_id)


# ========== 启动长连接客户端 ==========

def start_ws_client() -> None:
    """
    在守护线程中启动飞书 WebSocket 长连接客户端。
    由 FastAPI lifespan 在应用启动时调用。
    """
    global _ws_thread

    if not FEISHU_BOT_ENABLED:
        logger.info("[feishu] 飞书机器人已禁用（FEISHU_BOT_ENABLED=false）")
        return

    if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
        logger.warning(
            "[feishu] 未配置 FEISHU_APP_ID / FEISHU_APP_SECRET，飞书长连接已跳过"
        )
        return

    event_dispatcher = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(_on_message)
        .build()
    )

    ws_client = _FastWSClient(
        FEISHU_APP_ID,
        FEISHU_APP_SECRET,
        event_handler=event_dispatcher,
        log_level=lark.LogLevel.INFO,
    )

    def _run() -> None:
        logger.info("[feishu] WebSocket 长连接客户端启动中…")
        try:
            ws_client.start()  # 阻塞，内部自动重连
        except Exception as e:
            logger.error(f"[feishu] 长连接异常退出: {e}", exc_info=True)

    _ws_thread = threading.Thread(target=_run, name="feishu-ws", daemon=True)
    _ws_thread.start()
    logger.info("[feishu] 飞书长连接线程已启动（daemon=True）")
