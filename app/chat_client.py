"""
Chat API 统一客户端
消除 rag.py、advanced_rag.py、main.py 中的重复代码
"""
import json
import logging
from typing import List, Dict, Any, Iterator

import httpx

from app.config import (
    CHAT_API_KEY,
    CHAT_BASE_URL,
    OPENAI_CHAT_MODEL,
    CHAT_API_MODE,
)

logger = logging.getLogger("chat_client")


class ChatClient:
    """统一的 Chat API 客户端，支持 completions 和 responses 两种模式"""

    def __init__(
        self,
        api_key: str = CHAT_API_KEY,
        base_url: str = CHAT_BASE_URL,
        model: str = OPENAI_CHAT_MODEL,
        mode: str = CHAT_API_MODE,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mode = mode

    def _get_url(self) -> str:
        """获取 API 端点 URL"""
        base = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        if self.mode == "responses":
            return f"{base}/responses"
        return f"{base}/chat/completions"

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: List[Dict], stream: bool) -> Dict[str, Any]:
        """根据 mode 构建请求体"""
        if self.mode == "responses":
            # OpenAI Responses API: messages -> input，system 提取为 instructions
            system_text = ""
            input_msgs = []
            for m in messages:
                if m["role"] == "system":
                    system_text = m["content"]
                else:
                    input_msgs.append({"role": m["role"], "content": m["content"]})

            payload = {
                "model": self.model,
                "input": input_msgs,
                "temperature": 0.1,
                "max_output_tokens": 2048,
                "stream": stream,
            }
            if system_text:
                payload["instructions"] = system_text
            return payload
        else:
            # 标准 Chat Completions API
            return {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2048,
                "stream": stream,
            }

    def complete(self, messages: List[Dict], timeout: int = 120) -> Dict[str, Any]:
        """
        非流式调用，返回完整响应。

        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            timeout: 超时时间（秒）

        Returns:
            API 响应 JSON

        Raises:
            httpx.HTTPStatusError: HTTP 错误
            RuntimeError: 响应格式异常
        """
        url = self._get_url()
        payload = self._build_payload(messages, stream=False)

        logger.info(f"[ChatClient] 调用 {url}  model={self.model}  mode={self.mode}")

        with httpx.Client() as client:
            resp = client.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()

    def extract_answer(self, response: Dict[str, Any]) -> str:
        """
        从 API 响应中提取回答文本。

        Args:
            response: API 响应 JSON

        Returns:
            回答文本

        Raises:
            RuntimeError: 响应格式异常
        """
        if self.mode == "responses":
            try:
                return response["output"][0]["content"][0]["text"]
            except (KeyError, IndexError) as e:
                logger.error(f"[ChatClient] Responses API 响应格式异常: {json.dumps(response, ensure_ascii=False)[:300]}")
                raise RuntimeError("Responses API 响应格式异常") from e
        else:
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                logger.error(f"[ChatClient] Completions API 响应格式异常: {json.dumps(response, ensure_ascii=False)[:300]}")
                raise RuntimeError("Completions API 响应格式异常") from e

    def stream(self, messages: List[Dict], timeout: int = 120) -> Iterator[str]:
        """
        流式调用，返回 SSE 行迭代器。

        Args:
            messages: 消息列表
            timeout: 超时时间（秒）

        Yields:
            SSE 数据行（格式：data: <json>\\n\\n）

        Raises:
            httpx.HTTPStatusError: HTTP 错误
        """
        url = self._get_url()
        payload = self._build_payload(messages, stream=True)

        logger.info(f"[ChatClient] 流式调用 {url}  model={self.model}  mode={self.mode}")

        with httpx.Client() as client:
            with client.stream(
                "POST",
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=timeout
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        yield line + "\n\n"


# 全局单例
_default_client: ChatClient = None


def get_chat_client() -> ChatClient:
    """获取默认 ChatClient 实例（单例）"""
    global _default_client
    if _default_client is None:
        _default_client = ChatClient()
    return _default_client
