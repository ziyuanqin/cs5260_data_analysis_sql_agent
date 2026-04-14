"""OpenAI 兼容协议提供方实现。"""

from __future__ import annotations

import json
import os
from collections.abc import Generator

import requests

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    OpenAI = None  # type: ignore[assignment]


class OpenAICompatibleProvider:
    """适配 OpenAI SDK 的流式调用。"""

    _ALLOWED_CHAT_COMPLETION_OPTIONS = {
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "n",
        "user",
        "tools",
        "tool_choice",
        "response_format",
        "parallel_tool_calls",
        "stream_options",
    }

    def __init__(self, base_url: str, api_key: str | None):
        self._base_url = str(base_url or "").strip()
        self._api_key = str(api_key or "").strip() or None
        self.client = OpenAI(base_url=self._base_url, api_key=self._api_key) if (self._api_key and OpenAI is not None) else None

    def _stream_chat_http(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        timeout: int,
        provider_options: dict | None,
    ) -> Generator[str, None, None]:
        base = self._base_url.rstrip("/")
        url = f"{base}/chat/completions"

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # OpenRouter recommends these headers (not always required). Add safe defaults.
        if "openrouter.ai" in base.lower():
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
            headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE", "CS5260-Agent")

        safe_options = {
            key: value
            for key, value in (provider_options or {}).items()
            if key in self._ALLOWED_CHAT_COMPLETION_OPTIONS
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **safe_options,
        }

        with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            # Force UTF-8 decoding to avoid mojibake when upstream omits charset.
            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                else:
                    line = str(raw_line).strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") if isinstance(event, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        timeout: int,
        provider_options: dict | None = None,
    ) -> Generator[str, None, None]:
        if not self._api_key:
            raise RuntimeError("未配置 OpenAI 兼容后端密钥。")

        # Prefer HTTP streaming for OpenRouter to ensure compatible headers and avoid SDK version issues.
        if "openrouter.ai" in self._base_url.lower():
            yield from self._stream_chat_http(
                messages=messages,
                model=model,
                timeout=timeout,
                provider_options=provider_options,
            )
            return

        safe_options = {
            key: value
            for key, value in (provider_options or {}).items()
            if key in self._ALLOWED_CHAT_COMPLETION_OPTIONS
        }

        if self.client is not None:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                timeout=timeout,
                **safe_options,
            )

            for chunk in stream:
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    yield content
            return

        # SDK missing: fall back to raw HTTP streaming.
        yield from self._stream_chat_http(
            messages=messages,
            model=model,
            timeout=timeout,
            provider_options=provider_options,
        )
