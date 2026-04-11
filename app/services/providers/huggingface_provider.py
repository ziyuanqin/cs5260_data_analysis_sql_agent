"""Hugging Face Router provider (OpenAI-compatible SSE endpoint)."""

from __future__ import annotations

import json
from collections.abc import Generator

import requests


class HuggingFaceProvider:
    """Call Hugging Face chat completions endpoint with streaming."""

    def __init__(self, api_url: str, api_key: str | None):
        self.api_url = api_url
        self.api_key = api_key

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        timeout: int,
        provider_options: dict | None = None,
    ) -> Generator[str, None, None]:
        if not self.api_key:
            raise RuntimeError("未配置 Hugging Face API Token。")
        if not self.api_url:
            raise RuntimeError("未配置 Hugging Face API URL。")

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if provider_options:
            payload.update(provider_options)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
        ) as response:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = response.text.strip()
                raise RuntimeError(f"Hugging Face 请求失败: {response.status_code} {detail}") from exc

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("data:"):
                    line = line[5:].strip()

                if line == "[DONE]":
                    break

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(obj, dict):
                    continue

                choices = obj.get("choices")
                if isinstance(choices, list) and choices:
                    choice0 = choices[0] if isinstance(choices[0], dict) else {}
                    delta = choice0.get("delta") if isinstance(choice0, dict) else {}
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            yield content
                            continue
                    message = choice0.get("message") if isinstance(choice0, dict) else {}
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content:
                            yield content
                            continue

                text = obj.get("text")
                if isinstance(text, str) and text:
                    yield text
