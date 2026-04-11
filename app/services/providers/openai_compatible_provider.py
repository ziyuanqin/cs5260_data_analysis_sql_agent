"""OpenAI 兼容协议提供方实现。"""

from __future__ import annotations

from collections.abc import Generator

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
        self.client = OpenAI(base_url=base_url, api_key=api_key) if (api_key and OpenAI is not None) else None

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        timeout: int,
        provider_options: dict | None = None,
    ) -> Generator[str, None, None]:
        if OpenAI is None:
            raise RuntimeError("未安装 openai 依赖。请先安装 `pip install openai`。")
        if self.client is None:
            raise RuntimeError("未配置 OpenAI 兼容后端密钥。")

        safe_options = {
            key: value
            for key, value in (provider_options or {}).items()
            if key in self._ALLOWED_CHAT_COMPLETION_OPTIONS
        }

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
