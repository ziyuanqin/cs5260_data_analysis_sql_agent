"""OpenAI 兼容协议提供方实现。"""

from collections.abc import Generator

from openai import OpenAI


class OpenAICompatibleProvider:
    """适配 OpenAI SDK 的流式调用。"""

    def __init__(self, base_url: str, api_key: str | None):
        self.client = OpenAI(base_url=base_url, api_key=api_key) if api_key else None

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        timeout: int,
        provider_options: dict | None = None,
    ) -> Generator[str, None, None]:
        if self.client is None:
            raise RuntimeError("未配置 OpenAI 兼容后端密钥。")

        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            timeout=timeout,
            **(provider_options or {}),
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
