"""本地 HTTP 后端提供方实现。"""

from __future__ import annotations

import json
from collections.abc import Generator
from urllib import request


class LocalHTTPProvider:
    """通过 HTTP 请求本地模型服务，并适配多种流式格式。"""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        timeout: int,
        provider_options: dict | None = None,
    ) -> Generator[str, None, None]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "provider_options": provider_options or {},
        }

        req = request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=timeout) as resp:
            # 支持三种常见返回：SSE(data: ...)、JSON 行、纯文本分片。
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("data:"):
                    line = line[5:].strip()

                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if isinstance(obj.get("delta"), str) and obj["delta"]:
                            yield obj["delta"]
                            continue
                        payload_obj = obj.get("payload")
                        if isinstance(payload_obj, dict):
                            delta = payload_obj.get("delta")
                            text = payload_obj.get("text")
                            if isinstance(delta, str) and delta:
                                yield delta
                                continue
                            if isinstance(text, str) and text:
                                yield text
                                continue
                        text = obj.get("text")
                        if isinstance(text, str) and text:
                            yield text
                            continue
                except json.JSONDecodeError:
                    pass

                yield line
