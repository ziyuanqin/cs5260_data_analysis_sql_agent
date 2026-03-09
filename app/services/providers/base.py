"""后端提供方适配器抽象定义。"""

from collections.abc import Generator
from typing import Protocol


class LLMProvider(Protocol):
    """统一的模型流式输出接口。"""

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        timeout: int,
        provider_options: dict | None = None,
    ) -> Generator[str, None, None]:
        """返回文本分片流。"""
