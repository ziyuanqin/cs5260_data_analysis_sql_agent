"""提供方注册与选择逻辑。"""

from app.config import AppConfig
from app.services.providers.base import LLMProvider
from app.services.providers.huggingface_provider import HuggingFaceProvider
from app.services.providers.local_http_provider import LocalHTTPProvider
from app.services.providers.openai_compatible_provider import OpenAICompatibleProvider


class ProviderRegistry:
    """按名称返回对应的后端适配器。"""

    def __init__(self, config: AppConfig):
        self._providers: dict[str, LLMProvider] = {
            "huggingface": HuggingFaceProvider(
                api_url=config.huggingface_api_url,
                api_key=config.huggingface_api_key,
            ),
            "openai_compatible": OpenAICompatibleProvider(
                base_url=config.openai_api_base_url,
                api_key=config.openai_api_key,
            ),
            "deepseek_compatible": OpenAICompatibleProvider(
                base_url=config.deepseek_api_base_url,
                api_key=config.deepseek_api_key,
            ),
            "local_http": LocalHTTPProvider(api_url=config.local_api_url),
        }

    def get(self, provider_name: str) -> LLMProvider:
        provider = self._providers.get(provider_name)
        if provider is None:
            supported = ", ".join(sorted(self._providers.keys()))
            raise ValueError(f"不支持的 provider: {provider_name}。可用 provider: {supported}")
        return provider

    def list_supported(self) -> list[str]:
        return sorted(self._providers.keys())
