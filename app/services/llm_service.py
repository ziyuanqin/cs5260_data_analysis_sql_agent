"""聊天核心业务逻辑。

职责：
- 按 session_id 维护内存会话历史
- 按 provider 调用不同后端并开启流式返回
- 把模型分片结果转换为前端可消费事件
"""

from collections.abc import Generator
from threading import Lock
from typing import Any

from app.config import AppConfig
from app.services.providers.registry import ProviderRegistry


class ChatService:
    """供路由层复用的聊天服务对象。"""

    def __init__(self, config: AppConfig):
        self.config = config
        # 提供方注册器，按名称返回不同后端适配器。
        self.provider_registry = ProviderRegistry(config)
        # 会话内存结构：session_id -> [{role, content}, ...]
        self._sessions: dict[str, list[dict[str, str]]] = {}
        # 多请求并发场景下，使用锁保护会话字典。
        self._lock = Lock()

    def list_supported_providers(self) -> list[str]:
        """返回当前后端支持的 provider 列表。"""

        return self.provider_registry.list_supported()

    def _resolve_provider_name(self, provider: str | None) -> str:
        """解析本次请求使用的 provider 名称。"""

        return provider or self.config.default_provider

    def _resolve_model_name(self, provider_name: str, model: str | None) -> str:
        """解析本次请求使用的模型名。"""

        if model:
            return model
        if provider_name == "local_http":
            return self.config.local_model_name
        return self.config.openai_model_name

    def _system_prompt(self, mode: str) -> str:
        """根据模式返回对应系统提示词。"""

        if mode == "analyst":
            return "你是分析师助手。回答要结构化、可执行，并尽量给出关键结论。"
        return "你是通用中文助手。回答简洁、准确。"

    def _append_session_message(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息，并在必要时裁剪历史长度。"""

        if not content:
            return

        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": role, "content": content})
            # 仅保留最近消息，控制上下文长度与调用成本。
            max_messages = max(2, self.config.history_max_messages)
            if len(history) > max_messages:
                self._sessions[session_id] = history[-max_messages:]

    def _build_messages(self, session_id: str, mode: str) -> list[dict[str, str]]:
        """构建发送给模型接口的消息列表。"""

        with self._lock:
            history = list(self._sessions.get(session_id, []))

        # 模型接口输入由系统提示词 + 历史对话组成。
        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt(mode)}]
        messages.extend(history)
        return messages

    def reset_session(self, session_id: str) -> bool:
        """删除单个会话历史；若会话存在则返回 True。"""

        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def stream_chat_events(
        self,
        session_id: str,
        mode: str,
        user_message: str,
        provider: str | None = None,
        model: str | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """按顺序产出聊天事件：token/final/error。

        该生成器由路由层消费，并封装为 SSE 流返回前端。
        """

        provider_name = self._resolve_provider_name(provider)
        selected_model = self._resolve_model_name(provider_name, model)

        # 先保存用户消息，使其进入本轮上下文。
        self._append_session_message(session_id=session_id, role="user", content=user_message)
        messages = self._build_messages(session_id=session_id, mode=mode)

        # 流式过程中持续累积完整回复文本。
        full_text = ""
        try:
            provider_impl = self.provider_registry.get(provider_name)
            stream = provider_impl.stream_chat(
                model=selected_model,
                messages=messages,
                timeout=self.config.request_timeout,
                provider_options=provider_options,
            )

            for chunk in stream:
                if not isinstance(chunk, str) or not chunk:
                    continue

                full_text += chunk
                # token 事件供前端实时增量渲染。
                yield {
                    "event": "token",
                    "content_type": "text",
                    "payload": {"delta": chunk},
                }

            # 保存助手完整回复，作为后续对话上下文。
            assistant_text = full_text or "处理完成。"
            self._append_session_message(session_id=session_id, role="assistant", content=assistant_text)

            # final 事件返回完整文本，作为兜底结果。
            yield {
                "event": "final",
                "content_type": "text",
                "payload": {"text": assistant_text},
            }
        except Exception as exc:
            # 将模型调用异常统一转换为错误事件给前端处理。
            yield {
                "event": "error",
                "content_type": "text",
                "payload": {"message": f"模型调用失败：{str(exc)}"},
            }
