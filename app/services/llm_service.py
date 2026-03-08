from collections.abc import Generator
from threading import Lock
from typing import Any

from openai import OpenAI

from app.config import AppConfig


class ChatService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OpenAI(base_url=config.api_base_url, api_key=config.api_key) if config.api_key else None
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = Lock()

    def _system_prompt(self, mode: str) -> str:
        if mode == "analyst":
            return "你是分析师助手。回答要结构化、可执行，并尽量给出关键结论。"
        return "你是通用中文助手。回答简洁、准确。"

    def _append_session_message(self, session_id: str, role: str, content: str) -> None:
        if not content:
            return

        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": role, "content": content})
            max_messages = max(2, self.config.history_max_messages)
            if len(history) > max_messages:
                self._sessions[session_id] = history[-max_messages:]

    def _build_messages(self, session_id: str, mode: str) -> list[dict[str, str]]:
        with self._lock:
            history = list(self._sessions.get(session_id, []))

        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt(mode)}]
        messages.extend(history)
        return messages

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def stream_chat_events(self, session_id: str, mode: str, user_message: str) -> Generator[dict[str, Any], None, None]:
        if self.client is None:
            yield {
                "event": "error",
                "content_type": "text",
                "payload": {"message": "后端未配置 YUNWU_API_KEY，无法调用模型接口。"},
            }
            return

        self._append_session_message(session_id=session_id, role="user", content=user_message)
        messages = self._build_messages(session_id=session_id, mode=mode)

        full_text = ""
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                stream=True,
                timeout=self.config.request_timeout,
            )

            for chunk in stream:
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if not content:
                    continue

                full_text += content
                yield {
                    "event": "token",
                    "content_type": "text",
                    "payload": {"delta": content},
                }

            assistant_text = full_text or "处理完成。"
            self._append_session_message(session_id=session_id, role="assistant", content=assistant_text)

            yield {
                "event": "final",
                "content_type": "text",
                "payload": {"text": assistant_text},
            }
        except Exception as exc:
            yield {
                "event": "error",
                "content_type": "text",
                "payload": {"message": f"模型调用失败：{str(exc)}"},
            }
