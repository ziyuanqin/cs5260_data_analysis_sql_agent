"""聊天核心业务逻辑。

职责：
- 按 session_id 维护内存会话历史
- 按 provider 调用不同后端并开启流式返回
- 把模型分片结果转换为前端可消费事件
"""
import os
import json
import asyncio
from typing import AsyncGenerator
from threading import Lock
from typing import Any
from urllib import request as urlrequest

from app.config import AppConfig
from app.services.providers.registry import ProviderRegistry

from langchain_core.messages import HumanMessage
from backend.SQLagent.main import get_sql_graph_app


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

    def _eda_chat(self, session_id: str, user_message: str) -> str:
        """调用 EDA 后端 /chat，并返回文本回复。"""

        endpoint = f"{self.config.eda_api_base_url.rstrip('/')}/chat"
        payload = {
            "thread_id": session_id,
            "message": user_message,
        }

        req = urlrequest.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        with urlrequest.urlopen(req, timeout=self.config.request_timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip() or "EDA 分析完成。"

        if not isinstance(parsed, dict):
            return raw.strip() or "EDA 分析完成。"

        text = parsed.get("message")
        if isinstance(text, str) and text.strip():
            return text

        detail = parsed.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail

        return raw.strip() or "EDA 分析完成。"

    async def stream_chat_events(
            self,
            session_id: str,
            mode: str,
            user_message: str,
            provider: str | None = None,
            model: str | None = None,
            provider_options: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """按顺序产出聊天事件：token/final/error。

        该生成器由路由层消费，并封装为 SSE 流返回前端。
        """

        opts = provider_options or {}
        is_eda_mode = opts.get("eda_analysis", False)
        is_sql_mode = opts.get("sql_analysis", False)

        # --- 情况 A: 运行 EDA Agent ---
        if is_eda_mode:
            self._append_session_message(session_id, "user", user_message)
            try:
                eda_text = await asyncio.to_thread(self._eda_chat, session_id, user_message)
                self._append_session_message(session_id, "assistant", eda_text)
                yield {
                    "event": "token",
                    "payload": {"delta": eda_text},
                }
                yield {
                    "event": "final",
                    "payload": {"text": eda_text},
                }
            except Exception as exc:
                yield {
                    "event": "error",
                    "payload": {"message": f"EDA 调用失败：{str(exc)}"},
                }
            return

        # --- 情况 B: 运行 SQL 专家 Agent ---
        if is_sql_mode:
            self._append_session_message(session_id, "user", user_message)
            raw_file_names = opts.get("file_name", "")
            if isinstance(raw_file_names, str):
                # 兼容 "data1.csv,data2.csv" 这种格式
                file_list = [f.strip() for f in raw_file_names.split(",") if f.strip()]
            else:
                file_list = [raw_file_names] if raw_file_names else []

            # 3. 构建绝对路径并校验文件是否存在
            valid_paths = []
            for f_name in file_list:
                # 确保路径与 upload.py 保存的位置严格一致
                full_path = os.path.join(os.getcwd(), "backend", "dataset", f_name)
                if os.path.exists(full_path):
                    valid_paths.append(full_path)
                else:
                    print(f"⚠️ 警告: 文件未找到，跳过: {full_path}")

            # 4. 获取动态编译的 Graph App
            # 每次请求动态创建 engine，保证多用户并发时数据库隔离
            graph_app = get_sql_graph_app(db_type="sqlite")

            config = {"configurable": {"thread_id": session_id}}
            inputs = {
                "messages": [HumanMessage(content=user_message)],
                "db_type": "sqlite",
                "excel_paths": valid_paths, # 这里现在是完整的路径列表
                "retry_count": 0
            }

            full_analysis_text = "" # 用于保存完整回复
            try:
                async for chunk in graph_app.astream(inputs, config=config, stream_mode="updates"):
                    if "sql_gen" in chunk:
                        sql = chunk["sql_gen"].get("sql_query")
                        yield {
                            "event": "token",
                            "payload": {"delta": f"\n> **🔍 正在生成 SQL:**\n> ```sql\n> {sql}\n> ```\n"}
                        }

                    if "analysis" in chunk:
                        ans = chunk["analysis"].get("analysis", "")
                        full_analysis_text += ans # 这一步非常重要！
                        yield {
                            "event": "token",
                            "payload": {"delta": ans}
                        }

                # 只有保存了，下一次对话才能带上这个上下文
                if full_analysis_text:
                    self._append_session_message(session_id, "assistant", full_analysis_text)
                    yield {
                        "event": "final",
                        "payload": {"text": full_analysis_text}
                    }

            except Exception as exc:
                yield {
                    "event": "error",
                    "payload": {"message": f"SQL Agent 运行出错：{str(exc)}"}
                }
        else:

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
                    yield {
                        "event": "token",
                        "payload": {"delta": chunk},
                    }
                    await asyncio.sleep(0)

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
