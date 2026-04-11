"""聊天核心业务逻辑。

职责：
- 统一维护 session 历史与预算使用
- 在 general/expert 模式间做路由
- 在 expert 模式下运行 LangGraph 的 plan/execute/review/summarize 闭环
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import requests

from app.agent import GeneralAgentState, build_general_agent_graph
from app.config import AppConfig
from app.services.memory_store import MemoryStore
from app.services.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

_STAGE_TO_MODEL_ATTR = {
    "intent": "agent_intent_model_name",
    "planner": "agent_planner_model_name",
    "executor": "agent_executor_model_name",
    "reviewer": "agent_reviewer_model_name",
    "summarizer": "agent_summarizer_model_name",
}

_PLANNING_KEYWORDS = (
    "plan",
    "steps",
    "roadmap",
    "implement",
    "debug",
    "architecture",
    "optimize",
    "design",
    "拆解",
    "步骤",
    "方案",
    "规划",
    "实现",
    "调试",
    "排查",
    "架构",
)

_EXPERT_MODES = {"expert", "analyst"}
_CONTROL_PROVIDER_OPTIONS = {"general_model", "eda_analysis", "sql_analysis", "file_name"}
_TOOL_OUTPUT_MAX_CHARS = 5000


class BudgetExceededError(RuntimeError):
    """Raised when session-level budget guardrails are exceeded."""


class ChatService:
    """供路由层复用的聊天服务对象。"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.provider_registry = ProviderRegistry(config)
        self._memory_store = MemoryStore(config.frontend_dir.parent / "runtime" / "memory.db")
        # session_id -> [{role, content}, ...]
        self._sessions: dict[str, list[dict[str, str]]] = {}
        # session_id -> {"estimated_tokens": int, "estimated_cost_usd": float}
        self._usage: dict[str, dict[str, float]] = {}
        self._lock = Lock()

    def _hydrate_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            if session_id in self._sessions:
                return
        messages = self._memory_store.load_messages(session_id, limit=self.config.history_max_messages)
        if messages:
            with self._lock:
                self._sessions[session_id] = messages[-self.config.history_max_messages :]
        usage = self._memory_store.load_usage(session_id)
        if usage:
            with self._lock:
                self._usage[session_id] = usage

    def list_supported_providers(self) -> list[str]:
        return self.provider_registry.list_supported()

    def _normalize_mode(self, mode: str) -> str:
        return "expert" if str(mode).strip().lower() in _EXPERT_MODES else "general"

    def _resolve_provider_name(self, provider: str | None) -> str:
        if provider:
            return provider

        if self.config.default_provider == "huggingface":
            if not self.config.huggingface_api_key and self.config.openai_api_key:
                logger.warning("Hugging Face token missing, fallback default provider to openai_compatible.")
                return "openai_compatible"
        return self.config.default_provider

    def _resolve_provider_fallbacks(self, primary: str) -> list[str]:
        supported = set(self.provider_registry.list_supported())
        order: list[str] = []

        def add(name: str) -> None:
            if name in supported and name not in order:
                order.append(name)

        add(primary)
        if primary == "openai_compatible":
            add("huggingface")
        elif primary == "huggingface":
            add("openai_compatible")
        elif primary == "local_http":
            add("openai_compatible")
            add("huggingface")
        else:
            add("openai_compatible")
            add("huggingface")

        return order or [primary]

    def _classify_error(self, exc: Exception) -> str:
        text = str(exc).lower()
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if "rate limit" in text or "429" in text or "too many requests" in text:
            return "rate_limit"
        if "quota" in text or "insufficient" in text or "payment" in text:
            return "quota"
        if "connection" in text or "connect" in text or "reset" in text or "dns" in text:
            return "connection"
        if "json" in text or "parse" in text or "decode" in text:
            return "parse"
        if "500" in text or "502" in text or "503" in text or "504" in text:
            return "server"
        return "unknown"

    def _is_retryable_error(self, category: str) -> bool:
        return category in {"timeout", "rate_limit", "connection", "parse", "server"}

    def _resolve_general_backend(
        self,
        *,
        provider: str | None,
        model: str | None,
        provider_options: dict[str, Any] | None,
    ) -> tuple[str, str | None]:
        # 请求显式指定时优先尊重请求参数。
        if provider or model:
            return self._resolve_provider_name(provider), model

        alias = str((provider_options or {}).get("general_model", "")).strip().lower()
        if alias == "openai":
            return self.config.general_openai_provider, self.config.general_openai_model_name
        if alias == "deepseek":
            return self.config.general_deepseek_provider, self.config.general_deepseek_model_name
        if alias in {"huggingface", "qwen", "hf"}:
            return "huggingface", self.config.huggingface_model_name

        return self._resolve_provider_name(None), None

    def _system_prompt(self, mode: str) -> str:
        if mode == "expert":
            return "你是数据分析专家助手。回答要结构化、可执行，并尽量给出关键结论。"
        return "你是执行型 AI Agent。请尽量产出可执行、可验证、可落地的答案。"

    def _append_session_message(self, session_id: str, role: str, content: str) -> None:
        if not content:
            return

        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": role, "content": content})
            max_messages = max(2, self.config.history_max_messages)
            if len(history) > max_messages:
                self._sessions[session_id] = history[-max_messages:]
        self._memory_store.append_message(session_id, role, content)

    def _build_messages(self, session_id: str, mode: str) -> list[dict[str, str]]:
        with self._lock:
            history = list(self._sessions.get(session_id, []))

        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt(mode)}]
        messages.extend(history)
        return messages

    def _sanitize_provider_options(self, provider_options: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(provider_options, dict):
            return None
        cleaned = {k: v for k, v in provider_options.items() if k not in _CONTROL_PROVIDER_OPTIONS}
        return cleaned or None

    def _project_root(self) -> Path:
        return self.config.frontend_dir.parent.resolve()

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            return {}
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end < start:
            return {}
        candidate = text[start : end + 1]
        try:
            parsed = __import__("json").loads(candidate)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _shorten(self, value: str, max_chars: int = _TOOL_OUTPUT_MAX_CHARS) -> str:
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "\n...(truncated)"

    def _resolve_read_path(self, raw_path: str) -> Path:
        project_root = self._project_root()
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        # Security boundary: keep reads inside current project root.
        try:
            candidate.relative_to(project_root)
        except ValueError as exc:
            raise ValueError("Path is outside project workspace.") from exc
        return candidate

    def _tool_file_read(self, raw_path: str) -> str:
        if not raw_path or not raw_path.strip():
            raise ValueError("file_read requires a non-empty path.")

        target = self._resolve_read_path(raw_path.strip())
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        if not target.is_file():
            raise ValueError(f"Path is not a file: {target}")
        if target.stat().st_size > 2_000_000:
            raise ValueError("File is too large; limit is 2MB.")

        content = target.read_text(encoding="utf-8", errors="ignore")
        return self._shorten(content)

    def _strip_html(self, content: str) -> str:
        cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", content)
        cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
        cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _tool_web_read(self, raw_url: str) -> str:
        if not raw_url or not raw_url.strip():
            raise ValueError("web_read requires a non-empty url.")

        url = raw_url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http/https URLs are allowed.")

        response = requests.get(
            url,
            timeout=min(self.config.request_timeout, 20),
            headers={"User-Agent": "CS5260-Agent/1.0"},
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        text = response.text or ""
        if "text/html" in content_type or "<html" in text.lower():
            text = self._strip_html(text)
        return self._shorten(text)

    def _run_general_tool(self, tool_name: str, tool_input: str) -> str:
        normalized = tool_name.strip().lower()
        if normalized == "file_read":
            return self._tool_file_read(tool_input)
        if normalized == "web_read":
            return self._tool_web_read(tool_input)
        raise ValueError(f"Unsupported tool: {tool_name}")

    def _run_executor_stage_with_tools(
        self,
        *,
        session_id: str,
        provider_name: str,
        requested_model: str | None,
        provider_options: dict[str, Any] | None,
        step_prompt: str,
    ) -> tuple[str, str, dict[str, Any] | None]:
        router_prompt = (
            "You are an executor with optional tool usage.\n"
            "Available tools:\n"
            "1) file_read: read local text file within project workspace.\n"
            "2) web_read: fetch and read a public web page.\n"
            "Return JSON object only in one of these forms:\n"
            '{"action":"RESPOND","response":"<final step result>"}\n'
            '{"action":"TOOL","tool":"file_read|web_read","input":"<tool input>","reason":"<why needed>"}\n'
            "No extra text.\n"
            f"Step context:\n{step_prompt}"
        )
        decision_text, selected_model = self._complete_with_fallback(
            stage="executor",
            session_id=session_id,
            provider_name=provider_name,
            requested_model=requested_model,
            provider_options=provider_options,
            messages=[
                {"role": "system", "content": self._system_prompt("general")},
                {"role": "user", "content": router_prompt},
            ],
        )
        decision = self._extract_json_object(decision_text)
        action = str(decision.get("action", "")).strip().upper()
        if action != "TOOL":
            response = str(decision.get("response", "")).strip()
            return (response or decision_text.strip() or "Step executed."), selected_model, None

        tool_name = str(decision.get("tool", "")).strip().lower()
        tool_input = str(decision.get("input", "")).strip()
        tool_reason = str(decision.get("reason", "")).strip()

        tool_meta: dict[str, Any] = {
            "tool": tool_name,
            "input": tool_input[:200],
            "reason": tool_reason[:200],
        }
        try:
            tool_output = self._run_general_tool(tool_name, tool_input)
            tool_meta["status"] = "ok"
        except Exception as exc:
            tool_output = f"Tool execution failed: {str(exc)}"
            tool_meta["status"] = "error"
            tool_meta["error"] = str(exc)[:200]
        tool_meta["output_excerpt"] = self._shorten(tool_output, max_chars=800)
        if tool_name == "web_read":
            tool_meta["source"] = tool_input
        elif tool_name == "file_read":
            try:
                tool_meta["source"] = str(self._resolve_read_path(tool_input))
            except Exception:
                tool_meta["source"] = tool_input

        synth_prompt = (
            "Based on the executed tool result, complete this step.\n"
            "Output concise actionable text.\n"
            f"Step context:\n{step_prompt}\n\n"
            f"Tool: {tool_name}\n"
            f"Tool input: {tool_input}\n"
            f"Tool status: {tool_meta['status']}\n"
            f"Tool output:\n{tool_output}"
        )
        final_text, final_model = self._complete_with_fallback(
            stage="executor",
            session_id=session_id,
            provider_name=provider_name,
            requested_model=requested_model,
            provider_options=provider_options,
            messages=[
                {"role": "system", "content": self._system_prompt("general")},
                {"role": "user", "content": synth_prompt},
            ],
        )
        if not final_text.strip():
            final_text = "Step executed with tool."
        return final_text.strip(), final_model, tool_meta

    def _estimate_tokens_text(self, text: str) -> int:
        # Lightweight estimate for budget gating.
        return max(1, len(text) // 4)

    def _estimate_tokens_messages(self, messages: list[dict[str, str]]) -> int:
        total = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens_text(content)
        return max(1, total)

    def _model_price_per_1k(self, model_name: str) -> float:
        if model_name in self.config.model_price_per_1k_tokens_usd:
            return self.config.model_price_per_1k_tokens_usd[model_name]
        return self.config.default_price_per_1k_tokens_usd

    def _projected_usage(self, session_id: str, model_name: str, extra_tokens: int) -> tuple[int, float]:
        with self._lock:
            usage = self._usage.get(session_id, {"estimated_tokens": 0.0, "estimated_cost_usd": 0.0})
            used_tokens = int(usage.get("estimated_tokens", 0.0))
            used_cost = float(usage.get("estimated_cost_usd", 0.0))

        projected_tokens = used_tokens + max(0, extra_tokens)
        projected_cost = used_cost + (max(0, extra_tokens) / 1000.0) * self._model_price_per_1k(model_name)
        return projected_tokens, projected_cost

    def _assert_budget(self, session_id: str, model_name: str, extra_tokens: int) -> None:
        projected_tokens, projected_cost = self._projected_usage(session_id, model_name, extra_tokens)
        if projected_tokens > self.config.session_token_budget:
            raise BudgetExceededError(
                f"会话预算已达到 token 上限 ({self.config.session_token_budget})，请重置会话或缩短请求。"
            )
        if projected_cost > self.config.session_cost_budget_usd:
            raise BudgetExceededError(
                f"会话预算已达到费用上限 (${self.config.session_cost_budget_usd:.2f})，请重置会话或缩短请求。"
            )

    def _commit_usage(self, session_id: str, model_name: str, input_tokens: int, output_tokens: int) -> None:
        consumed = max(0, input_tokens) + max(0, output_tokens)
        if consumed <= 0:
            return

        with self._lock:
            usage = self._usage.setdefault(session_id, {"estimated_tokens": 0.0, "estimated_cost_usd": 0.0})
            usage["estimated_tokens"] = float(usage.get("estimated_tokens", 0.0)) + consumed
            usage["estimated_cost_usd"] = float(usage.get("estimated_cost_usd", 0.0)) + (
                consumed / 1000.0
            ) * self._model_price_per_1k(model_name)
            current_tokens = int(usage["estimated_tokens"])
            current_cost = float(usage["estimated_cost_usd"])
        logger.info(
            "Usage committed: session_id=%s model=%s input_tokens=%s output_tokens=%s total_tokens=%s total_cost=%.6f",
            session_id,
            model_name,
            input_tokens,
            output_tokens,
            current_tokens,
            current_cost,
        )
        self._memory_store.upsert_usage(session_id, self._get_usage_snapshot(session_id))

    def _record_task_event(self, session_id: str, payload: dict[str, Any]) -> None:
        if not session_id or not isinstance(payload, dict):
            return
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        self._memory_store.append_task_event(
            session_id,
            {
                "stage": meta.get("stage", ""),
                "message": payload.get("delta", ""),
                "model": meta.get("model"),
                "reason": meta.get("reason"),
                "step_index": meta.get("step_index"),
                "total_steps": meta.get("total_steps"),
                "retry": meta.get("retry"),
                "meta": meta,
            },
        )

    def _record_evidence(self, session_id: str, item: dict[str, Any]) -> None:
        if not session_id or not isinstance(item, dict):
            return
        self._memory_store.append_evidence(session_id, item)

    def _get_usage_snapshot(self, session_id: str) -> dict[str, float]:
        with self._lock:
            usage = self._usage.get(session_id, {"estimated_tokens": 0.0, "estimated_cost_usd": 0.0})
            return {
                "estimated_tokens": float(usage.get("estimated_tokens", 0.0)),
                "estimated_cost_usd": float(usage.get("estimated_cost_usd", 0.0)),
                "token_budget": float(self.config.session_token_budget),
                "cost_budget_usd": float(self.config.session_cost_budget_usd),
            }

    def _make_status_token_event(
        self,
        *,
        delta: str,
        session_id: str,
        stage: str,
        model: str | None = None,
        reason: str | None = None,
        step_index: int | None = None,
        retry: int | None = None,
        total_steps: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "delta": delta,
            "meta": {
                "kind": "status",
                "session_id": session_id,
                "stage": stage,
                "usage": self._get_usage_snapshot(session_id),
            },
        }
        if model:
            payload["meta"]["model"] = model
        if reason:
            payload["meta"]["reason"] = reason
        if step_index is not None:
            payload["meta"]["step_index"] = step_index
        if retry is not None:
            payload["meta"]["retry"] = retry
        if total_steps is not None:
            payload["meta"]["total_steps"] = total_steps
        return {"event": "token", "content_type": "text", "payload": payload}

    def _resolve_stage_models(self, provider_name: str, requested_model: str | None, stage: str) -> list[str]:
        candidates: list[str] = []

        def add(model_name: str | None) -> None:
            if not model_name:
                return
            normalized = model_name.strip()
            if not normalized:
                return
            if normalized not in candidates:
                candidates.append(normalized)

        if requested_model:
            add(requested_model)

        if provider_name == "local_http":
            add(self.config.local_model_name)
        elif provider_name == "huggingface":
            stage_attr = _STAGE_TO_MODEL_ATTR.get(stage)
            stage_model = getattr(self.config, stage_attr, "") if stage_attr else ""
            add(stage_model)
            add(self.config.huggingface_model_name)
        else:
            stage_attr = _STAGE_TO_MODEL_ATTR.get(stage)
            stage_model = getattr(self.config, stage_attr, "") if stage_attr else ""
            add(stage_model)
            add(self.config.openai_model_name)

        add(self.config.fallback_model_name)
        if candidates:
            return candidates
        if provider_name == "huggingface":
            return [self.config.huggingface_model_name]
        if provider_name == "local_http":
            return [self.config.local_model_name]
        return [self.config.openai_model_name]

    def _stream_chunks_with_fallback(
        self,
        *,
        session_id: str,
        provider_name: str,
        model_candidates: list[str],
        messages: list[dict[str, str]],
        provider_options: dict[str, Any] | None,
    ) -> Generator[tuple[str, str], None, None]:
        provider_impl = self.provider_registry.get(provider_name)
        input_tokens = self._estimate_tokens_messages(messages)
        last_error: Exception | None = None

        for candidate in model_candidates:
            has_output = False
            output_tokens = 0
            try:
                self._assert_budget(session_id=session_id, model_name=candidate, extra_tokens=input_tokens)
                stream = provider_impl.stream_chat(
                    model=candidate,
                    messages=messages,
                    timeout=self.config.request_timeout,
                    provider_options=provider_options,
                )

                for chunk in stream:
                    if not isinstance(chunk, str) or not chunk:
                        continue
                    has_output = True
                    output_tokens += self._estimate_tokens_text(chunk)
                    self._assert_budget(
                        session_id=session_id,
                        model_name=candidate,
                        extra_tokens=input_tokens + output_tokens,
                    )
                    yield chunk, candidate

                self._commit_usage(
                    session_id=session_id,
                    model_name=candidate,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                return
            except Exception as exc:
                if isinstance(exc, BudgetExceededError):
                    raise
                last_error = exc
                if has_output:
                    # Partial stream is already visible to users; do not retry a second model.
                    self._commit_usage(
                        session_id=session_id,
                        model_name=candidate,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    raise RuntimeError(f"模型 {candidate} 流式输出中断：{str(exc)}") from exc
                logger.warning(
                    "Model fallback on empty failure: session_id=%s model=%s error=%s",
                    session_id,
                    candidate,
                    str(exc),
                )

        if last_error is None:
            raise RuntimeError("未找到可用模型。")
        raise RuntimeError(f"所有候选模型均失败：{str(last_error)}") from last_error

    def _stream_chunks_with_cross_provider_fallback(
        self,
        *,
        session_id: str,
        provider_name: str,
        requested_model: str | None,
        stage: str,
        messages: list[dict[str, str]],
        provider_options: dict[str, Any] | None,
    ) -> Generator[tuple[str, str], None, None]:
        last_error: Exception | None = None
        for provider in self._resolve_provider_fallbacks(provider_name):
            model_candidates = self._resolve_stage_models(
                provider,
                requested_model if provider == provider_name else None,
                stage,
            )
            try:
                yield from self._stream_chunks_with_fallback(
                    session_id=session_id,
                    provider_name=provider,
                    model_candidates=model_candidates,
                    messages=messages,
                    provider_options=provider_options,
                )
                return
            except Exception as exc:
                if isinstance(exc, BudgetExceededError):
                    raise
                category = self._classify_error(exc)
                last_error = exc
                logger.warning(
                    "Provider fallback failure: session_id=%s provider=%s category=%s error=%s",
                    session_id,
                    provider,
                    category,
                    str(exc),
                )
                if not self._is_retryable_error(category):
                    raise
                continue

        if last_error is None:
            raise RuntimeError("未找到可用模型。")
        raise RuntimeError(f"所有候选 provider 均失败：{str(last_error)}") from last_error

    def _complete_with_fallback(
        self,
        *,
        stage: str,
        session_id: str,
        provider_name: str,
        requested_model: str | None,
        provider_options: dict[str, Any] | None,
        messages: list[dict[str, str]],
    ) -> tuple[str, str]:
        collected_text = ""
        selected_model = ""
        for chunk, model_name in self._stream_chunks_with_cross_provider_fallback(
            session_id=session_id,
            provider_name=provider_name,
            requested_model=requested_model,
            stage=stage,
            messages=messages,
            provider_options=provider_options,
        ):
            selected_model = model_name
            collected_text += chunk
        if not selected_model:
            selected_model = requested_model or self.config.fallback_model_name
        return collected_text, selected_model

    def _classify_need_planning(
        self,
        *,
        session_id: str,
        provider_name: str,
        requested_model: str | None,
        provider_options: dict[str, Any] | None,
        user_message: str,
    ) -> tuple[bool, str, str]:
        if not user_message.strip():
            return False, "heuristic_empty", "heuristic"

        lowered = user_message.lower()
        has_keyword = any(keyword in lowered for keyword in _PLANNING_KEYWORDS)
        question_marks = len(re.findall(r"[?？]", user_message))

        if (
            len(user_message.strip()) <= self.config.planning_short_direct_chars
            and not has_keyword
            and question_marks == 0
        ):
            return False, "heuristic_short_direct", "heuristic"

        if self.config.planning_keyword_routing_enabled and has_keyword:
            return True, "heuristic_keyword", "heuristic"
        if len(user_message) >= self.config.planning_long_input_chars:
            return True, "heuristic_long_input", "heuristic"
        if question_marks >= self.config.planning_multi_question_marks:
            return True, "heuristic_multi_question", "heuristic"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a router for a general assistant.\n"
                    "Return exactly one token:\n"
                    "- PLAN (needs decomposition and iterative execution)\n"
                    "- DIRECT (can be answered in one direct response)"
                ),
            },
            {"role": "user", "content": user_message},
        ]
        intent_text, model_name = self._complete_with_fallback(
            stage="intent",
            session_id=session_id,
            provider_name=provider_name,
            requested_model=requested_model,
            provider_options=provider_options,
            messages=messages,
        )
        normalized = intent_text.strip().upper()
        if "PLAN" in normalized and "DIRECT" not in normalized:
            return True, "intent_model_plan", model_name
        return False, "intent_model_direct", model_name

    def _run_direct_reply(
        self,
        *,
        session_id: str,
        mode: str,
        provider_name: str,
        requested_model: str | None,
        provider_options: dict[str, Any] | None,
    ) -> Generator[tuple[str, str], None, None]:
        messages = self._build_messages(session_id=session_id, mode=mode)
        yield from self._stream_chunks_with_cross_provider_fallback(
            session_id=session_id,
            provider_name=provider_name,
            requested_model=requested_model,
            stage="direct",
            messages=messages,
            provider_options=provider_options,
        )

    def _run_agent_loop(
        self,
        *,
        session_id: str,
        user_message: str,
        provider_name: str,
        requested_model: str | None,
        provider_options: dict[str, Any] | None,
    ) -> Generator[dict[str, Any], None, str]:
        def run_stage(stage: str, run_session_id: str, prompt: str) -> tuple[str, str] | tuple[str, str, dict[str, Any] | None]:
            if stage == "executor":
                text, selected_model, tool_meta = self._run_executor_stage_with_tools(
                    session_id=run_session_id,
                    provider_name=provider_name,
                    requested_model=requested_model,
                    provider_options=provider_options,
                    step_prompt=prompt,
                )
                logger.info(
                    "Agent stage completed: session_id=%s stage=%s model=%s tool=%s",
                    run_session_id,
                    stage,
                    selected_model,
                    tool_meta.get("tool") if isinstance(tool_meta, dict) else None,
                )
                return text, selected_model, tool_meta

            messages = [
                {"role": "system", "content": self._system_prompt("general")},
                {"role": "user", "content": prompt},
            ]
            text, selected_model = self._complete_with_fallback(
                stage=stage,
                session_id=run_session_id,
                provider_name=provider_name,
                requested_model=requested_model,
                provider_options=provider_options,
                messages=messages,
            )
            logger.info(
                "Agent stage completed: session_id=%s stage=%s model=%s",
                run_session_id,
                stage,
                selected_model,
            )
            return text, selected_model

        graph = build_general_agent_graph(run_stage)
        initial_state: GeneralAgentState = {
            "question": user_message,
            "session_id": session_id,
            "task_list": [],
            "current_step_index": 0,
            "review_feedback": "",
            "review_retries": 0,
            "max_review_retries": max(0, self.config.agent_max_review_retries),
            "review_decision": "next_step",
            "response": "",
            "selected_models": {},
            "last_tool_meta": None,
        }

        final_text = ""
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "planner":
                    task_count = len(node_output.get("task_list", []))
                    selected_models = node_output.get("selected_models", {})
                    planner_model = selected_models.get("planner") if isinstance(selected_models, dict) else None
                    event_payload = self._make_status_token_event(
                        delta=f"[planner] 已生成/修订步骤，当前步骤数：{task_count}\n",
                        session_id=session_id,
                        stage="planner",
                        model=planner_model,
                        step_index=int(node_output.get("current_step_index", 0)),
                        total_steps=task_count,
                    )
                    self._record_task_event(session_id, event_payload["payload"])
                    yield event_payload
                elif node_name == "executor":
                    task_list = node_output.get("task_list", [])
                    done_count = sum(1 for item in task_list if item.get("status") == "done")
                    selected_models = node_output.get("selected_models", {})
                    executor_model = selected_models.get("executor") if isinstance(selected_models, dict) else None
                    event_payload = self._make_status_token_event(
                        delta=f"[executor] 已执行步骤：{done_count}\n",
                        session_id=session_id,
                        stage="executor",
                        model=executor_model,
                        step_index=int(node_output.get("current_step_index", 0)),
                        total_steps=len(task_list),
                    )
                    self._record_task_event(session_id, event_payload["payload"])
                    yield event_payload
                    tool_meta = node_output.get("last_tool_meta")
                    if isinstance(tool_meta, dict) and tool_meta.get("tool"):
                        tool_name = str(tool_meta.get("tool", "tool"))
                        tool_status = str(tool_meta.get("status", "unknown"))
                        reason = str(tool_meta.get("reason", "")).strip()
                        tool_input = str(tool_meta.get("input", "")).strip()
                        detail = ""
                        if reason:
                            detail = f" reason={reason}"
                        elif tool_input:
                            detail = f" input={tool_input}"
                        tool_event = self._make_status_token_event(
                            delta=f"[tool] {tool_name} {tool_status}.{detail}\n",
                            session_id=session_id,
                            stage="tool",
                            model=executor_model,
                            reason=str(tool_meta.get("error") or reason or "").strip() or None,
                        )
                        self._record_task_event(session_id, tool_event["payload"])
                        yield tool_event
                        step_index = int(node_output.get("current_step_index", 0))
                        step_task = ""
                        if 0 <= step_index < len(task_list):
                            step_task = str(task_list[step_index].get("task", ""))
                        evidence_item = {
                            "step_index": step_index,
                            "step_task": step_task,
                            "tool": tool_name,
                            "source": tool_meta.get("source"),
                            "status": tool_status,
                            "reason": reason,
                            "model": executor_model,
                            "output_excerpt": tool_meta.get("output_excerpt", ""),
                        }
                        self._record_evidence(session_id, evidence_item)
                        yield {
                            "event": "token",
                            "content_type": "text",
                            "payload": {
                                "delta": "",
                                "meta": {
                                    "kind": "evidence",
                                    "session_id": session_id,
                                    "item": evidence_item,
                                },
                            },
                        }
                elif node_name == "reviewer":
                    feedback = node_output.get("review_feedback", "")
                    retries = node_output.get("review_retries", 0)
                    selected_models = node_output.get("selected_models", {})
                    reviewer_model = selected_models.get("reviewer") if isinstance(selected_models, dict) else None
                    task_list = node_output.get("task_list", [])
                    current_step_index = int(node_output.get("current_step_index", 0))
                    review_event = self._make_status_token_event(
                        delta=f"[reviewer] {feedback} (retry={retries})\n",
                        session_id=session_id,
                        stage="reviewer",
                        model=reviewer_model,
                        retry=int(retries),
                        reason=str(feedback) if feedback else None,
                        step_index=current_step_index,
                        total_steps=len(task_list),
                    )
                    self._record_task_event(session_id, review_event["payload"])
                    yield review_event
                elif node_name == "summarizer":
                    final_text = node_output.get("response", "") or final_text
                    if final_text:
                        yield {
                            "event": "token",
                            "content_type": "text",
                            "payload": {"delta": final_text},
                        }

        return final_text

    def _eda_chat(self, session_id: str, user_message: str) -> str:
        import backend.data_analysis.agent as agent
        from app.api.eda import _last_ai_message

        if agent.llm is None:
            agent.init_llm()

        try:
            state = agent.run_chat(user_message, thread_id=session_id)
            text = _last_ai_message(state)
            if text and text.strip():
                return text
            return "EDA 分析完成，请查看报告或继续提问。"
        except Exception as exc:
            logger.exception("EDA local chat failed: session_id=%s", session_id)
            return f"抱歉，数据分析模块暂时无法响应: {str(exc)}"

    async def _run_sql_analysis(
        self,
        *,
        session_id: str,
        user_message: str,
        provider_options: dict[str, Any],
        sql_app: Any = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        from langchain_core.messages import HumanMessage

        if sql_app is not None:
            graph_app = sql_app
            db_type = "mysql"
            valid_paths: list[str] = []
        else:
            from backend.SQLagent.main import get_sql_graph_app

            graph_app = get_sql_graph_app(db_type="sqlite")
            db_type = "sqlite"
            raw_file_names = provider_options.get("file_name", "")
            if isinstance(raw_file_names, str):
                file_list = [f.strip() for f in raw_file_names.split(",") if f.strip()]
            elif raw_file_names:
                file_list = [str(raw_file_names)]
            else:
                file_list = []

            valid_paths = []
            for file_name in file_list:
                full_path = os.path.join(os.getcwd(), "backend", "dataset", file_name)
                if os.path.exists(full_path):
                    valid_paths.append(full_path)

        config = {"configurable": {"thread_id": session_id}}
        inputs = {
            "messages": [HumanMessage(content=user_message)],
            "db_type": db_type,
            "excel_paths": valid_paths,
            "retry_count": 0,
        }

        full_analysis_text = ""
        try:
            async for chunk in graph_app.astream(inputs, config=config, stream_mode="updates"):
                if "sql_gen" in chunk:
                    sql = chunk["sql_gen"].get("sql_query")
                    if sql:
                        yield {
                            "event": "token",
                            "payload": {"delta": f"\n> **🔍 正在生成 SQL:**\n> ```sql\n> {sql}\n> ```\n"},
                        }

                if "analysis" in chunk:
                    ans = chunk["analysis"].get("analysis", "")
                    if isinstance(ans, str) and ans:
                        full_analysis_text += ans
                        yield {"event": "token", "payload": {"delta": ans}}

                await asyncio.sleep(0)

            final_text = full_analysis_text.strip() or "SQL 分析完成。"
            self._append_session_message(session_id=session_id, role="assistant", content=final_text)
            yield {"event": "final", "payload": {"text": final_text}}
        except Exception as exc:
            yield {"event": "error", "payload": {"message": f"SQL Agent 运行出错：{str(exc)}"}}

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            had_history = self._sessions.pop(session_id, None) is not None
            self._usage.pop(session_id, None)
        self._memory_store.clear_session(session_id)
        return had_history

    async def stream_chat_events(
        self,
        session_id: str,
        mode: str,
        user_message: str,
        provider: str | None = None,
        model: str | None = None,
        provider_options: dict[str, Any] | None = None,
        sql_app: Any = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """按顺序产出聊天事件：token/final/error。"""

        opts = provider_options or {}
        model_provider_options = self._sanitize_provider_options(opts)
        normalized_mode = self._normalize_mode(mode)
        is_eda_mode = bool(opts.get("eda_analysis", False))
        is_sql_mode = bool(opts.get("sql_analysis", False))
        self._hydrate_session(session_id)

        # expert + data-analysis: 走现有 EDA/SQL 分支。
        if normalized_mode == "expert" and is_eda_mode:
            self._append_session_message(session_id=session_id, role="user", content=user_message)
            try:
                eda_text = await asyncio.to_thread(self._eda_chat, session_id, user_message)
                self._append_session_message(session_id=session_id, role="assistant", content=eda_text)
                yield {"event": "token", "payload": {"delta": eda_text}}
                yield {"event": "final", "payload": {"text": eda_text}}
            except Exception as exc:
                yield {"event": "error", "payload": {"message": f"EDA 调用失败：{str(exc)}"}}
            return

        if normalized_mode == "expert" and is_sql_mode:
            self._append_session_message(session_id=session_id, role="user", content=user_message)
            try:
                async for event in self._run_sql_analysis(
                    session_id=session_id,
                    user_message=user_message,
                    provider_options=opts,
                    sql_app=sql_app,
                ):
                    yield event
            except Exception as exc:
                yield {"event": "error", "payload": {"message": f"SQL Agent 运行出错：{str(exc)}"}}
            return

        # general 模式：Manus-like 流程（含 intent -> direct/plan 路由）。
        if normalized_mode == "general":
            provider_name, model_from_alias = self._resolve_general_backend(
                provider=provider,
                model=model,
                provider_options=opts,
            )
            requested_model = model_from_alias or model
            self._append_session_message(session_id=session_id, role="user", content=user_message)

            final_text = ""
            try:
                should_plan, route_reason, route_model = self._classify_need_planning(
                    session_id=session_id,
                    provider_name=provider_name,
                    requested_model=requested_model,
                    provider_options=model_provider_options,
                    user_message=user_message,
                )
                logger.info(
                    "Intent routed: session_id=%s mode=%s should_plan=%s reason=%s model=%s",
                    session_id,
                    normalized_mode,
                    should_plan,
                    route_reason,
                    route_model,
                )
                router_event = self._make_status_token_event(
                    delta=(
                        f"[router] mode={normalized_mode}, decision={'PLAN' if should_plan else 'DIRECT'}, "
                        f"reason={route_reason}\n"
                    ),
                    session_id=session_id,
                    stage="router",
                    model=None if route_model == "heuristic" else route_model,
                    reason=route_reason,
                )
                self._record_task_event(session_id, router_event["payload"])
                yield router_event

                if should_plan:
                    for event in self._run_agent_loop(
                        session_id=session_id,
                        user_message=user_message,
                        provider_name=provider_name,
                        requested_model=requested_model,
                        provider_options=model_provider_options,
                    ):
                        if event.get("event") == "token":
                            payload = event.get("payload", {})
                            meta = payload.get("meta")
                            is_status = isinstance(meta, dict) and meta.get("kind") == "status"
                            if not is_status:
                                final_text += str(payload.get("delta", ""))
                        yield event
                        await asyncio.sleep(0)
                else:
                    selected_model = ""
                    for chunk, model_name in self._run_direct_reply(
                        session_id=session_id,
                        mode=normalized_mode,
                        provider_name=provider_name,
                        requested_model=requested_model,
                        provider_options=model_provider_options,
                    ):
                        selected_model = model_name
                        final_text += chunk
                        yield {"event": "token", "content_type": "text", "payload": {"delta": chunk}}
                        await asyncio.sleep(0)

                    logger.info(
                        "General-direct response finished: session_id=%s model=%s",
                        session_id,
                        selected_model or "unknown",
                    )

                assistant_text = final_text.strip() or "处理完成。"
                self._append_session_message(session_id=session_id, role="assistant", content=assistant_text)
                yield {
                    "event": "final",
                    "content_type": "text",
                    "payload": {"text": assistant_text, "usage": self._get_usage_snapshot(session_id)},
                }
            except BudgetExceededError as exc:
                logger.warning("Budget exceeded: session_id=%s mode=%s error=%s", session_id, normalized_mode, str(exc))
                budget_event = self._make_status_token_event(
                    delta=f"[budget] {str(exc)}\n",
                    session_id=session_id,
                    stage="budget",
                    reason="budget_exceeded",
                )
                self._record_task_event(session_id, budget_event["payload"])
                yield budget_event
                yield {
                    "event": "error",
                    "content_type": "text",
                    "payload": {"message": f"模型调用失败：{str(exc)}", "usage": self._get_usage_snapshot(session_id)},
                }
            except Exception as exc:
                logger.exception("General stream failed: session_id=%s", session_id)
                yield {
                    "event": "error",
                    "content_type": "text",
                    "payload": {"message": f"模型调用失败：{str(exc)}", "usage": self._get_usage_snapshot(session_id)},
                }
            return

        # expert 模式但无分析开关：仅专家直答，不进入 Manus 流程。
        provider_name = self._resolve_provider_name(provider)
        self._append_session_message(session_id=session_id, role="user", content=user_message)
        final_text = ""
        try:
            selected_model = ""
            for chunk, model_name in self._run_direct_reply(
                session_id=session_id,
                mode="expert",
                provider_name=provider_name,
                requested_model=model,
                provider_options=model_provider_options,
            ):
                selected_model = model_name
                final_text += chunk
                yield {"event": "token", "content_type": "text", "payload": {"delta": chunk}}
                await asyncio.sleep(0)
            logger.info(
                "Expert-direct response finished: session_id=%s model=%s",
                session_id,
                selected_model or "unknown",
            )

            assistant_text = final_text.strip() or "处理完成。"
            self._append_session_message(session_id=session_id, role="assistant", content=assistant_text)
            yield {
                "event": "final",
                "content_type": "text",
                "payload": {"text": assistant_text, "usage": self._get_usage_snapshot(session_id)},
            }
        except BudgetExceededError as exc:
            yield self._make_status_token_event(
                delta=f"[budget] {str(exc)}\n",
                session_id=session_id,
                stage="budget",
                reason="budget_exceeded",
            )
            yield {
                "event": "error",
                "content_type": "text",
                "payload": {"message": f"模型调用失败：{str(exc)}", "usage": self._get_usage_snapshot(session_id)},
            }
        except Exception as exc:
            yield {
                "event": "error",
                "content_type": "text",
                "payload": {"message": f"模型调用失败：{str(exc)}", "usage": self._get_usage_snapshot(session_id)},
            }
