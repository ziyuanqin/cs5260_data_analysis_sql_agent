"""聊天核心业务逻辑。

职责：
- 统一维护 session 历史与预算使用
- 在 general/expert 模式间做路由
- 在 expert 模式下运行 LangGraph 的 plan/execute/review/summarize 闭环
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import zipfile
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote_plus, urlparse

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
    "direct": "agent_executor_model_name",
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
_WEBSITE_KEYWORDS = (
    "website",
    "web site",
    "webpage",
    "landing page",
    "网页",
    "网站",
    "前端页面",
)
_SLIDE_KEYWORDS = (
    "slide",
    "slides",
    "ppt",
    "powerpoint",
    "deck",
    "演示",
    "幻灯片",
    "汇报页",
)
_CAPABILITY_TASK_TEMPLATES: dict[str, list[str]] = {
    "website_builder": [
        "Plan the information architecture and page sections",
        "Implement complete HTML/CSS/JS",
        "Deploy and validate the website",
        "Prepare delivery notes and provide final output",
    ],
    "slide_builder": [
        "Define audience and presentation structure (sections and slide count)",
        "Generate slide titles, key points, and speaker notes",
        "Produce previewable HTML slides and downloadable files",
        "Prepare delivery notes and provide final output",
    ],
}

_EXPERT_MODES = {"expert", "analyst"}
_CONTROL_PROVIDER_OPTIONS = {
    "general_model",
    "general_capability",
    "general_web_search",
    "general_web_read",
    "general_file_access",
    "eda_analysis",
    "sql_analysis",
    "file_name",
}
_TOOL_OUTPUT_MAX_CHARS = 5000
_MODEL_EMPTY_OUTPUT_RETRY_LIMIT = 1
_ARTIFACT_WRITE_MAX_CHARS = 400_000


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
        # session_id -> compressed summary text for older turns
        self._session_summaries: dict[str, str] = {}
        # session_id -> {"estimated_tokens": int, "estimated_cost_usd": float}
        self._usage: dict[str, dict[str, float]] = {}
        self._lock = Lock()

    def _hydrate_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            if session_id in self._sessions and session_id in self._session_summaries:
                return
        messages = self._memory_store.load_messages(session_id, limit=self.config.history_max_messages)
        summary_text = self._memory_store.load_summary(session_id)
        with self._lock:
            self._sessions[session_id] = messages[-self.config.history_max_messages :] if messages else []
            self._session_summaries[session_id] = summary_text
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
            add("deepseek_compatible")
            add("huggingface")
        elif primary == "deepseek_compatible":
            add("openai_compatible")
            add("huggingface")
        elif primary == "huggingface":
            add("openai_compatible")
            add("deepseek_compatible")
        elif primary == "local_http":
            add("openai_compatible")
            add("deepseek_compatible")
            add("huggingface")
        else:
            add("openai_compatible")
            add("deepseek_compatible")
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
        if (
            "connection" in text
            or "connect" in text
            or "reset" in text
            or "dns" in text
            or "response ended prematurely" in text
            or "incomplete chunked read" in text
            or "chunkedencodingerror" in text
            or "remote end closed connection" in text
            or "connection aborted" in text
        ):
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

        raw_alias = str((provider_options or {}).get("general_model", "")).strip()
        alias = raw_alias.lower()
        if alias == "openai":
            return self.config.general_openai_provider, self.config.general_openai_model_name
        if alias == "deepseek":
            if self.config.deepseek_api_key:
                return "deepseek_compatible", self.config.general_deepseek_model_name
            return self.config.general_deepseek_provider, self.config.general_deepseek_model_name
        if alias in {"huggingface", "qwen", "hf"}:
            return "huggingface", self.config.huggingface_model_name
        # Allow passing raw model IDs via general_model (routed to Hugging Face router).
        if raw_alias:
            return "huggingface", raw_alias

        return self._resolve_provider_name(None), None

    def _system_prompt(self, mode: str) -> str:
        if mode == "expert":
            return "You are a data analysis expert assistant. Provide structured, actionable answers with key conclusions."
        return "You are an execution-focused AI agent. Provide actionable, verifiable, and practical outputs."

    def _append_session_message(self, session_id: str, role: str, content: str) -> None:
        if not content:
            return

        summary_to_persist: str | None = None
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": role, "content": content})
            max_messages = max(2, self.config.history_max_messages)
            if len(history) > max_messages:
                if self.config.memory_auto_compress_enabled:
                    keep_recent = max(2, min(self.config.memory_keep_recent_messages, max_messages))
                    overflow_count = max(0, len(history) - keep_recent)
                    overflow_chunk = history[:overflow_count]
                    retained = history[overflow_count:]
                    summary = self._merge_summary_text(
                        self._session_summaries.get(session_id, ""),
                        overflow_chunk,
                    )
                    self._session_summaries[session_id] = summary
                    summary_to_persist = summary
                    self._sessions[session_id] = retained
                else:
                    self._sessions[session_id] = history[-max_messages:]
        self._memory_store.append_message(session_id, role, content)
        if summary_to_persist is not None:
            self._memory_store.upsert_summary(session_id, summary_to_persist)

    def _merge_summary_text(self, existing_summary: str, overflow_chunk: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for item in overflow_chunk:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            role_label = "User" if role == "user" else "Assistant"
            content = str(item.get("content") or "").strip().replace("\n", " ")
            if not content:
                continue
            if len(content) > 180:
                content = f"{content[:180]}..."
            lines.append(f"- {role_label}: {content}")
            if len(lines) >= 20:
                break

        if not lines:
            combined = str(existing_summary or "").strip()
        else:
            chunk_text = "Older conversation summary:\n" + "\n".join(lines)
            existing = str(existing_summary or "").strip()
            combined = f"{existing}\n{chunk_text}".strip() if existing else chunk_text

        max_chars = max(600, int(self.config.memory_summary_max_chars))
        if len(combined) > max_chars:
            combined = combined[-max_chars:]
        return combined

    def _build_messages(self, session_id: str, mode: str) -> list[dict[str, str]]:
        with self._lock:
            history = list(self._sessions.get(session_id, []))
            summary_text = str(self._session_summaries.get(session_id, "")).strip()

        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt(mode)}]
        if summary_text:
            messages.append(
                {
                    "role": "system",
                    "content": f"Conversation memory summary (compressed older turns):\n{summary_text}",
                }
            )
        messages.extend(history)
        return messages

    def _sanitize_provider_options(self, provider_options: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(provider_options, dict):
            return None
        cleaned = {k: v for k, v in provider_options.items() if k not in _CONTROL_PROVIDER_OPTIONS}
        return cleaned or None

    def _project_root(self) -> Path:
        return self.config.frontend_dir.parent.resolve()

    def _normalize_session_key(self, session_id: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", str(session_id or "").strip())
        return cleaned or "default"

    def _artifacts_root(self) -> Path:
        root = self._project_root() / "runtime" / "artifacts"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _artifact_session_dir(self, session_id: str) -> Path:
        session_dir = self._artifacts_root() / self._normalize_session_key(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _resolve_artifact_path(self, session_id: str, filename: str) -> Path:
        session_dir = self._artifact_session_dir(session_id)
        candidate = (session_dir / filename).resolve()
        try:
            candidate.relative_to(session_dir.resolve())
        except ValueError as exc:
            raise ValueError("Artifact path is outside session workspace.") from exc
        return candidate

    def _detect_general_capability(self, user_message: str) -> str | None:
        lowered = str(user_message or "").lower()
        if any(keyword in lowered for keyword in _WEBSITE_KEYWORDS):
            return "website_builder"
        if any(keyword in lowered for keyword in _SLIDE_KEYWORDS):
            return "slide_builder"
        return None

    def _capability_task_template(self, capability: str | None) -> list[dict[str, str]]:
        if not capability:
            return []
        raw_tasks = _CAPABILITY_TASK_TEMPLATES.get(capability, [])
        return [{"task": item, "status": "pending", "result": ""} for item in raw_tasks]

    def _task_items_from_tasks(
        self,
        tasks: list[dict[str, Any]],
        *,
        running_index: int | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for idx, task in enumerate(tasks):
            title = str(task.get("task", "")).strip() if isinstance(task, dict) else ""
            if not title:
                continue
            status = "pending"
            if str(task.get("status", "")).lower() == "done":
                status = "done"
            elif running_index is not None and idx == running_index:
                status = "running"
            items.append({"index": idx, "title": title, "status": status})
        return items

    def _write_artifact_file(self, session_id: str, relative_path: str, content: str) -> dict[str, Any]:
        payload = {"path": relative_path, "content": content, "overwrite": True}
        result_text = self._tool_file_write(session_id, json.dumps(payload, ensure_ascii=False))
        parsed = self._extract_json_object(result_text)
        return {
            "name": str(parsed.get("name") or Path(relative_path).name),
            "relative_path": str(parsed.get("relative_path") or relative_path),
            "size_bytes": int(parsed.get("size_bytes") or 0),
            "previewable": bool(parsed.get("previewable", False)),
        }

    def _fallback_website_files(self, user_message: str) -> dict[str, str]:
        title = "AI Agent Project Showcase"
        index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <link rel="stylesheet" href="./style.css" />
</head>
<body>
  <header class="nav">
    <div class="brand">{title}</div>
    <nav>
      <a href="#hero">Home</a>
      <a href="#features">Features</a>
      <a href="#contact">Contact</a>
    </nav>
  </header>
  <main>
    <section id="hero" class="hero">
      <h1>{title}</h1>
      <p>A reusable one-page website template with clear information architecture, hierarchy, and mobile responsiveness.</p>
    </section>
    <section id="features" class="section">
      <h2>Core Sections</h2>
      <ul>
        <li>Navigation bar: quick access to key sections.</li>
        <li>Hero section: communicate core value and call to action.</li>
        <li>Feature section: present capabilities and scenarios.</li>
        <li>Contact section: provide feedback and collaboration channels.</li>
      </ul>
    </section>
    <section id="contact" class="section">
      <h2>Local Run</h2>
      <p>Open <code>index.html</code> directly to preview the page.</p>
    </section>
  </main>
</body>
</html>
"""
        style_css = """* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "PingFang SC", sans-serif;
  color: #1f2937;
  background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}
.nav {
  position: sticky;
  top: 0;
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(6px);
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 20px;
}
.nav a { margin-left: 14px; color: #334155; text-decoration: none; font-weight: 600; }
.hero, .section {
  width: min(900px, 92%);
  margin: 26px auto;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 26px;
}
.hero h1 { margin: 0 0 12px; font-size: 2rem; }
.section h2 { margin-top: 0; }
ul { padding-left: 18px; line-height: 1.75; }
@media (max-width: 640px) {
  .nav { flex-direction: column; align-items: flex-start; gap: 8px; }
  .hero, .section { padding: 18px; }
}
"""
        readme = """# Website Artifacts

## Included files
- `index.html`
- `style.css`

## Local run
Open `index.html` directly in your browser.
"""
        return {"index.html": index_html, "style.css": style_css, "README.md": readme}

    def _fallback_slide_files(self, user_message: str) -> dict[str, str]:
        title = "AI Agent Architecture Review"
        slides_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", "PingFang SC", sans-serif; background: #0f172a; color: #e2e8f0; }}
    .slide {{ min-height: 100vh; display: grid; place-content: center; padding: 48px; border-bottom: 1px solid #334155; }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ margin: 0; line-height: 1.7; max-width: 860px; }}
  </style>
</head>
<body>
  <section class="slide"><div><h1>{title}</h1><p>Goal: present architecture, execution flow, and delivery capability.</p></div></section>
  <section class="slide"><div><h2>System Topology</h2><p>Router -> Planner -> Executor -> Reviewer -> Summarizer.</p></div></section>
  <section class="slide"><div><h2>Capability Demo</h2><p>Automatic Website/Slides generation with artifact download and preview.</p></div></section>
</body>
</html>
"""
        notes = """# Speaker Notes

1. Background and goals
2. Unified routing and state flow
3. Task execution and review loop
4. File delivery and traceability
"""
        return {"slides.html": slides_html, "speaker_notes.md": notes}

    def _generate_capability_files_with_llm(
        self,
        *,
        capability: str,
        session_id: str,
        user_message: str,
        final_text: str,
        provider_name: str,
        requested_model: str | None,
        provider_options: dict[str, Any] | None,
    ) -> dict[str, str]:
        if capability == "website_builder":
            prompt = (
                "Return JSON only with keys: index_html, style_css, readme_md.\n"
                "Generate practical website artifacts according to the user request.\n"
                "Keep HTML and CSS complete and runnable.\n"
                f"User request:\n{user_message}\n\n"
                f"Execution summary:\n{final_text}"
            )
            response_text, _ = self._complete_with_fallback(
                stage="executor",
                session_id=session_id,
                provider_name=provider_name,
                requested_model=requested_model,
                provider_options=provider_options,
                messages=[{"role": "system", "content": self._system_prompt("general")}, {"role": "user", "content": prompt}],
            )
            parsed = self._extract_json_object(response_text)
            index_html = str(parsed.get("index_html", "")).strip()
            style_css = str(parsed.get("style_css", "")).strip()
            readme_md = str(parsed.get("readme_md", "")).strip()
            if index_html and style_css:
                return {"index.html": index_html, "style.css": style_css, "README.md": readme_md or "# Delivery\n"}
            return {}

        if capability == "slide_builder":
            prompt = (
                "Return JSON only with keys: slides_html, speaker_notes_md.\n"
                "Generate a compact deck as HTML slides and speaker notes.\n"
                f"User request:\n{user_message}\n\n"
                f"Execution summary:\n{final_text}"
            )
            response_text, _ = self._complete_with_fallback(
                stage="executor",
                session_id=session_id,
                provider_name=provider_name,
                requested_model=requested_model,
                provider_options=provider_options,
                messages=[{"role": "system", "content": self._system_prompt("general")}, {"role": "user", "content": prompt}],
            )
            parsed = self._extract_json_object(response_text)
            slides_html = str(parsed.get("slides_html", "")).strip()
            notes_md = str(parsed.get("speaker_notes_md", "")).strip()
            if slides_html:
                return {"slides.html": slides_html, "speaker_notes.md": notes_md or "# Notes\n"}
            return {}

        return {}

    def _ensure_capability_artifacts(
        self,
        *,
        capability: str | None,
        session_id: str,
        user_message: str,
        final_text: str,
        provider_name: str,
        requested_model: str | None,
        provider_options: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if capability not in {"website_builder", "slide_builder"}:
            return []
        existing = self.list_artifacts(session_id)
        if existing:
            return []

        files = self._generate_capability_files_with_llm(
            capability=capability,
            session_id=session_id,
            user_message=user_message,
            final_text=final_text,
            provider_name=provider_name,
            requested_model=requested_model,
            provider_options=provider_options,
        )
        if not files:
            files = (
                self._fallback_website_files(user_message)
                if capability == "website_builder"
                else self._fallback_slide_files(user_message)
            )

        created: list[dict[str, Any]] = []
        for relative_path, content in files.items():
            if not isinstance(content, str) or not content.strip():
                continue
            created.append(self._write_artifact_file(session_id, relative_path, content))
        return created

    def list_artifacts(self, session_id: str) -> list[dict[str, Any]]:
        session_dir = self._artifact_session_dir(session_id)
        if not session_dir.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(session_dir.rglob("*"), key=lambda p: str(p).lower()):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".zip":
                continue
            previewable = self._is_previewable_artifact(path)
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            relative_path = str(path.relative_to(session_dir).as_posix())
            items.append(
                {
                    "name": path.name,
                    "relative_path": relative_path,
                    "size_bytes": int(path.stat().st_size),
                    "updated_at": mtime,
                    "previewable": previewable,
                }
            )
        return items

    def resolve_artifact_file(self, session_id: str, artifact_name: str) -> Path:
        if not artifact_name or not artifact_name.strip():
            raise FileNotFoundError("Artifact name is empty.")
        target = self._resolve_artifact_path(session_id, artifact_name.strip())
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_name}")
        return target

    def build_artifact_bundle(self, session_id: str) -> Path:
        session_dir = self._artifact_session_dir(session_id)
        artifacts = self.list_artifacts(session_id)
        bundle_name = f"{self._normalize_session_key(session_id)}_artifacts.zip"
        bundle_path = self._resolve_artifact_path(session_id, bundle_name)
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in artifacts:
                relative_path = str(item.get("relative_path") or item.get("name") or "").strip()
                if not relative_path:
                    continue
                file_path = self.resolve_artifact_file(session_id, relative_path)
                zf.write(file_path, arcname=relative_path)
        return bundle_path

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            return {}
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end < start:
            return {}
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
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

    def _is_previewable_artifact(self, path: Path) -> bool:
        return path.suffix.lower() in {".html", ".htm", ".md", ".txt", ".json", ".css", ".js"}

    def _tool_file_write(self, session_id: str, raw_payload: str) -> str:
        if not raw_payload or not raw_payload.strip():
            raise ValueError("file_write requires JSON input with path/content.")

        payload = self._extract_json_object(raw_payload)
        if not payload:
            raise ValueError("file_write expects JSON object input.")

        raw_path = str(payload.get("path") or payload.get("file") or "").strip()
        if not raw_path:
            raise ValueError("file_write input missing 'path'.")
        if len(raw_path) > 200:
            raise ValueError("file_write path is too long.")

        normalized_path = raw_path.replace("\\", "/").lstrip("/")
        path_parts = [part for part in normalized_path.split("/") if part and part != "."]
        if not path_parts:
            raise ValueError("file_write path is empty after normalization.")
        if any(part == ".." for part in path_parts):
            raise ValueError("file_write path cannot escape artifact workspace.")

        relative_path = "/".join(path_parts)
        content = payload.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if len(content) > _ARTIFACT_WRITE_MAX_CHARS:
            raise ValueError(f"file_write content too large (>{_ARTIFACT_WRITE_MAX_CHARS} chars).")

        overwrite = bool(payload.get("overwrite", True))
        target = self._resolve_artifact_path(session_id, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise ValueError(f"file_write target already exists: {relative_path}")
        target.write_text(content, encoding="utf-8")

        result = {
            "ok": True,
            "name": target.name,
            "relative_path": str(Path(relative_path)),
            "size_bytes": int(target.stat().st_size),
            "previewable": self._is_previewable_artifact(target),
        }
        return json.dumps(result, ensure_ascii=False)

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

    def _tool_web_search(self, raw_query: str) -> str:
        if not raw_query or not raw_query.strip():
            raise ValueError("web_search requires a non-empty query.")

        query = raw_query.strip()
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=min(self.config.request_timeout, 20),
            headers={"User-Agent": "CS5260-Agent/1.0"},
        )
        response.raise_for_status()
        data = response.json() if response.content else {}
        if not isinstance(data, dict):
            data = {}

        results: list[tuple[str, str]] = []
        abstract_url = str(data.get("AbstractURL", "")).strip()
        abstract_text = str(data.get("AbstractText", "")).strip()
        if abstract_url:
            title = abstract_text or "DuckDuckGo abstract"
            results.append((title, abstract_url))

        for item in data.get("RelatedTopics", []) if isinstance(data.get("RelatedTopics"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("FirstURL"), str) and isinstance(item.get("Text"), str):
                results.append((item["Text"], item["FirstURL"]))
            topics = item.get("Topics") if isinstance(item, dict) else None
            if isinstance(topics, list):
                for sub in topics:
                    if isinstance(sub, dict) and isinstance(sub.get("FirstURL"), str) and isinstance(sub.get("Text"), str):
                        results.append((sub["Text"], sub["FirstURL"]))
            if len(results) >= 5:
                break

        if not results:
            fallback = f"https://duckduckgo.com/?q={quote_plus(query)}"
            results.append((f"Search results for {query}", fallback))

        lines = [f"{idx}. {title} | {url}" for idx, (title, url) in enumerate(results[:5], start=1)]
        return self._shorten("\n".join(lines))

    def _run_general_tool(self, session_id: str, tool_name: str, tool_input: str) -> str:
        normalized = tool_name.strip().lower()
        if normalized == "file_read":
            return self._tool_file_read(tool_input)
        if normalized == "web_read":
            return self._tool_web_read(tool_input)
        if normalized == "web_search":
            return self._tool_web_search(tool_input)
        if normalized == "file_write":
            return self._tool_file_write(session_id, tool_input)
        raise ValueError(f"Unsupported tool: {tool_name}")

    def _run_executor_stage_with_tools(
        self,
        *,
        session_id: str,
        provider_name: str,
        requested_model: str | None,
        provider_options: dict[str, Any] | None,
        step_prompt: str,
        allow_file_access: bool,
        allow_web_read: bool,
        allow_web_search: bool,
        allow_file_write: bool,
    ) -> tuple[str, str, dict[str, Any] | None]:
        allowed_tools: list[str] = []
        tool_lines: list[str] = []
        if allow_file_access:
            allowed_tools.append("file_read")
            tool_lines.append("file_read: read local text file within project workspace.")
        if allow_web_read:
            allowed_tools.append("web_read")
            tool_lines.append("web_read: fetch and read a public web page.")
        if allow_web_search:
            allowed_tools.append("web_search")
            tool_lines.append("web_search: search the web and return top relevant links.")
        if allow_file_write:
            allowed_tools.append("file_write")
            tool_lines.append(
                "file_write: write artifact files under runtime/artifacts/<session_id>/ with JSON input "
                '{"path":"<name.ext>","content":"<text>","overwrite":true}.'
            )

        allowed_tool_text = "|".join(allowed_tools)
        tool_actions = (
            f'{{"action":"TOOL","tool":"{allowed_tool_text}","input":"<tool input>","reason":"<why needed>"}}'
            if allowed_tools
            else ""
        )

        router_prompt = (
            "You are an executor with optional tool usage.\n"
            + (
                "Available tools:\n"
                + "\n".join(f"{idx + 1}) {line}" for idx, line in enumerate(tool_lines))
                + "\n"
                if allowed_tools
                else "No tools enabled for this request.\n"
            )
            + "Return JSON object only.\n"
            + (
                "Use one of these forms:\n"
                '{"action":"RESPOND","response":"<final step result>"}\n'
                + tool_actions
                + "\n"
                if allowed_tools
                else 'Use this form only:\n{"action":"RESPOND","response":"<final step result>"}\n'
            )
            + "No extra text.\n"
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
        if tool_name not in allowed_tools:
            response = str(decision.get("response", "")).strip()
            fallback_text = response or f"Tool '{tool_name or 'unknown'}' is disabled for this request."
            return fallback_text, selected_model, None

        tool_meta: dict[str, Any] = {
            "tool": tool_name,
            "input": tool_input[:200],
            "reason": tool_reason[:200],
        }
        try:
            tool_output = self._run_general_tool(session_id, tool_name, tool_input)
            tool_meta["status"] = "ok"
        except Exception as exc:
            tool_output = f"Tool execution failed: {str(exc)}"
            tool_meta["status"] = "error"
            tool_meta["error"] = str(exc)[:200]
        tool_meta["output_excerpt"] = self._shorten(tool_output, max_chars=800)
        if tool_name == "web_read":
            tool_meta["source"] = tool_input
        elif tool_name == "web_search":
            tool_meta["source"] = f"https://duckduckgo.com/?q={quote_plus(tool_input)}"
        elif tool_name == "file_read":
            try:
                tool_meta["source"] = str(self._resolve_read_path(tool_input))
            except Exception:
                tool_meta["source"] = tool_input
        elif tool_name == "file_write":
            write_meta = self._extract_json_object(tool_output)
            if write_meta:
                tool_meta["artifact"] = {
                    "name": write_meta.get("name"),
                    "relative_path": write_meta.get("relative_path"),
                    "size_bytes": write_meta.get("size_bytes"),
                    "previewable": bool(write_meta.get("previewable", False)),
                }
                tool_meta["source"] = str(write_meta.get("relative_path") or write_meta.get("name") or "")

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
                f"Session token budget reached ({self.config.session_token_budget}). Please reset the session or shorten the request."
            )
        if projected_cost > self.config.session_cost_budget_usd:
            raise BudgetExceededError(
                f"Session cost budget reached (${self.config.session_cost_budget_usd:.2f}). Please reset the session or shorten the request."
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
        task_items: list[dict[str, Any]] | None = None,
        capability: str | None = None,
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
        if task_items:
            payload["meta"]["task_items"] = task_items
        if capability:
            payload["meta"]["capability"] = capability
        return {"event": "token", "content_type": "text", "payload": payload}

    def _make_progress_token_event(
        self,
        *,
        delta: str,
        session_id: str,
        stage: str,
        model: str | None = None,
        step_index: int | None = None,
        total_steps: int | None = None,
        task_items: list[dict[str, Any]] | None = None,
        capability: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "delta": delta,
            "meta": {
                "kind": "progress",
                "session_id": session_id,
                "stage": stage,
                "usage": self._get_usage_snapshot(session_id),
            },
        }
        if model:
            payload["meta"]["model"] = model
        if step_index is not None:
            payload["meta"]["step_index"] = step_index
        if total_steps is not None:
            payload["meta"]["total_steps"] = total_steps
        if task_items:
            payload["meta"]["task_items"] = task_items
        if capability:
            payload["meta"]["capability"] = capability
        return {"event": "token", "content_type": "text", "payload": payload}

    def _make_artifact_token_event(self, *, session_id: str, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": "token",
            "content_type": "text",
            "payload": {
                "delta": "",
                "meta": {
                    "kind": "artifact",
                    "session_id": session_id,
                    "item": item,
                },
            },
        }

    def _resolve_stage_models(self, provider_name: str, requested_model: str | None, stage: str) -> list[str]:
        if requested_model:
            normalized_requested = requested_model.strip()
            if normalized_requested:
                # Respect explicit routing from caller (e.g. General model selector).
                # Also append safe fallbacks to prevent hard failures when a selected
                # model is temporarily unavailable or lacks access permission.
                candidates = [normalized_requested]
                if provider_name == "openai_compatible":
                    for extra in (
                        self.config.general_openai_model_name,
                        self.config.openai_model_name,
                        self.config.fallback_model_name,
                    ):
                        normalized_extra = str(extra or "").strip()
                        if normalized_extra and normalized_extra not in candidates:
                            candidates.append(normalized_extra)
                if provider_name == "huggingface":
                    for extra in (
                        self.config.huggingface_model_name,
                        self.config.fallback_model_name,
                    ):
                        normalized_extra = str(extra or "").strip()
                        if normalized_extra and normalized_extra not in candidates:
                            candidates.append(normalized_extra)
                return candidates

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
        elif provider_name == "deepseek_compatible":
            add(self.config.general_deepseek_model_name)
            add(self.config.fallback_model_name)
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
        if provider_name == "deepseek_compatible":
            return [self.config.general_deepseek_model_name]
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
        request_timeout = self.config.request_timeout
        if provider_name == "huggingface":
            # Keep HF router requests responsive in general-mode multi-model fallback.
            # Without this cap, one unavailable model can block the UI for a long time.
            request_timeout = min(request_timeout, 25)

        for candidate in model_candidates:
            for attempt in range(_MODEL_EMPTY_OUTPUT_RETRY_LIMIT + 1):
                has_output = False
                output_tokens = 0
                try:
                    self._assert_budget(session_id=session_id, model_name=candidate, extra_tokens=input_tokens)
                    stream = provider_impl.stream_chat(
                        model=candidate,
                        messages=messages,
                        timeout=request_timeout,
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
                        raise RuntimeError(f"Streaming output interrupted for model {candidate}: {str(exc)}") from exc

                    category = self._classify_error(exc)
                    can_retry_same_model = (
                        attempt < _MODEL_EMPTY_OUTPUT_RETRY_LIMIT and self._is_retryable_error(category)
                    )
                    logger.warning(
                        "Model empty-output failure: session_id=%s model=%s attempt=%s category=%s retry=%s error=%s",
                        session_id,
                        candidate,
                        attempt + 1,
                        category,
                        can_retry_same_model,
                        str(exc),
                    )
                    if can_retry_same_model:
                        continue
                    break

        if last_error is None:
            raise RuntimeError("No available model found.")
        raise RuntimeError(f"All candidate models failed: {str(last_error)}") from last_error

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
        provider_candidates = [provider_name] if requested_model else self._resolve_provider_fallbacks(provider_name)
        for provider in provider_candidates:
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
            raise RuntimeError("No available model found.")
        raise RuntimeError(f"All candidate providers failed: {str(last_error)}") from last_error

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
        allow_file_access: bool,
        allow_web_read: bool,
        allow_web_search: bool,
        allow_file_write: bool,
        capability: str | None = None,
        seeded_tasks: list[dict[str, str]] | None = None,
    ) -> Generator[dict[str, Any], None, str]:
        def run_stage(stage: str, run_session_id: str, prompt: str) -> tuple[str, str] | tuple[str, str, dict[str, Any] | None]:
            if stage == "executor":
                runtime_prompt = prompt
                if capability == "website_builder":
                    runtime_prompt = (
                        f"{prompt}\n\n"
                        "Execution policy for website_builder:\n"
                        "- Prefer creating real artifacts with file_write.\n"
                        "- Generate at least index.html and a delivery note (README.md).\n"
                        "- Keep paths relative, for example: index.html, styles.css, app.js, README.md."
                    )
                elif capability == "slide_builder":
                    runtime_prompt = (
                        f"{prompt}\n\n"
                        "Execution policy for slide_builder:\n"
                        "- Prefer creating real artifacts with file_write.\n"
                        "- Generate slides.html and speaker_notes.md at minimum.\n"
                        "- Keep paths relative, for example: slides.html, speaker_notes.md, assets/theme.css."
                    )
                text, selected_model, tool_meta = self._run_executor_stage_with_tools(
                    session_id=run_session_id,
                    provider_name=provider_name,
                    requested_model=requested_model,
                    provider_options=provider_options,
                    step_prompt=runtime_prompt,
                    allow_file_access=allow_file_access,
                    allow_web_read=allow_web_read,
                    allow_web_search=allow_web_search,
                    allow_file_write=allow_file_write,
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
        initial_tasks = [dict(item) for item in (seeded_tasks or []) if isinstance(item, dict)]
        initial_state: GeneralAgentState = {
            "question": user_message,
            "session_id": session_id,
            "task_list": initial_tasks,
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

        def compact_text(value: Any, max_chars: int = 120) -> str:
            text = str(value or "").strip()
            if not text:
                return ""
            one_line = " ".join(text.split())
            return self._shorten(one_line, max_chars=max_chars)

        def first_pending_index(tasks: list[dict[str, Any]]) -> int | None:
            for index, item in enumerate(tasks):
                if str(item.get("status", "")).lower() != "done":
                    return index
            return None

        def sanitize_delivery_text(text: Any) -> str:
            raw = str(text or "")
            if capability not in {"website_builder", "slide_builder"}:
                return raw.strip()
            # Hide large inline code blocks in capability delivery replies.
            replaced = re.sub(
                r"```[\s\S]*?```",
                "(Code content has been saved to files. You can preview or download it from the file cards below.)",
                raw,
            )
            replaced = re.sub(r"\n{3,}", "\n\n", replaced).strip()
            return replaced or "Task completed. Previewable and downloadable files have been generated."

        def summarize_tool_meta(tool_meta: dict[str, Any] | None) -> str:
            if not isinstance(tool_meta, dict):
                return ""
            tool_name = str(tool_meta.get("tool") or "").strip().lower()
            if not tool_name:
                return ""

            status = str(tool_meta.get("status") or "").strip().lower()
            if status and status != "ok":
                error_text = compact_text(tool_meta.get("error", ""), max_chars=180)
                if error_text:
                    return f"Tool {tool_name} failed: {error_text}"
                return f"Tool {tool_name} failed"

            if tool_name == "file_write":
                artifact_meta = tool_meta.get("artifact")
                if isinstance(artifact_meta, dict):
                    relative_path = str(
                        artifact_meta.get("relative_path") or artifact_meta.get("name") or ""
                    ).strip()
                    if relative_path:
                        return f"Generated file: {relative_path}"
                source = str(tool_meta.get("source") or "").strip()
                if source:
                    return f"Generated file: {source}"
                return "File write completed"

            if tool_name == "file_read":
                source = str(tool_meta.get("source") or "").strip()
                return f"Read file: {source}" if source else "File read completed"

            if tool_name == "web_search":
                query = str(tool_meta.get("input") or "").strip()
                return f"Web search completed: {query}" if query else "Web search completed"

            if tool_name == "web_read":
                source = str(tool_meta.get("source") or tool_meta.get("input") or "").strip()
                return f"Read web page: {source}" if source else "Web page read completed"

            return f"Tool call completed: {tool_name}"

        def summarize_step_result_text(raw_result: Any) -> str:
            text = str(raw_result or "").strip()
            if not text:
                return ""
            if text.lower() == "step executed.":
                return "Step executed."

            path_hits = re.findall(r'([A-Za-z0-9._-]+\.(?:html|css|js|md|json|txt|csv|py))', text, flags=re.IGNORECASE)
            unique_paths: list[str] = []
            for path in path_hits:
                normalized = str(path).strip()
                if normalized and normalized not in unique_paths:
                    unique_paths.append(normalized)
            if unique_paths:
                return f"Generated files: {', '.join(unique_paths[:3])}"

            lower = text.lower()
            if (
                "<!doctype html" in lower
                or "<html" in lower
                or "```" in text
                or len(text) > 260
            ):
                return "Step executed (long output was converted to file deliverables)."

            return compact_text(text, max_chars=180)

        generated_artifact_names: set[str] = set()

        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "planner":
                    task_list = node_output.get("task_list", [])
                    task_count = len(task_list)
                    running_index = int(node_output.get("current_step_index", 0)) if task_count else None
                    task_items = self._task_items_from_tasks(task_list, running_index=running_index)
                    selected_models = node_output.get("selected_models", {})
                    planner_model = selected_models.get("planner") if isinstance(selected_models, dict) else None
                    task_preview_items: list[str] = []
                    for index, item in enumerate(task_list, start=1):
                        task_text = compact_text(item.get("task", ""), max_chars=60) if isinstance(item, dict) else ""
                        if task_text:
                            task_preview_items.append(f"{index}. {task_text}")
                    task_preview = ""
                    if task_preview_items:
                        task_preview = "\n" + "\n".join(task_preview_items)
                    event_payload = self._make_status_token_event(
                        delta=f"[planner] Planned/revised steps. Current step count: {task_count}{task_preview}\n",
                        session_id=session_id,
                        stage="planner",
                        model=planner_model,
                        step_index=int(node_output.get("current_step_index", 0)),
                        total_steps=task_count,
                        task_items=task_items,
                        capability=capability,
                    )
                    self._record_task_event(session_id, event_payload["payload"])
                    yield event_payload
                    progress_preview = task_preview_items[0] if task_preview_items else "Generating executable steps"
                    yield self._make_progress_token_event(
                        delta=f"Planner: {progress_preview}\n",
                        session_id=session_id,
                        stage="planner",
                        model=planner_model,
                        step_index=int(node_output.get("current_step_index", 0)),
                        total_steps=task_count,
                        task_items=task_items,
                        capability=capability,
                    )
                elif node_name == "executor":
                    task_list = node_output.get("task_list", [])
                    done_count = sum(1 for item in task_list if item.get("status") == "done")
                    selected_models = node_output.get("selected_models", {})
                    executor_model = selected_models.get("executor") if isinstance(selected_models, dict) else None
                    current_step_index = int(node_output.get("current_step_index", 0))
                    next_pending = first_pending_index(task_list)
                    task_items = self._task_items_from_tasks(task_list, running_index=next_pending)
                    tool_meta = node_output.get("last_tool_meta")
                    tool_result_summary = summarize_tool_meta(tool_meta if isinstance(tool_meta, dict) else None)
                    current_step_task = ""
                    current_step_result = ""
                    if 0 <= current_step_index < len(task_list):
                        current_item = task_list[current_step_index] or {}
                        if isinstance(current_item, dict):
                            current_step_task = compact_text(current_item.get("task", ""), max_chars=90)
                            current_step_result = summarize_step_result_text(current_item.get("result", ""))
                    if tool_result_summary:
                        current_step_result = tool_result_summary
                    executor_lines = [f"[executor] Executed steps: {done_count}"]
                    if current_step_task:
                        executor_lines.append(f"Current step: {current_step_task}")
                    if current_step_result:
                        executor_lines.append(f"Result: {current_step_result}")
                    event_payload = self._make_status_token_event(
                        delta="\n".join(executor_lines) + "\n",
                        session_id=session_id,
                        stage="executor",
                        model=executor_model,
                        step_index=current_step_index,
                        total_steps=len(task_list),
                        task_items=task_items,
                        capability=capability,
                    )
                    self._record_task_event(session_id, event_payload["payload"])
                    yield event_payload
                    executor_progress = current_step_task or "Execute step"
                    if current_step_result:
                        executor_progress = f"{executor_progress} -> {compact_text(current_step_result, max_chars=140)}"
                    yield self._make_progress_token_event(
                        delta=f"Executor: {executor_progress}\n",
                        session_id=session_id,
                        stage="executor",
                        model=executor_model,
                        step_index=current_step_index,
                        total_steps=len(task_list),
                        task_items=task_items,
                        capability=capability,
                    )
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
                            task_items=task_items,
                            capability=capability,
                        )
                        self._record_task_event(session_id, tool_event["payload"])
                        yield tool_event
                        yield self._make_progress_token_event(
                            delta=f"Tool: {tool_name} ({tool_status})\n",
                            session_id=session_id,
                            stage="tool",
                            model=executor_model,
                            step_index=current_step_index,
                            total_steps=len(task_list),
                            task_items=task_items,
                            capability=capability,
                        )
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
                        artifact_meta = tool_meta.get("artifact")
                        if isinstance(artifact_meta, dict):
                            artifact_item = {
                                "name": str(artifact_meta.get("name") or "").strip(),
                                "relative_path": str(artifact_meta.get("relative_path") or "").strip(),
                                "size_bytes": int(artifact_meta.get("size_bytes") or 0),
                                "previewable": bool(artifact_meta.get("previewable", False)),
                            }
                            if artifact_item["name"]:
                                generated_artifact_names.add(artifact_item["name"])
                                yield self._make_artifact_token_event(session_id=session_id, item=artifact_item)
                elif node_name == "reviewer":
                    feedback = node_output.get("review_feedback", "")
                    retries = node_output.get("review_retries", 0)
                    selected_models = node_output.get("selected_models", {})
                    reviewer_model = selected_models.get("reviewer") if isinstance(selected_models, dict) else None
                    task_list = node_output.get("task_list", [])
                    current_step_index = int(node_output.get("current_step_index", 0))
                    running_index = first_pending_index(task_list)
                    task_items = self._task_items_from_tasks(task_list, running_index=running_index)
                    review_decision = compact_text(node_output.get("review_decision", ""), max_chars=30)
                    current_step_task = ""
                    if 0 <= current_step_index < len(task_list):
                        current_item = task_list[current_step_index] or {}
                        if isinstance(current_item, dict):
                            current_step_task = compact_text(current_item.get("task", ""), max_chars=90)
                    feedback_text = compact_text(feedback, max_chars=140)
                    pass_like = feedback_text.upper().startswith("PASS")
                    if pass_like:
                        if current_step_task:
                            review_message = f"PASS: Step \"{current_step_task}\" passed review and can move to the next step."
                        else:
                            review_message = "PASS: This step passed review and can move to the next step."
                    else:
                        review_parts = [feedback_text] if feedback_text else []
                        if current_step_task:
                            review_parts.append(f"Step: {current_step_task}")
                        if review_decision:
                            review_parts.append(f"Decision: {review_decision}")
                        review_message = " | ".join(review_parts) if review_parts else "Review completed"
                    review_event = self._make_status_token_event(
                        delta=f"[reviewer] {review_message} (retry={retries})\n",
                        session_id=session_id,
                        stage="reviewer",
                        model=reviewer_model,
                        retry=int(retries),
                        reason=str(feedback) if feedback else None,
                        step_index=current_step_index,
                        total_steps=len(task_list),
                        task_items=task_items,
                        capability=capability,
                    )
                    self._record_task_event(session_id, review_event["payload"])
                    yield review_event
                    yield self._make_progress_token_event(
                        delta=f"Reviewer: {review_message}\n",
                        session_id=session_id,
                        stage="reviewer",
                        model=reviewer_model,
                        step_index=current_step_index,
                        total_steps=len(task_list),
                        task_items=task_items,
                        capability=capability,
                    )
                elif node_name == "summarizer":
                    final_text = sanitize_delivery_text(node_output.get("response", "") or final_text)
                    if final_text:
                        yield {
                            "event": "token",
                            "content_type": "text",
                            "payload": {"delta": final_text},
                        }
                        yield self._make_progress_token_event(
                            delta="Summarizer: Final summary completed.\n",
                            session_id=session_id,
                            stage="summarizer",
                            model=(node_output.get("selected_models", {}) or {}).get("summarizer"),
                            task_items=self._task_items_from_tasks(node_output.get("task_list", [])),
                            capability=capability,
                        )

        if generated_artifact_names:
            yield self._make_progress_token_event(
                delta=f"Artifacts ready: {', '.join(sorted(generated_artifact_names))}\n",
                session_id=session_id,
                stage="artifact",
                capability=capability,
            )

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
            return "EDA analysis completed. Please review the report or continue asking questions."
        except Exception as exc:
            logger.exception("EDA local chat failed: session_id=%s", session_id)
            return f"Sorry, the data analysis module is temporarily unavailable: {str(exc)}"

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
                    if sql and sql.strip().upper() != "SKIP":
                        yield {
                            "event": "token",
                            "payload": {"delta": f"\n> **🔍 Generating SQL:**\n> ```sql\n> {sql}\n> ```\n"},
                        }

                if "analysis" in chunk:
                    ans = chunk["analysis"].get("analysis", "")
                    if isinstance(ans, str) and ans:
                        full_analysis_text += ans
                        yield {"event": "token", "payload": {"delta": ans}}

                await asyncio.sleep(0)

            final_text = full_analysis_text.strip() or "SQL analysis finished"
            self._append_session_message(session_id=session_id, role="assistant", content=final_text)
            yield {"event": "final", "payload": {"text": final_text}}
        except Exception as exc:
            yield {"event": "error", "payload": {"message": f"SQL Agent error：{str(exc)}"}}

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            had_history = self._sessions.pop(session_id, None) is not None
            self._session_summaries.pop(session_id, None)
            self._usage.pop(session_id, None)
        self._memory_store.clear_session(session_id)
        return had_history

    def rewrite_last_user_turn(self, session_id: str) -> dict[str, int | bool]:
        """Remove the latest user turn and everything after it for one session."""
        with self._lock:
            session_loaded = session_id in self._sessions
        if not session_loaded:
            self._hydrate_session(session_id)

        with self._lock:
            history = list(self._sessions.get(session_id, []))
            last_user_index = -1
            for idx in range(len(history) - 1, -1, -1):
                if str(history[idx].get("role")) == "user":
                    last_user_index = idx
                    break

            if last_user_index < 0:
                return {"rewritten": False, "removed_messages": 0, "remaining_messages": len(history)}

            retained = history[:last_user_index]
            removed = len(history) - len(retained)
            self._sessions[session_id] = retained
            self._session_summaries[session_id] = ""

        self._memory_store.replace_messages(session_id, retained)
        self._memory_store.upsert_summary(session_id, "")
        return {
            "rewritten": True,
            "removed_messages": removed,
            "remaining_messages": len(retained),
        }

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
        enable_general_file_access = bool(opts.get("general_file_access", False))
        enable_general_web_read = bool(opts.get("general_web_read", False))
        enable_general_web_search = bool(opts.get("general_web_search", False))
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
                yield {"event": "error", "payload": {"message": f"EDA call failed：{str(exc)}"}}
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
                yield {"event": "error", "payload": {"message": f"SQL Agent running error：{str(exc)}"}}
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
                requested_capability = str(opts.get("general_capability", "")).strip().lower()
                capability = requested_capability if requested_capability in {"website_builder", "slide_builder"} else ""
                if not capability:
                    capability = self._detect_general_capability(user_message)
                seeded_task_template = self._capability_task_template(capability)
                if capability:
                    should_plan = True
                    route_reason = f"capability_{capability}"
                    route_model = "capability_router"
                else:
                    should_plan, route_reason, route_model = self._classify_need_planning(
                        session_id=session_id,
                        provider_name=provider_name,
                        requested_model=requested_model,
                        provider_options=model_provider_options,
                        user_message=user_message,
                    )
                logger.info(
                    "Intent routed: session_id=%s mode=%s should_plan=%s reason=%s model=%s capability=%s",
                    session_id,
                    normalized_mode,
                    should_plan,
                    route_reason,
                    route_model,
                    capability,
                )
                router_task_items = self._task_items_from_tasks(seeded_task_template) if seeded_task_template else []
                router_event = self._make_status_token_event(
                    delta=(
                        f"[router] mode={normalized_mode}, decision={'PLAN' if should_plan else 'DIRECT'}, "
                        f"reason={route_reason}"
                        + (f", capability={capability}" if capability else "")
                        + "\n"
                    ),
                    session_id=session_id,
                    stage="router",
                    model=None if route_model == "heuristic" else route_model,
                    reason=route_reason,
                    task_items=router_task_items,
                    capability=capability,
                )
                self._record_task_event(session_id, router_event["payload"])
                yield router_event
                yield self._make_progress_token_event(
                    delta=(
                        f"Router: {'Step by step execution of the task' if should_plan else 'Direct answer'}"
                        + (f"（Ability: {capability}）" if capability else "")
                        + "\n"
                    ),
                    session_id=session_id,
                    stage="router",
                    model=None if route_model == "heuristic" else route_model,
                    task_items=router_task_items,
                    capability=capability,
                )

                if should_plan:
                    for event in self._run_agent_loop(
                        session_id=session_id,
                        user_message=user_message,
                        provider_name=provider_name,
                        requested_model=requested_model,
                        provider_options=model_provider_options,
                        allow_file_access=enable_general_file_access,
                        allow_web_read=enable_general_web_read,
                        allow_web_search=enable_general_web_search,
                        allow_file_write=bool(capability),
                        capability=capability,
                        seeded_tasks=seeded_task_template,
                    ):
                        if event.get("event") == "token":
                            payload = event.get("payload", {})
                            meta = payload.get("meta")
                            kind = meta.get("kind") if isinstance(meta, dict) else ""
                            if kind not in {"status", "progress", "evidence", "artifact"}:
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

                created_artifacts = self._ensure_capability_artifacts(
                    capability=capability,
                    session_id=session_id,
                    user_message=user_message,
                    final_text=final_text,
                    provider_name=provider_name,
                    requested_model=requested_model,
                    provider_options=model_provider_options,
                )
                for item in created_artifacts:
                    yield self._make_artifact_token_event(session_id=session_id, item=item)
                if created_artifacts:
                    created_names = ", ".join(str(item.get("relative_path") or item.get("name")) for item in created_artifacts)
                    yield self._make_progress_token_event(
                        delta=f"Artifacts generated: {created_names}\n",
                        session_id=session_id,
                        stage="artifact",
                        capability=capability,
                    )

                assistant_text = final_text.strip() or "Processing completed."
                artifacts = self.list_artifacts(session_id)
                self._append_session_message(session_id=session_id, role="assistant", content=assistant_text)
                yield {
                    "event": "final",
                    "content_type": "text",
                    "payload": {
                        "text": assistant_text,
                        "usage": self._get_usage_snapshot(session_id),
                        "artifacts": artifacts,
                        "capability": capability,
                    },
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
                    "payload": {"message": f"Model call failed：{str(exc)}", "usage": self._get_usage_snapshot(session_id)},
                }
            except Exception as exc:
                logger.exception("General stream failed: session_id=%s", session_id)
                yield {
                    "event": "error",
                    "content_type": "text",
                    "payload": {"message": f"Model call failed：{str(exc)}", "usage": self._get_usage_snapshot(session_id)},
                }
            return

        # expert 模式但无分析开关：仅专家直答，不进入 Manus 流程。
        # Keep expert decoupled from general-model routing:
        # - If caller specifies provider/model, respect it.
        # - Otherwise prefer OpenAI-compatible backend (independent API quota).
        if provider:
            provider_name = self._resolve_provider_name(provider)
        elif self.config.openai_api_key:
            provider_name = "openai_compatible"
        else:
            provider_name = self._resolve_provider_name(None)
        expert_requested_model = model or (self.config.openai_model_name if provider_name == "openai_compatible" else None)
        self._append_session_message(session_id=session_id, role="user", content=user_message)
        final_text = ""
        try:
            selected_model = ""
            for chunk, model_name in self._run_direct_reply(
                session_id=session_id,
                mode="expert",
                provider_name=provider_name,
                requested_model=expert_requested_model,
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

            assistant_text = final_text.strip() or "Processing completed."
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
                "payload": {"message": f"Model call failed：{str(exc)}", "usage": self._get_usage_snapshot(session_id)},
            }
        except Exception as exc:
            yield {
                "event": "error",
                "content_type": "text",
                "payload": {"message": f"Model call failed：{str(exc)}", "usage": self._get_usage_snapshot(session_id)},
            }
