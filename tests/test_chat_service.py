from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig
from app.services.llm_service import ChatService

try:
    from langgraph.graph import StateGraph as _StateGraph  # noqa: F401

    HAS_LANGGRAPH = True
except ModuleNotFoundError:
    HAS_LANGGRAPH = False


class FakeProvider:
    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    def stream_chat(self, messages, model, timeout, provider_options=None):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "timeout": timeout,
                "provider_options": provider_options or {},
            }
        )
        output = self._responder(messages, model, provider_options or {})
        if isinstance(output, Exception):
            raise output
        text = str(output)
        if not text:
            return
        midpoint = max(1, len(text) // 2)
        yield text[:midpoint]
        yield text[midpoint:]


class FakeRegistry:
    def __init__(self, provider):
        self._provider = provider

    def get(self, _provider_name):
        return self._provider

    def list_supported(self):
        return ["openai_compatible", "deepseek_compatible", "local_http", "huggingface"]


def make_config(**overrides) -> AppConfig:
    temp_frontend_dir = PROJECT_ROOT / "tests" / f"_tmp_frontend_{uuid.uuid4().hex}"
    values = {
        "default_provider": "openai_compatible",
        "openai_api_base_url": "https://example.com/v1",
        "openai_api_key": "openai-key",
        "openai_model_name": "gpt-4o-mini",
        "deepseek_api_base_url": "https://api.deepseek.com/v1",
        "deepseek_api_key": "deepseek-key",
        "huggingface_api_url": "https://router.huggingface.co/v1/chat/completions",
        "huggingface_api_key": "hf-token",
        "huggingface_model_name": "Qwen/Qwen3.5-9B:together",
        "local_api_url": "http://127.0.0.1:9000/chat",
        "local_model_name": "local-model",
        "general_openai_provider": "openai_compatible",
        "general_openai_model_name": "gpt-4o-mini",
        "general_deepseek_provider": "openai_compatible",
        "general_deepseek_model_name": "deepseek-chat",
        "agent_intent_model_name": "gpt-4o-mini",
        "agent_planner_model_name": "gpt-4o-mini",
        "agent_executor_model_name": "gpt-4o-mini",
        "agent_reviewer_model_name": "gpt-4o-mini",
        "agent_summarizer_model_name": "gpt-4o-mini",
        "agent_max_review_retries": 1,
        "fallback_model_name": "gpt-4o-mini",
        "session_token_budget": 32000,
        "session_cost_budget_usd": 1.0,
        "default_price_per_1k_tokens_usd": 0.001,
        "model_price_per_1k_tokens_usd": {
            "gpt-4o-mini": 0.001,
            "deepseek-chat": 0.001,
        },
        "planning_keyword_routing_enabled": True,
        "planning_long_input_chars": 180,
        "planning_multi_question_marks": 2,
        "planning_short_direct_chars": 20,
        "request_timeout": 30,
        "history_max_messages": 20,
        "memory_auto_compress_enabled": True,
        "memory_keep_recent_messages": 12,
        "memory_summary_max_chars": 3000,
        "frontend_dir": temp_frontend_dir,
    }
    values.update(overrides)
    return AppConfig(**values)


async def collect_events(service: ChatService, **kwargs):
    events = []
    async for event in service.stream_chat_events(**kwargs):
        events.append(event)
    return events


class ChatServiceTests(unittest.TestCase):
    def test_memory_auto_compress_creates_summary(self):
        service = ChatService(
            make_config(
                history_max_messages=6,
                memory_auto_compress_enabled=True,
                memory_keep_recent_messages=4,
                memory_summary_max_chars=1200,
            )
        )
        for idx in range(8):
            service._append_session_message("s-memory", "user", f"用户消息 {idx}")
            service._append_session_message("s-memory", "assistant", f"助手回复 {idx}")

        self.assertLessEqual(len(service._sessions.get("s-memory", [])), 4)
        summary = service._session_summaries.get("s-memory", "")
        self.assertTrue(summary)
        self.assertIn("用户消息", summary)

        built = service._build_messages("s-memory", "general")
        self.assertTrue(any("Conversation memory summary" in str(item.get("content")) for item in built))

    def test_rewrite_last_user_turn_keeps_previous_context(self):
        service = ChatService(make_config())
        service._append_session_message("s-edit", "user", "第一轮问题")
        service._append_session_message("s-edit", "assistant", "第一轮回答")
        service._append_session_message("s-edit", "user", "第二轮问题")
        service._append_session_message("s-edit", "assistant", "第二轮回答")

        result = service.rewrite_last_user_turn("s-edit")

        self.assertEqual(result["rewritten"], True)
        self.assertEqual(result["removed_messages"], 2)
        session_messages = service._sessions.get("s-edit", [])
        self.assertEqual(len(session_messages), 2)
        self.assertEqual(session_messages[-1]["role"], "assistant")
        self.assertEqual(session_messages[-1]["content"], "第一轮回答")

        # Rehydrate to confirm SQLite memory was updated as well.
        service_2 = ChatService(make_config(frontend_dir=service.config.frontend_dir))
        service_2._hydrate_session("s-edit")
        hydrated = service_2._sessions.get("s-edit", [])
        self.assertEqual(len(hydrated), 2)

    def test_general_mode_direct_returns_final(self):
        def responder(_messages, _model, _options):
            return "你好，已收到。"

        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(FakeProvider(responder))

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-general",
                mode="general",
                user_message="你好",
                provider="openai_compatible",
                provider_options={"general_model": "openai"},
            )
        )

        self.assertTrue(any(event.get("event") == "token" for event in events))
        self.assertEqual(events[-1]["event"], "final")
        self.assertEqual(events[-1]["payload"]["text"], "你好，已收到。")

    def test_general_direct_retries_on_premature_stream_end(self):
        calls = {"count": 0}

        def responder(_messages, _model, _options):
            calls["count"] += 1
            if calls["count"] == 1:
                return RuntimeError("Response ended prematurely")
            return "第二次成功"

        service = ChatService(make_config())
        fake_provider = FakeProvider(responder)
        service.provider_registry = FakeRegistry(fake_provider)

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-retry-premature-end",
                mode="general",
                user_message="你好",
                provider="openai_compatible",
                provider_options={"general_model": "openai"},
            )
        )

        self.assertEqual(events[-1]["event"], "final")
        self.assertEqual(events[-1]["payload"]["text"], "第二次成功")
        self.assertGreaterEqual(calls["count"], 2)

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph is required for plan-loop tests")
    def test_general_mode_plan_loop_emits_stage_status(self):
        def responder(messages, _model, _options):
            prompt = messages[-1]["content"]
            if "Return JSON array only" in prompt:
                return '[{"task":"检查环境"},{"task":"修复配置"}]'
            if "Execute this step" in prompt:
                return "步骤执行完成"
            if "Reply exactly in one line" in prompt:
                return "PASS"
            if "Create the final user-facing answer" in prompt:
                return "最终总结"
            if "router for a general assistant" in prompt:
                return "PLAN"
            return "OK"

        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(FakeProvider(responder))

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-plan",
                mode="general",
                user_message="请规划一个上线前排查步骤",
                provider="openai_compatible",
                provider_options={"general_model": "openai"},
            )
        )
        stage_events = [
            event
            for event in events
            if event.get("event") == "token"
            and isinstance(event.get("payload", {}).get("meta"), dict)
            and event.get("payload", {}).get("meta", {}).get("kind") == "status"
        ]
        stage_names = [event["payload"]["meta"].get("stage") for event in stage_events]
        self.assertIn("router", stage_names)
        self.assertIn("planner", stage_names)
        self.assertIn("executor", stage_names)
        self.assertIn("reviewer", stage_names)
        planner_status = next(
            event
            for event in stage_events
            if event["payload"]["meta"].get("stage") == "planner"
        )
        self.assertIn("task_items", planner_status["payload"]["meta"])
        progress_events = [
            event
            for event in events
            if event.get("event") == "token"
            and isinstance(event.get("payload", {}).get("meta"), dict)
            and event.get("payload", {}).get("meta", {}).get("kind") == "progress"
        ]
        self.assertTrue(progress_events)
        self.assertEqual(events[-1]["event"], "final")
        self.assertEqual(events[-1]["payload"]["text"], "最终总结")

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph is required for capability routing tests")
    def test_general_mode_website_capability_forces_plan(self):
        def responder(messages, _model, _options):
            prompt = messages[-1]["content"]
            if "Execute this step" in prompt:
                return "步骤执行完成"
            if "Reply exactly in one line" in prompt:
                return "PASS"
            if "Create the final user-facing answer" in prompt:
                return "网站交付完成"
            return "OK"

        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(FakeProvider(responder))

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-website-cap",
                mode="general",
                user_message="请帮我生成一个项目介绍网站并可下载",
                provider="openai_compatible",
                provider_options={"general_model": "openai"},
            )
        )

        router_status = next(
            event
            for event in events
            if event.get("event") == "token"
            and event.get("payload", {}).get("meta", {}).get("stage") == "router"
        )
        self.assertIn("capability_website_builder", router_status["payload"].get("delta", ""))
        self.assertTrue(router_status["payload"]["meta"].get("task_items"))
        self.assertEqual(events[-1]["event"], "final")
        self.assertEqual(events[-1]["payload"].get("capability"), "website_builder")

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph is required for capability routing tests")
    def test_general_mode_manual_capability_overrides_keyword_detection(self):
        def responder(messages, _model, _options):
            prompt = messages[-1]["content"]
            if "Execute this step" in prompt:
                return "步骤执行完成"
            if "Reply exactly in one line" in prompt:
                return "PASS"
            if "Create the final user-facing answer" in prompt:
                return "幻灯片交付完成"
            return "OK"

        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(FakeProvider(responder))

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-manual-capability",
                mode="general",
                user_message="请帮我做一个项目总结",
                provider="openai_compatible",
                provider_options={"general_model": "openai", "general_capability": "slide_builder"},
            )
        )

        router_status = next(
            event
            for event in events
            if event.get("event") == "token"
            and event.get("payload", {}).get("meta", {}).get("stage") == "router"
        )
        self.assertIn("capability_slide_builder", router_status["payload"].get("delta", ""))
        self.assertEqual(events[-1]["event"], "final")
        self.assertEqual(events[-1]["payload"].get("capability"), "slide_builder")

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph is required for website enforcement tests")
    def test_website_builder_forces_complete_assets_when_executor_skips_tool(self):
        def responder(messages, _model, _options):
            prompt = str(messages[-1]["content"])
            lower_prompt = prompt.lower()

            if "you are an executor with optional tool usage" in lower_prompt:
                # Simulate a bad model behavior: it refuses to call file_write.
                return '{"action":"RESPOND","response":"Step done without writing files."}'
            if "return json only with keys: index_html, style_css, app_js, readme_md." in lower_prompt:
                return (
                    '{"index_html":"<!doctype html><html><head><title>Demo</title>'
                    '<link rel=\\"stylesheet\\" href=\\"./style.css\\"></head>'
                    '<body><main><h1>Demo Site</h1><p>Content</p></main><script src=\\"./app.js\\"></script></body></html>",'
                    '"style_css":"body{font-family:Arial;} main{max-width:680px;margin:40px auto;}",'
                    '"app_js":"console.log(\\"ok\\");",'
                    '"readme_md":"# Demo\\nGenerated by test."}'
                )
            if "based on the executed tool result" in lower_prompt:
                return "Generated files: index.html, style.css, app.js"
            if "reply exactly in one line" in lower_prompt:
                if "html/css/js" in lower_prompt:
                    # Reviewer intentionally says FAIL; graph heuristics should still pass
                    # when execution result already contains html/css/js outputs.
                    return "FAIL: style.css and app.js are missing."
                return "PASS"
            if "create the final user-facing answer" in lower_prompt:
                return "Website delivery completed."
            return "OK"

        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(FakeProvider(responder))

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-website-enforce",
                mode="general",
                user_message="Build a portfolio website for a student developer.",
                provider="openai_compatible",
                provider_options={"general_model": "openai", "general_capability": "website_builder"},
            )
        )

        self.assertEqual(events[-1]["event"], "final")
        artifact_names = {
            str(item.get("relative_path") or item.get("name") or "")
            for item in (events[-1]["payload"].get("artifacts") or [])
            if isinstance(item, dict)
        }
        self.assertIn("index.html", artifact_names)
        self.assertIn("style.css", artifact_names)
        self.assertIn("app.js", artifact_names)

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph is required for evidence tests")
    def test_general_mode_emits_evidence_for_tool(self):
        def responder(messages, _model, _options):
            prompt = messages[-1]["content"]
            if "Return JSON array only" in prompt:
                return '[{"task":"读取项目说明"}]'
            if "You are an executor with optional tool usage" in prompt:
                return '{"action":"TOOL","tool":"file_read","input":"README.md","reason":"need context"}'
            if "Based on the executed tool result" in prompt:
                return "已读取文件，完成步骤"
            if "Reply exactly in one line" in prompt:
                return "PASS"
            if "Create the final user-facing answer" in prompt:
                return "最终总结"
            if "router for a general assistant" in prompt:
                return "PLAN"
            return "OK"

        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(FakeProvider(responder))

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-evidence",
                mode="general",
                user_message="请规划步骤并读取 README",
                provider="openai_compatible",
                provider_options={"general_model": "openai", "general_file_access": True},
            )
        )

        evidence_events = [
            event
            for event in events
            if event.get("event") == "token"
            and isinstance(event.get("payload", {}).get("meta"), dict)
            and event.get("payload", {}).get("meta", {}).get("kind") == "evidence"
        ]
        self.assertTrue(evidence_events)
        source = evidence_events[0]["payload"]["meta"]["item"].get("source", "")
        self.assertIn("README.md", source)

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph is required for web-search evidence tests")
    def test_general_mode_web_search_tool_emits_evidence(self):
        def responder(messages, _model, _options):
            prompt = messages[-1]["content"]
            if "Return JSON array only" in prompt:
                return '[{"task":"检索最新信息"}]'
            if "You are an executor with optional tool usage" in prompt:
                return '{"action":"TOOL","tool":"web_search","input":"latest llm benchmark","reason":"need up-to-date links"}'
            if "Based on the executed tool result" in prompt:
                return "已完成检索并总结"
            if "Reply exactly in one line" in prompt:
                return "PASS"
            if "Create the final user-facing answer" in prompt:
                return "最终总结"
            if "router for a general assistant" in prompt:
                return "PLAN"
            return "OK"

        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(FakeProvider(responder))
        service._tool_web_search = lambda _query: "1. Example result | https://example.com"  # type: ignore[method-assign]

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-web-search",
                mode="general",
                user_message="请规划并搜索最新资料",
                provider="openai_compatible",
                provider_options={"general_model": "openai", "general_web_search": True},
            )
        )

        evidence_events = [
            event
            for event in events
            if event.get("event") == "token"
            and isinstance(event.get("payload", {}).get("meta"), dict)
            and event.get("payload", {}).get("meta", {}).get("kind") == "evidence"
        ]
        self.assertTrue(evidence_events)
        item = evidence_events[0]["payload"]["meta"]["item"]
        self.assertEqual(item.get("tool"), "web_search")
        self.assertIn("duckduckgo.com", str(item.get("source", "")))

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph is required for tool-guardrail tests")
    def test_general_mode_disables_file_tool_by_default(self):
        def responder(messages, _model, _options):
            prompt = messages[-1]["content"]
            if "Return JSON array only" in prompt:
                return '[{"task":"读取本地文件"}]'
            if "You are an executor with optional tool usage" in prompt:
                return '{"action":"TOOL","tool":"file_read","input":"requirements.txt","reason":"need context"}'
            if "Reply exactly in one line" in prompt:
                return "PASS"
            if "Create the final user-facing answer" in prompt:
                return "最终总结"
            if "router for a general assistant" in prompt:
                return "PLAN"
            return "OK"

        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(FakeProvider(responder))

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-tool-guardrail",
                mode="general",
                user_message="请读取 requirements 文件",
                provider="openai_compatible",
                provider_options={"general_model": "openai"},
            )
        )

        evidence_events = [
            event
            for event in events
            if event.get("event") == "token"
            and isinstance(event.get("payload", {}).get("meta"), dict)
            and event.get("payload", {}).get("meta", {}).get("kind") == "evidence"
        ]
        self.assertFalse(evidence_events)
        self.assertEqual(events[-1]["event"], "final")

    def test_file_write_tool_rejects_path_escape(self):
        service = ChatService(make_config())
        with self.assertRaises(ValueError):
            service._tool_file_write("s-write-guard", '{"path":"../hack.txt","content":"x"}')

    def test_file_write_tool_writes_inside_artifact_workspace(self):
        service = ChatService(make_config())
        output = service._tool_file_write(
            "s-write-ok",
            '{"path":"deliverables/index.html","content":"<html><body>ok</body></html>","overwrite":true}',
        )
        parsed = service._extract_json_object(output)
        self.assertTrue(parsed.get("ok"))
        resolved = service.resolve_artifact_file("s-write-ok", "deliverables/index.html")
        self.assertTrue(resolved.exists())

    def test_budget_guardrail_returns_error_event(self):
        def responder(_messages, _model, _options):
            return "should not be called"

        service = ChatService(make_config(session_token_budget=3))
        service.provider_registry = FakeRegistry(FakeProvider(responder))

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-budget",
                mode="general",
                user_message="This input intentionally exceeds the configured budget threshold.",
                provider="openai_compatible",
                provider_options={"general_model": "openai"},
            )
        )

        self.assertEqual(events[-1]["event"], "error")
        self.assertIn("budget", events[-1]["payload"]["message"].lower())

    def test_general_model_alias_routes_deepseek_model(self):
        def responder(_messages, _model, _options):
            return "ok"

        fake_provider = FakeProvider(responder)
        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(fake_provider)

        asyncio.run(
            collect_events(
                service,
                session_id="s-alias",
                mode="general",
                user_message="hello",
                provider=None,
                provider_options={"general_model": "deepseek"},
            )
        )

        self.assertTrue(fake_provider.calls)
        self.assertEqual(fake_provider.calls[0]["model"], "deepseek-chat")

    def test_general_model_alias_routes_huggingface_model(self):
        service = ChatService(make_config(huggingface_model_name="Qwen/Qwen3.5-9B:together"))
        provider_name, model_name = service._resolve_general_backend(
            provider=None,
            model=None,
            provider_options={"general_model": "huggingface"},
        )
        self.assertEqual(provider_name, "huggingface")
        self.assertEqual(model_name, "Qwen/Qwen3.5-9B:together")

    def test_general_raw_model_id_prefers_openai_compatible_when_openai_key_exists(self):
        service = ChatService(
            make_config(
                general_openai_provider="huggingface",
                openai_api_key="openai-key",
            )
        )
        provider_name, model_name = service._resolve_general_backend(
            provider=None,
            model=None,
            provider_options={"general_model": "zai-org/GLM-5.1:together"},
        )
        self.assertEqual(provider_name, "openai_compatible")
        self.assertEqual(model_name, "zai-org/GLM-5.1:together")

    def test_general_raw_model_id_falls_back_to_configured_provider_without_openai_key(self):
        service = ChatService(
            make_config(
                general_openai_provider="huggingface",
                openai_api_key=None,
            )
        )
        provider_name, model_name = service._resolve_general_backend(
            provider=None,
            model=None,
            provider_options={"general_model": "zai-org/GLM-5.1:together"},
        )
        self.assertEqual(provider_name, "huggingface")
        self.assertEqual(model_name, "zai-org/GLM-5.1:together")

    def test_general_model_alias_routes_deepseek_provider_when_deepseek_key_exists(self):
        service = ChatService(
            make_config(
                deepseek_api_base_url="https://api.deepseek.com/v1",
                deepseek_api_key="deepseek-key",
            )
        )
        provider_name, model_name = service._resolve_general_backend(
            provider=None,
            model=None,
            provider_options={"general_model": "deepseek"},
        )
        self.assertEqual(provider_name, "deepseek_compatible")
        self.assertEqual(model_name, "deepseek-chat")

    @unittest.skipUnless(HAS_LANGGRAPH, "langgraph is required for plan-loop tests")
    def test_general_alias_model_is_used_for_all_agent_stages(self):
        selected_hf_model = "Qwen/Qwen3.5-9B:together"

        def responder(messages, _model, _options):
            prompt = messages[-1]["content"]
            if "Return JSON array only" in prompt:
                return '[{"task":"检查环境"},{"task":"整理结论"}]'
            if "Execute this step" in prompt:
                return "步骤执行完成"
            if "Reply exactly in one line" in prompt:
                return "PASS"
            if "Create the final user-facing answer" in prompt:
                return "最终总结"
            if "router for a general assistant" in prompt:
                return "PLAN"
            return "OK"

        fake_provider = FakeProvider(responder)
        service = ChatService(
            make_config(
                default_provider="huggingface",
                huggingface_model_name=selected_hf_model,
                # Keep agent defaults as GPT to verify they do not override the selected alias model.
                agent_intent_model_name="gpt-4o-mini",
                agent_planner_model_name="gpt-4o-mini",
                agent_executor_model_name="gpt-4o",
                agent_reviewer_model_name="gpt-4o-mini",
                agent_summarizer_model_name="gpt-4o-mini",
            )
        )
        service.provider_registry = FakeRegistry(fake_provider)

        asyncio.run(
            collect_events(
                service,
                session_id="s-hf-lock",
                mode="general",
                user_message="请规划一个上线检查步骤",
                provider=None,
                provider_options={"general_model": "huggingface"},
            )
        )

        used_models = {call["model"] for call in fake_provider.calls}
        self.assertEqual(used_models, {selected_hf_model})

    def test_expert_sql_not_intercepted_by_general_router(self):
        service = ChatService(make_config())
        service.provider_registry = FakeRegistry(FakeProvider(lambda *_args, **_kwargs: "unused"))

        async def fake_sql(**_kwargs):
            yield {"event": "token", "payload": {"delta": "sql-result"}}
            yield {"event": "final", "payload": {"text": "sql-final"}}

        service._run_sql_analysis = fake_sql  # type: ignore[method-assign]

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-sql",
                mode="expert",
                user_message="请做SQL分析",
                provider_options={"sql_analysis": True},
            )
        )

        self.assertEqual(events[-1]["event"], "final")
        self.assertEqual(events[-1]["payload"]["text"], "sql-final")
        router_status = [
            event
            for event in events
            if event.get("event") == "token"
            and event.get("payload", {}).get("meta", {}).get("stage") == "router"
        ]
        self.assertFalse(router_status)


if __name__ == "__main__":
    unittest.main()
