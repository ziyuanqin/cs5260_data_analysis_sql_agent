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
        return ["openai_compatible", "local_http", "huggingface"]


def make_config(**overrides) -> AppConfig:
    temp_frontend_dir = PROJECT_ROOT / "tests" / f"_tmp_frontend_{uuid.uuid4().hex}"
    values = {
        "default_provider": "openai_compatible",
        "openai_api_base_url": "https://example.com/v1",
        "openai_api_key": "openai-key",
        "openai_model_name": "gpt-4o-mini",
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
        "session_token_budget": 3000,
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
        self.assertEqual(events[-1]["event"], "final")
        self.assertEqual(events[-1]["payload"]["text"], "最终总结")

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
        self.assertTrue(evidence_events)
        source = evidence_events[0]["payload"]["meta"]["item"].get("source", "")
        self.assertIn("README.md", source)

    def test_budget_guardrail_returns_error_event(self):
        def responder(_messages, _model, _options):
            return "不会被调用"

        service = ChatService(make_config(session_token_budget=3))
        service.provider_registry = FakeRegistry(FakeProvider(responder))

        events = asyncio.run(
            collect_events(
                service,
                session_id="s-budget",
                mode="general",
                user_message="这是一个明显超过预算阈值的测试输入",
                provider="openai_compatible",
                provider_options={"general_model": "openai"},
            )
        )

        self.assertEqual(events[-1]["event"], "error")
        self.assertIn("预算", events[-1]["payload"]["message"])

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
