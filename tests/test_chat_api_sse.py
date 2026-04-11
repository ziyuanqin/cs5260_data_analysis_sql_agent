from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.chat import router as chat_router

    HAS_FASTAPI = True
except ModuleNotFoundError:
    HAS_FASTAPI = False


class FakeChatService:
    def __init__(self):
        self.last_call = None

    async def stream_chat_events(
        self,
        session_id: str,
        mode: str,
        user_message: str,
        provider=None,
        model=None,
        provider_options=None,
        sql_app=None,
    ):
        self.last_call = {
            "session_id": session_id,
            "mode": mode,
            "user_message": user_message,
            "provider": provider,
            "model": model,
            "provider_options": provider_options,
            "sql_app": sql_app,
        }
        yield {"event": "token", "payload": {"delta": "hello"}}
        yield {"event": "final", "payload": {"text": "hello"}}

    def reset_session(self, _session_id: str) -> bool:
        return True

    def list_supported_providers(self):
        return ["openai_compatible", "local_http"]


@unittest.skipUnless(HAS_FASTAPI, "fastapi is required for API integration tests")
class ChatApiSseTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.service = FakeChatService()
        self.app.state.chat_service = self.service
        self.app.state.sql_apps = {}
        self.app.include_router(chat_router)
        self.client = TestClient(self.app)

    def test_stream_endpoint_returns_sse_frames(self):
        response = self.client.post(
            "/api/chat/stream",
            json={
                "session_id": "api-sse",
                "mode": "general",
                "message": "hello",
                "provider_options": {"general_model": "openai"},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        self.assertIn('"event": "token"', response.text)
        self.assertIn('"event": "final"', response.text)

    def test_stream_endpoint_passes_provider_options(self):
        self.client.post(
            "/api/chat/stream",
            json={
                "session_id": "api-pass",
                "mode": "expert",
                "message": "run sql",
                "provider_options": {"sql_analysis": True, "file_name": "demo.csv"},
            },
        )
        self.assertIsNotNone(self.service.last_call)
        self.assertEqual(self.service.last_call["provider_options"]["sql_analysis"], True)
        self.assertEqual(self.service.last_call["provider_options"]["file_name"], "demo.csv")


if __name__ == "__main__":
    unittest.main()
