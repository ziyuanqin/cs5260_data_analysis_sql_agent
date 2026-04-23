from __future__ import annotations

import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

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
        self._tmpdir_ctx = TemporaryDirectory()
        self._tmpdir = Path(self._tmpdir_ctx.name)
        self._artifact_file = self._tmpdir / "demo.html"
        self._artifact_file.write_text("<html><body>demo</body></html>", encoding="utf-8")
        self._bundle_file = self._tmpdir / "bundle.zip"
        self._bundle_file.write_bytes(b"PK\x03\x04")

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

    def rewrite_last_user_turn(self, _session_id: str):
        return {"rewritten": True, "removed_messages": 2, "remaining_messages": 4}

    def list_supported_providers(self):
        return ["openai_compatible", "local_http"]

    def list_artifacts(self, _session_id: str):
        return [
            {
                "name": "demo.html",
                "relative_path": "demo.html",
                "size_bytes": int(self._artifact_file.stat().st_size),
                "previewable": True,
            }
        ]

    def resolve_artifact_file(self, _session_id: str, artifact_name: str) -> Path:
        if artifact_name != "demo.html":
            raise FileNotFoundError(f"Artifact not found: {artifact_name}")
        return self._artifact_file

    def build_artifact_bundle(self, _session_id: str) -> Path:
        return self._bundle_file


@unittest.skipUnless(HAS_FASTAPI, "fastapi is required for API integration tests")
class ChatApiSseTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.service = FakeChatService()
        self.app.state.chat_service = self.service
        self.app.state.sql_apps = {}
        self.app.include_router(chat_router)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.service._tmpdir_ctx.cleanup()

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

    def test_artifact_list_endpoint(self):
        response = self.client.get("/api/chat/artifacts/demo-session")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session_id"], "demo-session")
        self.assertTrue(payload["items"])

    def test_artifact_download_endpoint(self):
        response = self.client.get("/api/chat/artifacts/demo-session/download/demo.html")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/octet-stream", response.headers.get("content-type", ""))
        self.assertIn("demo", response.text)

    def test_artifact_preview_endpoint(self):
        response = self.client.get("/api/chat/artifacts/demo-session/preview/demo.html")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("demo", response.text)

    def test_artifact_bundle_endpoint(self):
        response = self.client.get("/api/chat/artifacts/demo-session/bundle")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/zip", response.headers.get("content-type", ""))

    def test_rewrite_last_user_endpoint(self):
        response = self.client.post("/api/chat/rewrite-last-user", json={"session_id": "demo-session"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["rewritten"], True)
        self.assertEqual(payload["removed_messages"], 2)


if __name__ == "__main__":
    unittest.main()
