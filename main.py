import json
import os
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel


# 请求体定义：与前端约定的会话字段
class ChatRequest(BaseModel):
	session_id: str
	mode: str  # general | analyst
	message: str
	stream: bool = True


app = FastAPI(title="Role C Frontend Demo API")


def load_api_key() -> str | None:
	key_from_env = os.getenv("YUNWU_API_KEY")
	if key_from_env:
		return key_from_env

	test_api_path = Path(__file__).parent / "utils" / "test_api.py"
	if not test_api_path.exists():
		return None

	try:
		content = test_api_path.read_text(encoding="utf-8")
	except OSError:
		return None

	match = re.search(r"^key\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)
	return match.group(1) if match else None


API_BASE_URL = os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1")
API_KEY = load_api_key()
MODEL_NAME = os.getenv("YUNWU_MODEL", "gpt-4o")

if API_KEY:
	client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
else:
	client = None

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


# 将 JSON 事件包装为 SSE data 帧
def sse_data(payload: dict) -> str:
	return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
	# 以生成器方式持续返回事件，实现“流式输出”
	async def event_generator():
		if client is None:
			yield sse_data(
				{
					"event": "error",
					"content_type": "text",
					"payload": {"message": "后端未配置 YUNWU_API_KEY，无法调用模型接口。"},
				}
			)
			return

		messages = [{"role": "user", "content": req.message}]
		full_text = ""

		try:
			stream = client.chat.completions.create(
				model=MODEL_NAME,
				messages=messages,
				stream=True,
				timeout=100,
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
				yield sse_data(
					{
						"event": "token",
						"content_type": "text",
						"payload": {"delta": content},
					}
				)

			yield sse_data(
				{
					"event": "final",
					"content_type": "text",
					"payload": {"text": full_text or "处理完成。"},
				}
			)
		except Exception as exc:
			yield sse_data(
				{
					"event": "error",
					"content_type": "text",
					"payload": {"message": f"模型调用失败：{str(exc)}"},
				}
			)

	# `text/event-stream` 是 SSE 标准媒体类型
	return StreamingResponse(
		event_generator(),
		media_type="text/event-stream",
		headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
	)


# 将前端静态资源挂载到根路径，直接访问 / 即可打开页面
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
	app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

