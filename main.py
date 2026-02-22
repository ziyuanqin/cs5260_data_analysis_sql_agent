import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# 请求体定义：与前端约定的会话字段
class ChatRequest(BaseModel):
	session_id: str
	mode: str  # general | analyst
	message: str
	stream: bool = True


app = FastAPI(title="Role C Frontend Demo API")

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
		# 根据模式返回不同说明文案（仅示例）
		text = (
			f"收到你的消息：{req.message}。"
			if req.mode == "general"
			else f"分析师模式已启动，正在分析：{req.message}。"
		)

		# 逐字发送 token 事件，模拟大模型流式输出
		for token in text:
			await asyncio.sleep(0.02)
			yield sse_data(
				{
					"event": "token",
					"content_type": "text",
					"payload": {"delta": token},
				}
			)

		# 分析师模式下额外返回“工具结果”：表格、图表、文件
		if req.mode == "analyst":
			await asyncio.sleep(0.15)
			yield sse_data(
				{
					"event": "tool_result",
					"content_type": "table",
					"payload": {
						"columns": ["metric", "value"],
						"rows": [
							{"metric": "样本数", "value": 1200},
							{"metric": "均值", "value": 37.42},
							{"metric": "标准差", "value": 4.9},
						],
					},
				}
			)

			# 1x1 透明像素 PNG（演示占位图）
			tiny_png_base64 = (
				"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAAFT"
				"B5cAAAAASUVORK5CYII="
			)
			await asyncio.sleep(0.1)
			yield sse_data(
				{
					"event": "tool_result",
					"content_type": "chart",
					"payload": {
						"image_base64": tiny_png_base64,
						"alt": "demo chart",
					},
				}
			)

			await asyncio.sleep(0.05)
			yield sse_data(
				{
					"event": "tool_result",
					"content_type": "file",
					"payload": {
						"name": "summary.csv",
						"url": "https://example.com/summary.csv",
						"size": "12KB",
					},
				}
			)

		# 最终完成事件
		await asyncio.sleep(0.05)
		yield sse_data(
			{
				"event": "final",
				"content_type": "text",
				"payload": {"text": "处理完成。"},
			}
		)

	# `text/event-stream` 是 SSE 标准媒体类型
	return StreamingResponse(event_generator(), media_type="text/event-stream")


# 将前端静态资源挂载到根路径，直接访问 / 即可打开页面
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
	app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

