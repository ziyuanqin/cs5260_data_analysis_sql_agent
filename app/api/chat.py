"""聊天接口路由。

包含接口：
- POST /api/chat/stream：流式返回助手回复
- POST /api/chat/reset：清理指定会话历史
"""

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.models import ChatRequest, ResetSessionRequest, ResetSessionResponse
from app.services.llm_service import ChatService
from app.utils.sse import sse_data

router = APIRouter(prefix="/api/chat", tags=["chat"])


def get_chat_service(request: Request) -> ChatService:
    """依赖注入函数：从 app.state 取出 ChatService。"""

    return request.app.state.chat_service


@router.post("/stream")
async def chat_stream(req: ChatRequest, chat_service: ChatService = Depends(get_chat_service)):
    """以流式方式返回回复，供前端增量渲染。"""

    async def event_generator() -> AsyncGenerator[str, None]:
        # 把服务层产生的事件字典转换为 SSE 文本帧。
        for event in chat_service.stream_chat_events(
            req.session_id,
            req.mode,
            req.message,
            provider=req.provider,
            model=req.model,
            provider_options=req.provider_options,
        ):
            yield sse_data(event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/reset", response_model=ResetSessionResponse)
async def reset_chat_session(req: ResetSessionRequest, chat_service: ChatService = Depends(get_chat_service)):
    """按 session_id 清理后端会话历史。"""

    cleared = chat_service.reset_session(req.session_id)
    return ResetSessionResponse(ok=True, cleared=cleared)


@router.get("/providers")
async def list_chat_providers(chat_service: ChatService = Depends(get_chat_service)):
    """返回可选 provider 列表，便于前端做后端切换。"""

    return {"providers": chat_service.list_supported_providers()}
