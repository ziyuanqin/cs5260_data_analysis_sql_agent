from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.models import ChatRequest, ResetSessionRequest, ResetSessionResponse
from app.services.llm_service import ChatService
from app.utils.sse import sse_data

router = APIRouter(prefix="/api/chat", tags=["chat"])


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


@router.post("/stream")
async def chat_stream(req: ChatRequest, chat_service: ChatService = Depends(get_chat_service)):
    async def event_generator() -> AsyncGenerator[str, None]:
        for event in chat_service.stream_chat_events(req.session_id, req.mode, req.message):
            yield sse_data(event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/reset", response_model=ResetSessionResponse)
async def reset_chat_session(req: ResetSessionRequest, chat_service: ChatService = Depends(get_chat_service)):
    cleared = chat_service.reset_session(req.session_id)
    return ResetSessionResponse(ok=True, cleared=cleared)
