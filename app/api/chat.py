"""聊天接口路由。

包含接口：
- POST /api/chat/stream：流式返回助手回复
- POST /api/chat/reset：清理指定会话历史
"""
import os
import shutil
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from app.models import (
    ChatRequest,
    ResetSessionRequest,
    ResetSessionResponse,
    RewriteLastUserRequest,
    RewriteLastUserResponse,
)
from app.services.llm_service import ChatService
from app.utils.sse import sse_data

router = APIRouter(prefix="/api/chat", tags=["chat"])


def get_chat_service(request: Request) -> ChatService:
    """依赖注入函数：从 app.state 取出 ChatService。"""

    return request.app.state.chat_service


@router.post("/stream")
async def chat_stream(req: ChatRequest, request: Request, chat_service: ChatService = Depends(get_chat_service)):
    """以流式方式返回回复，供前端增量渲染。"""
    sql_app = None
    if hasattr(request.app.state, "sql_apps"):
        # 🌟 强制转为字符串进行匹配，防止 UUID 对象与字符串 Key 匹配失败
        sid = str(req.session_id)
        sql_app = request.app.state.sql_apps.get(sid)

        if sql_app:
            print(f"--- [DEBUG] 成功匹配 Session: {sid} ---")
        else:
            # 加上这行调试，看看字典里到底存了什么，以及你现在查的是什么
            print(f"--- [DEBUG] 匹配失败。当前字典内容: {list(request.app.state.sql_apps.keys())}")
            print(f"--- [DEBUG] 当前请求查询的 ID: {sid} ---")
    async def event_generator() -> AsyncGenerator[str, None]:
        # 把服务层产生的事件字典转换为 SSE 文本帧。
        async for event in chat_service.stream_chat_events(
            req.session_id,
            req.mode,
            req.message,
            provider=req.provider,
            model=req.model,
            provider_options=req.provider_options,
            sql_app=sql_app,
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


@router.post("/rewrite-last-user", response_model=RewriteLastUserResponse)
async def rewrite_last_user(req: RewriteLastUserRequest, chat_service: ChatService = Depends(get_chat_service)):
    """删除最后一条用户消息及其后续内容，用于“编辑最后一条并重跑”场景。"""

    result = chat_service.rewrite_last_user_turn(req.session_id)
    return RewriteLastUserResponse(
        ok=True,
        rewritten=bool(result.get("rewritten", False)),
        removed_messages=int(result.get("removed_messages", 0)),
        remaining_messages=int(result.get("remaining_messages", 0)),
    )


@router.get("/providers")
async def list_chat_providers(chat_service: ChatService = Depends(get_chat_service)):
    """返回可选 provider 列表，便于前端做后端切换。"""

    return {"providers": chat_service.list_supported_providers()}


@router.get("/artifacts/{session_id}")
async def list_chat_artifacts(session_id: str, chat_service: ChatService = Depends(get_chat_service)):
    """列出指定会话当前可下载产物。"""

    return {"session_id": session_id, "items": chat_service.list_artifacts(session_id)}


@router.get("/artifacts/{session_id}/download/{artifact_name:path}")
async def download_chat_artifact(
    session_id: str,
    artifact_name: str,
    chat_service: ChatService = Depends(get_chat_service),
):
    """下载单个产物文件。"""

    try:
        target = chat_service.resolve_artifact_file(session_id, artifact_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(path=str(target), filename=target.name, media_type="application/octet-stream")


@router.get("/artifacts/{session_id}/bundle")
async def download_chat_artifact_bundle(session_id: str, chat_service: ChatService = Depends(get_chat_service)):
    """下载会话产物 ZIP 包。"""

    items = chat_service.list_artifacts(session_id)
    if not items:
        raise HTTPException(status_code=404, detail="No artifacts available for this session.")

    bundle_path = chat_service.build_artifact_bundle(session_id)
    return FileResponse(path=str(bundle_path), filename=bundle_path.name, media_type="application/zip")


@router.post("/cleanup")
async def cleanup_session_files():
    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, "dataset")
    if "backend" in os.listdir(base_dir):
        dataset_dir = os.path.join(base_dir, "backend", "dataset")

    print(f"--- [Cleanup] 正在执行全量清理: {dataset_dir} ---")

    if not os.path.exists(dataset_dir):
        return {"status": "ok", "message": "目录不存在"}

    try:
        count = 0
        # 遍历目录下的所有内容
        for filename in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path) # 删除文件或链接
                    count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) # 删除子目录
                    count += 1
            except Exception as e:
                print(f"--- [Cleanup] 无法删除 {file_path}: {e} ---")

        return {"status": "success", "deleted_count": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}
