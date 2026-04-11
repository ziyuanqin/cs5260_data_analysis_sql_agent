"""接口请求与响应的数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """流式聊天接口的请求体。"""

    # 前端会话 ID，在后端作为 session_id 使用。
    session_id: str
    # 对话模式（general / expert）。
    mode: str
    # 当前用户输入。
    message: str
    # 可选：本次请求指定后端提供方（如 openai_compatible / local_http）。
    provider: str | None = None
    # 可选：覆盖默认模型名。
    model: str | None = None
    # 可选：透传给具体后端的附加参数。
    provider_options: dict[str, Any] | None = None
    # 为协议兼容保留；本项目后端默认走流式返回。
    stream: bool = True


class ResetSessionRequest(BaseModel):
    """清理单个会话历史的请求体。"""

    session_id: str


class ResetSessionResponse(BaseModel):
    """清理会话后的响应结构。"""

    # 请求是否被正常处理。
    ok: bool
    # 会话是否真实存在且已被删除。
    cleared: bool


class RewriteLastUserRequest(BaseModel):
    """截断最后一条用户消息及后续内容的请求体。"""

    session_id: str


class RewriteLastUserResponse(BaseModel):
    """截断最后一条用户消息后的响应结构。"""

    ok: bool
    rewritten: bool
    removed_messages: int
    remaining_messages: int
