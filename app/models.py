from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    mode: str
    message: str
    stream: bool = True


class ResetSessionRequest(BaseModel):
    session_id: str


class ResetSessionResponse(BaseModel):
    ok: bool
    cleared: bool
