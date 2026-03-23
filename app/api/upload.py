import os
import json
from urllib import parse, request as urlrequest

from fastapi import APIRouter, UploadFile, File, HTTPException, Request

# 模仿 chat.py 的 prefix 风格
router = APIRouter(prefix="/api/upload", tags=["upload"])


def _build_multipart_body(field_name: str, filename: str, file_bytes: bytes, content_type: str = "application/octet-stream") -> tuple[bytes, str]:
    """构造 multipart/form-data 请求体，用于代理上传到 EDA 服务。"""

    boundary = "----WebKitFormBoundaryYUNWUEDAUpload"
    head = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = head + file_bytes + tail
    return body, boundary


def _proxy_upload_to_eda(base_url: str, session_id: str | None, filename: str, file_bytes: bytes, timeout: int) -> dict:
    """把文件转发给 EDA 后端 /upload 接口，返回其 JSON。"""

    query = parse.urlencode({"thread_id": session_id}) if session_id else ""
    upload_url = f"{base_url.rstrip('/')}/upload"
    if query:
        upload_url = f"{upload_url}?{query}"

    body, boundary = _build_multipart_body("file", filename, file_bytes)
    req = urlrequest.Request(
        upload_url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"raw": raw}

    return parsed if isinstance(parsed, dict) else {"raw": raw}

@router.post("")
async def save_uploaded_file(
    request: Request,
    file: UploadFile = File(...),
    use_eda: bool = False,
    session_id: str | None = None,
):
    # 从 factory.py 传过来的可靠绝对路径获取
    upload_dir = request.app.state.upload_dir

    print(f"--- 收到上传请求: {file.filename} ---", flush=True)

    try:
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)

        content = await file.read()

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        eda_payload = None
        if use_eda:
            eda_payload = _proxy_upload_to_eda(
                base_url=request.app.state.config.eda_api_base_url,
                session_id=session_id,
                filename=file.filename,
                file_bytes=content,
                timeout=request.app.state.config.request_timeout,
            )

        print(f"✅ 文件已保存至: {file_path}")
        response_payload = {"file_name": file.filename, "status": "success"}
        if isinstance(eda_payload, dict):
            response_payload["eda"] = eda_payload
            thread_id = eda_payload.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                response_payload["eda_thread_id"] = thread_id

        return response_payload
    except Exception as e:
        print(f"❌ 写入失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))