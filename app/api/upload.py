import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import backend.data_analysis.agent as agent
from app.api.eda import _state_summary, _last_ai_message

router = APIRouter(prefix="/api/upload", tags=["upload"])

@router.post("")
async def save_uploaded_file(
        request: Request,
        file: UploadFile = File(...),
        use_eda: bool = False,
        session_id: str | None = None,
):
    upload_dir = request.app.state.upload_dir
    print(f"--- 收到上传请求: {file.filename} ---", flush=True)

    try:
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)

        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        print(f"✅ 文件已保存至: {file_path}")

        # 预设响应
        response_payload = {"file_name": file.filename, "status": "success"}

        if use_eda:
            # 确保 LLM 已初始化（自动从 .env 或默认配置读取）
            if agent.llm is None:
                agent.init_llm()

            # 运行分析：这里一定要确保 thread_id 传给了 agent
            # 如果没有传入 session_id，建议手动生成一个，保证前后端 ID 统一
            active_thread_id = session_id or f"sid_{os.urandom(4).hex()}"

            state = agent.run_pipeline(file_path, thread_id=active_thread_id)

            # 组装 EDA 数据结构
            response_payload["eda"] = {
                "thread_id": active_thread_id,
                "message": _last_ai_message(state),
                "state": _state_summary(state)
            }
            response_payload["eda_thread_id"] = active_thread_id

        return response_payload

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))