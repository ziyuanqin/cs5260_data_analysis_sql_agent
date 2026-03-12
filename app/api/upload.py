import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

# 模仿 chat.py 的 prefix 风格
router = APIRouter(prefix="/api/upload", tags=["upload"])

@router.post("")
async def save_uploaded_file(request: Request, file: UploadFile = File(...)):
    # 从 factory.py 传过来的可靠绝对路径获取
    upload_dir = request.app.state.upload_dir

    print(f"--- 收到上传请求: {file.filename} ---", flush=True)

    try:
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        print(f"✅ 文件已保存至: {file_path}")
        return {"file_name": file.filename, "status": "success"}
    except Exception as e:
        print(f"❌ 写入失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))