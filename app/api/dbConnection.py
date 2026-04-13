from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from backend.SQLagent.main import get_sql_graph_app
from app.utils.crypto import PUBLIC_KEY, decrypt_password

router = APIRouter(prefix="/api/db", tags=["database"])

class DBConnectRequest(BaseModel):
    session_id: str
    host: str
    port: int | str = 3306
    user: str
    password: str
    database: str


@router.get("/public-key")
async def get_public_key():
    return {"public_key": PUBLIC_KEY}

@router.post("/connect")
async def connect_db(request: Request, req: DBConnectRequest):
    print(f"--- Try to connect to the database for Session {req.session_id} : {req.database} ---")
    try:
        # --- 核心改动：解密密码 ---
        # 此时 req.password 是前端传来的长串 RSA 密文
        real_password = decrypt_password(req.password)

        # 1. 使用解密后的真实密码构造连接 URL
        mysql_url = f"mysql+pymysql://{req.user}:{real_password}@{req.host}:{req.port}/{req.database}"

        # 2. 生成实例 (逻辑保持不变)
        sql_app = get_sql_graph_app(db_type="mysql", mysql_url=mysql_url)

        # 3. 存储到全局状态
        if not hasattr(request.app.state, "sql_apps"):
            request.app.state.sql_apps = {}

        request.app.state.sql_apps[req.session_id] = sql_app

        print(f"✅ Connection succeeded，Session {req.session_id} password has been securely decrypted and verified")
        return {"ok": True, "message": f"Successfully connected to {req.database}"}

    except ValueError as ve:
        # 处理解密相关的安全错误
        print(f"🔒 Security verification failed: {str(ve)}")
        raise HTTPException(status_code=403, detail="Password decryption failed. The transmission may have been interfered.")
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Database connection failed: {str(e)}")

@router.post("/disconnect")
async def disconnect_db(req: dict, request: Request):
    """断开指定 session 的数据库连接并清理资源"""
    session_id = req.get("session_id")
    if not session_id:
        return {"ok": False, "msg": "Lack session_id"}

    if hasattr(request.app.state, "sql_apps"):
        # 1. 弹出实例
        sql_app = request.app.state.sql_apps.pop(session_id, None)

        if sql_app:
            # 2. 如果你的 sql_app 暴露了 engine，可以主动 dispose
            # 这一步能立即释放 MySQL 的连接池资源
            try:
                # 假设你的 graph_app 里存了 engine 或者可以通过某种方式访问
                # 如果暂时拿不到 engine，pop 掉实例也会让 GC 回收连接
                print(f"--- [Session {session_id}] database connection has been disconnected and cleaned up ---")
            except Exception as e:
                print(f"Failed to clean up connection pool: {e}")

            return {"ok": True, "msg": "Connection dropped."}

    return {"ok": False, "msg": "Active connection not found."}
