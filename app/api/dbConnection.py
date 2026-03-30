from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from backend.SQLagent.main import get_sql_graph_app

router = APIRouter(prefix="/api/db", tags=["database"])

class DBConnectRequest(BaseModel):
    session_id: str
    host: str
    port: int | str = 3306
    user: str
    password: str
    database: str

@router.post("/connect")
async def connect_db(request: Request, req: DBConnectRequest):
    print(f"--- 尝试为 Session {req.session_id} 连接数据库: {req.database} ---")
    try:
        # 1. 构造连接 URL
        mysql_url = f"mysql+pymysql://{req.user}:{req.password}@{req.host}:{req.port}/{req.database}"

        # 2. 生成针对该 MySQL 连接的模式 B 实例
        sql_app = get_sql_graph_app(db_type="mysql", mysql_url=mysql_url)

        # 3. 存储到全局状态
        if not hasattr(request.app.state, "sql_apps"):
            request.app.state.sql_apps = {}

        # 绑定到当前会话
        request.app.state.sql_apps[req.session_id] = sql_app

        print(f"✅ 连接成功，Session {req.session_id} 现在优先使用数据库表")
        return {"ok": True, "message": f"成功连接至 {req.database}"}

    except Exception as e:
        print(f"❌ 连接失败: {str(e)}")
        # 返回 400 会让前端 JavaScript 的 try...catch 捕获到错误并 alert
        raise HTTPException(status_code=400, detail=f"数据库连接失败: {str(e)}")

@router.post("/disconnect")
async def disconnect_db(req: dict, request: Request):
    """断开指定 session 的数据库连接并清理资源"""
    session_id = req.get("session_id")
    if not session_id:
        return {"ok": False, "msg": "缺少 session_id"}

    if hasattr(request.app.state, "sql_apps"):
        # 1. 弹出实例
        sql_app = request.app.state.sql_apps.pop(session_id, None)

        if sql_app:
            # 2. 如果你的 sql_app 暴露了 engine，可以主动 dispose
            # 这一步能立即释放 MySQL 的连接池资源
            try:
                # 假设你的 graph_app 里存了 engine 或者可以通过某种方式访问
                # 如果暂时拿不到 engine，pop 掉实例也会让 GC 回收连接
                print(f"--- [Session {session_id}] 数据库连接已断开并清理 ---")
            except Exception as e:
                print(f"清理连接池失败: {e}")

            return {"ok": True, "msg": "连接已断开"}

    return {"ok": False, "msg": "未找到活跃连接"}