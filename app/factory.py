"""FastAPI 应用工厂。

本模块负责组装配置、服务、接口路由和前端静态资源挂载。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.config import load_config
from app.services.llm_service import ChatService
from app.api.upload import router as upload_router
from app.api.eda import router as eda_router
from app.api.dbConnection import router as db_router

def create_app() -> FastAPI:
    # 创建应用对象，统一承载路由、中间件和全局状态。
    app = FastAPI(title="Role C Frontend Demo API")

    # 计算项目根目录，用于定位前端目录和辅助文件。
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root)

    # 把配置和服务放到 app.state，路由处理函数可按需取用。
    app.state.upload_dir = str(project_root / "backend" / "dataset")
    app.state.config = config
    app.state.chat_service = ChatService(config)
    app.state.sql_apps = {}

    # 开发阶段放开跨域，便于本地调试。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册聊天相关接口（/api/chat/*）。
    app.include_router(chat_router)
    # 文件上传接口
    app.include_router(upload_router)
    # EDA 报告代理接口
    app.include_router(eda_router)
    #数据库连接
    app.include_router(db_router)

    # 把前端静态目录挂到根路径，前后端使用同一域名与端口。
    if config.frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(config.frontend_dir), html=True), name="frontend")

    return app
