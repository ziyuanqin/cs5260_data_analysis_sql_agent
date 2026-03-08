from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.config import load_config
from app.services.llm_service import ChatService


def create_app() -> FastAPI:
    app = FastAPI(title="Role C Frontend Demo API")

    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root)

    app.state.config = config
    app.state.chat_service = ChatService(config)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)

    if config.frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(config.frontend_dir), html=True), name="frontend")

    return app
