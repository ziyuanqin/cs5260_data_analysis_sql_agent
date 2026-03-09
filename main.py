"""ASGI 启动入口。

启动命令：
	uvicorn main:app --reload --host 0.0.0.0 --port 8001
"""

from app import create_app


# 在模块加载时创建应用实例，供 uvicorn 直接导入。
app = create_app()

