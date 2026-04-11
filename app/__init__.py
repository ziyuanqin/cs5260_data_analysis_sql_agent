"""应用包导出入口。

导出 create_app，方便 main.py 使用 `from app import create_app`。
"""

def create_app():
    # 延迟导入，避免在纯单元测试场景下强依赖 FastAPI。
    from .factory import create_app as _create_app

    return _create_app()

__all__ = ["create_app"]
