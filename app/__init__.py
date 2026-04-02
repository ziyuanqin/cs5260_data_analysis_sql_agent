"""应用包导出入口。

导出 create_app，方便 main.py 使用 `from app import create_app`。
"""

from .factory import create_app

__all__ = ["create_app"]
