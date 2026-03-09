"""SSE 文本帧格式化工具。"""

import json
from typing import Any


def sse_data(payload: dict[str, Any]) -> str:
    """将单个事件字典编码为 SSE 帧字符串。

    本项目使用的帧格式：
        data: <json>\n\n
    """

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
