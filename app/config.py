"""配置加载模块。

优先级：
1) 配置文件 app_config.json
2) 环境变量
3) 代码默认值
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    """应用运行时配置对象。"""

    # 默认使用的后端提供方名称。
    default_provider: str

    # OpenAI 兼容后端配置。
    openai_api_base_url: str
    openai_api_key: str | None
    openai_model_name: str

    # 本地 HTTP 后端配置。
    local_api_url: str
    local_model_name: str

    request_timeout: int
    history_max_messages: int
    eda_api_base_url: str
    frontend_dir: Path


def _load_json_config(project_root: Path) -> dict:
    """读取项目根目录配置文件 app_config.json。"""

    config_path = project_root / "app_config.json"
    if not config_path.exists():
        return {}

    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _pick_str(config_data: dict, key: str, env_name: str | None, default: str) -> str:
    """优先从配置文件读取字符串，再读取环境变量，最后使用默认值。"""

    value = config_data.get(key)
    if isinstance(value, str) and value.strip():
        return value

    if env_name:
        env_value = os.getenv(env_name)
        if isinstance(env_value, str) and env_value.strip():
            return env_value

    return default


def _pick_int(config_data: dict, key: str, env_name: str | None, default: int) -> int:
    """优先从配置文件读取整数，再读取环境变量，最后使用默认值。"""

    value = config_data.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass

    if env_name:
        env_value = os.getenv(env_name)
        if isinstance(env_value, str) and env_value.strip():
            try:
                return int(env_value)
            except ValueError:
                pass

    return default


def load_config(project_root: Path) -> AppConfig:
    """构建配置对象，并提供适合本地开发的默认值。"""

    config_data = _load_json_config(project_root)
    openai_api_key_from_file = config_data.get("openai_api_key")
    openai_api_key = openai_api_key_from_file if isinstance(openai_api_key_from_file, str) and openai_api_key_from_file.strip() else None

    return AppConfig(
        default_provider=_pick_str(config_data, "default_provider", "LLM_DEFAULT_PROVIDER", "openai_compatible"),
        openai_api_base_url=_pick_str(
            config_data,
            "openai_api_base_url",
            "OPENAI_COMPAT_BASE_URL",
            os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1"),
        ),
        openai_api_key=openai_api_key,
        openai_model_name=_pick_str(
            config_data,
            "openai_model_name",
            "OPENAI_COMPAT_MODEL",
            os.getenv("YUNWU_MODEL", "gpt-5.2"),
        ),
        local_api_url=_pick_str(config_data, "local_api_url", "LOCAL_LLM_API_URL", "http://127.0.0.1:9000/chat"),
        local_model_name=_pick_str(config_data, "local_model_name", "LOCAL_LLM_MODEL", "local-default-model"),
        request_timeout=_pick_int(config_data, "request_timeout", "YUNWU_TIMEOUT", 100),
        history_max_messages=_pick_int(config_data, "history_max_messages", "YUNWU_HISTORY_MAX_MESSAGES", 20),
        eda_api_base_url=_pick_str(config_data, "eda_api_base_url", "EDA_API_BASE_URL", "http://127.0.0.1:8002"),
        frontend_dir=project_root / "frontend",
    )
