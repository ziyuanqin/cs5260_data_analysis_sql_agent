"""配置加载模块。

优先级：
1) 配置文件 app_config.json
2) 环境变量
3) 代码默认值
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """应用运行时配置对象。"""

    # 默认使用的后端提供方名称。
    default_provider: str

    # OpenAI 兼容后端配置。
    openai_api_base_url: str
    openai_api_key: str | None
    openai_model_name: str
    deepseek_api_base_url: str
    deepseek_api_key: str | None

    # Hugging Face 后端配置。
    huggingface_api_url: str
    huggingface_api_key: str | None
    huggingface_model_name: str

    # 本地 HTTP 后端配置。
    local_api_url: str
    local_model_name: str

    # General 模式前端模型别名映射。
    general_openai_provider: str
    general_openai_model_name: str
    general_deepseek_provider: str
    general_deepseek_model_name: str

    # Agent 分阶段模型配置。
    agent_intent_model_name: str
    agent_planner_model_name: str
    agent_executor_model_name: str
    agent_reviewer_model_name: str
    agent_summarizer_model_name: str
    agent_max_review_retries: int
    fallback_model_name: str

    # 会话级预算控制。
    session_token_budget: int
    session_cost_budget_usd: float
    default_price_per_1k_tokens_usd: float
    model_price_per_1k_tokens_usd: dict[str, float]

    # General 模式路由阈值。
    planning_keyword_routing_enabled: bool
    planning_long_input_chars: int
    planning_multi_question_marks: int
    planning_short_direct_chars: int

    request_timeout: int
    history_max_messages: int
    memory_auto_compress_enabled: bool
    memory_keep_recent_messages: int
    memory_summary_max_chars: int
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


def _pick_bool(config_data: dict, key: str, env_name: str | None, default: bool) -> bool:
    """优先从配置文件读取布尔值，再读取环境变量，最后使用默认值。"""

    value = config_data.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False

    if env_name:
        env_value = os.getenv(env_name)
        if isinstance(env_value, str):
            lowered = env_value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False

    return default


def _pick_float(config_data: dict, key: str, env_name: str | None, default: float) -> float:
    """优先从配置文件读取浮点数，再读取环境变量，最后使用默认值。"""

    value = config_data.get(key)
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass

    if env_name:
        env_value = os.getenv(env_name)
        if isinstance(env_value, str) and env_value.strip():
            try:
                return float(env_value)
            except ValueError:
                pass

    return default


def _pick_price_map(config_data: dict, key: str) -> dict[str, float]:
    """读取每 1k token 价格映射。"""

    raw = config_data.get(key)
    if not isinstance(raw, dict):
        return {}

    result: dict[str, float] = {}
    for model_name, price in raw.items():
        if not isinstance(model_name, str) or not model_name.strip():
            continue
        if not isinstance(price, (float, int, str)):
            continue
        try:
            normalized = float(price)
        except ValueError:
            continue
        if normalized < 0:
            continue
        result[model_name.strip()] = normalized
    return result


def load_config(project_root: Path) -> AppConfig:
    """构建配置对象，并提供适合本地开发的默认值。"""

    config_data = _load_json_config(project_root)

    openai_api_key_from_file = config_data.get("openai_api_key")
    openai_api_key = (
        openai_api_key_from_file
        if isinstance(openai_api_key_from_file, str) and openai_api_key_from_file.strip()
        else (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_COMPAT_API_KEY") or None)
    )

    deepseek_api_key_from_file = config_data.get("deepseek_api_key")
    deepseek_api_key = (
        deepseek_api_key_from_file
        if isinstance(deepseek_api_key_from_file, str) and deepseek_api_key_from_file.strip()
        else (os.getenv("DEEPSEEK_API_KEY") or None)
    )

    huggingface_api_key_from_file = config_data.get("huggingface_api_key")
    huggingface_api_key = (
        huggingface_api_key_from_file
        if isinstance(huggingface_api_key_from_file, str) and huggingface_api_key_from_file.strip()
        else (os.getenv("HF_API_TOKEN") or None)
    )

    openai_model_name = _pick_str(
        config_data,
        "openai_model_name",
        "OPENAI_COMPAT_MODEL",
        os.getenv("YUNWU_MODEL", "gpt-5.2"),
    )
    huggingface_model_name = _pick_str(
        config_data,
        "huggingface_model_name",
        "HF_MODEL_ID",
        "Qwen/Qwen3.5-9B:together",
    )

    default_provider = _pick_str(config_data, "default_provider", "LLM_DEFAULT_PROVIDER", "openai_compatible")
    stage_default_model = huggingface_model_name if default_provider == "huggingface" else openai_model_name

    fallback_model_name = _pick_str(
        config_data,
        "fallback_model_name",
        "FALLBACK_MODEL_NAME",
        stage_default_model,
    )

    return AppConfig(
        default_provider=default_provider,
        openai_api_base_url=_pick_str(
            config_data,
            "openai_api_base_url",
            "OPENAI_COMPAT_BASE_URL",
            os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1"),
        ),
        openai_api_key=openai_api_key,
        openai_model_name=openai_model_name,
        deepseek_api_base_url=_pick_str(
            config_data,
            "deepseek_api_base_url",
            "DEEPSEEK_API_BASE_URL",
            "https://api.deepseek.com/v1",
        ),
        deepseek_api_key=deepseek_api_key,
        huggingface_api_url=_pick_str(
            config_data,
            "huggingface_api_url",
            "HF_API_URL",
            "https://router.huggingface.co/v1/chat/completions",
        ),
        huggingface_api_key=huggingface_api_key,
        huggingface_model_name=huggingface_model_name,
        local_api_url=_pick_str(config_data, "local_api_url", "LOCAL_LLM_API_URL", "http://127.0.0.1:9000/chat"),
        local_model_name=_pick_str(config_data, "local_model_name", "LOCAL_LLM_MODEL", "local-default-model"),
        general_openai_provider=_pick_str(
            config_data,
            "general_openai_provider",
            "GENERAL_OPENAI_PROVIDER",
            "openai_compatible",
        ),
        general_openai_model_name=_pick_str(
            config_data,
            "general_openai_model_name",
            "GENERAL_OPENAI_MODEL",
            openai_model_name,
        ),
        general_deepseek_provider=_pick_str(
            config_data,
            "general_deepseek_provider",
            "GENERAL_DEEPSEEK_PROVIDER",
            "openai_compatible",
        ),
        general_deepseek_model_name=_pick_str(
            config_data,
            "general_deepseek_model_name",
            "GENERAL_DEEPSEEK_MODEL",
            "deepseek-chat",
        ),
        agent_intent_model_name=_pick_str(
            config_data,
            "agent_intent_model_name",
            "AGENT_INTENT_MODEL",
            stage_default_model,
        ),
        agent_planner_model_name=_pick_str(
            config_data,
            "agent_planner_model_name",
            "AGENT_PLANNER_MODEL",
            stage_default_model,
        ),
        agent_executor_model_name=_pick_str(
            config_data,
            "agent_executor_model_name",
            "AGENT_EXECUTOR_MODEL",
            stage_default_model,
        ),
        agent_reviewer_model_name=_pick_str(
            config_data,
            "agent_reviewer_model_name",
            "AGENT_REVIEWER_MODEL",
            stage_default_model,
        ),
        agent_summarizer_model_name=_pick_str(
            config_data,
            "agent_summarizer_model_name",
            "AGENT_SUMMARIZER_MODEL",
            stage_default_model,
        ),
        agent_max_review_retries=_pick_int(
            config_data,
            "agent_max_review_retries",
            "AGENT_MAX_REVIEW_RETRIES",
            2,
        ),
        fallback_model_name=fallback_model_name,
        session_token_budget=_pick_int(
            config_data,
            "session_token_budget",
            "SESSION_TOKEN_BUDGET",
            1200000,
        ),
        session_cost_budget_usd=_pick_float(
            config_data,
            "session_cost_budget_usd",
            "SESSION_COST_BUDGET_USD",
            0.8,
        ),
        default_price_per_1k_tokens_usd=_pick_float(
            config_data,
            "default_price_per_1k_tokens_usd",
            "DEFAULT_PRICE_PER_1K_TOKENS_USD",
            0.002,
        ),
        model_price_per_1k_tokens_usd=_pick_price_map(config_data, "model_price_per_1k_tokens_usd"),
        planning_keyword_routing_enabled=_pick_bool(
            config_data,
            "planning_keyword_routing_enabled",
            "PLANNING_KEYWORD_ROUTING_ENABLED",
            True,
        ),
        planning_long_input_chars=_pick_int(
            config_data,
            "planning_long_input_chars",
            "PLANNING_LONG_INPUT_CHARS",
            180,
        ),
        planning_multi_question_marks=_pick_int(
            config_data,
            "planning_multi_question_marks",
            "PLANNING_MULTI_QUESTION_MARKS",
            2,
        ),
        planning_short_direct_chars=_pick_int(
            config_data,
            "planning_short_direct_chars",
            "PLANNING_SHORT_DIRECT_CHARS",
            20,
        ),
        request_timeout=_pick_int(config_data, "request_timeout", "YUNWU_TIMEOUT", 100),
        history_max_messages=_pick_int(config_data, "history_max_messages", "YUNWU_HISTORY_MAX_MESSAGES", 20),
        memory_auto_compress_enabled=_pick_bool(
            config_data,
            "memory_auto_compress_enabled",
            "MEMORY_AUTO_COMPRESS_ENABLED",
            True,
        ),
        memory_keep_recent_messages=_pick_int(
            config_data,
            "memory_keep_recent_messages",
            "MEMORY_KEEP_RECENT_MESSAGES",
            12,
        ),
        memory_summary_max_chars=_pick_int(
            config_data,
            "memory_summary_max_chars",
            "MEMORY_SUMMARY_MAX_CHARS",
            3000,
        ),
        frontend_dir=project_root / "frontend",
    )
