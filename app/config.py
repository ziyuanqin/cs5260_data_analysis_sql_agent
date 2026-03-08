import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    api_base_url: str
    api_key: str | None
    model_name: str
    request_timeout: int
    history_max_messages: int
    frontend_dir: Path


def _load_api_key(project_root: Path) -> str | None:
    key_from_env = os.getenv("YUNWU_API_KEY")
    if key_from_env:
        return key_from_env

    test_api_path = project_root / "utils" / "test_api.py"
    if not test_api_path.exists():
        return None

    try:
        content = test_api_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r"^key\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)
    return match.group(1) if match else None


def load_config(project_root: Path) -> AppConfig:
    return AppConfig(
        api_base_url=os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1"),
        api_key=_load_api_key(project_root),
        model_name=os.getenv("YUNWU_MODEL", "gpt-5.2"),
        request_timeout=int(os.getenv("YUNWU_TIMEOUT", "100")),
        history_max_messages=int(os.getenv("YUNWU_HISTORY_MAX_MESSAGES", "20")),
        frontend_dir=project_root / "frontend",
    )
