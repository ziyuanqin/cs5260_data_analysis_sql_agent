from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_analysis import agent


class DataAnalysisConfigTests(unittest.TestCase):
    def _write_config(self, tmp_dir: str, payload: dict) -> str:
        path = Path(tmp_dir) / "app_config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_env_credentials_take_precedence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "openai_api_key": "cfg-openai",
                    "deepseek_api_key": "cfg-deepseek",
                },
            )
            env = {
                "APP_CONFIG_PATH": config_path,
                "OPENAI_API_KEY": "env-openai",
                "DEEPSEEK_API_KEY": "env-deepseek",
                "EDA_USE_APP_CONFIG": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                resolved = agent._resolve_llm_credentials()

        self.assertEqual(resolved["openai_key"], "env-openai")
        self.assertEqual(resolved["openai_key_source"], "env")
        self.assertEqual(resolved["deepseek_key"], "env-deepseek")
        self.assertEqual(resolved["deepseek_key_source"], "env")

    def test_blank_env_falls_back_to_app_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "openai_api_key": "cfg-openai",
                    "deepseek_api_key": "cfg-deepseek",
                },
            )
            env = {
                "APP_CONFIG_PATH": config_path,
                "OPENAI_API_KEY": "   ",
                "DEEPSEEK_API_KEY": "",
                "EDA_USE_APP_CONFIG": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                resolved = agent._resolve_llm_credentials()

        self.assertEqual(resolved["openai_key"], "cfg-openai")
        self.assertEqual(resolved["openai_key_source"], "app_config")
        self.assertEqual(resolved["deepseek_key"], "cfg-deepseek")
        self.assertEqual(resolved["deepseek_key_source"], "app_config")

    def test_blank_env_without_opt_in_does_not_fallback_to_app_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(
                tmp_dir,
                {
                    "openai_api_key": "cfg-openai",
                    "deepseek_api_key": "cfg-deepseek",
                },
            )
            env = {
                "APP_CONFIG_PATH": config_path,
                "OPENAI_API_KEY": "   ",
                "DEEPSEEK_API_KEY": "",
                "EDA_USE_APP_CONFIG": "0",
            }
            with patch.dict(os.environ, env, clear=False):
                resolved = agent._resolve_llm_credentials()

        self.assertIsNone(resolved["openai_key"])
        self.assertEqual(resolved["openai_key_source"], "none")
        self.assertIsNone(resolved["deepseek_key"])
        self.assertEqual(resolved["deepseek_key_source"], "none")


if __name__ == "__main__":
    unittest.main()
