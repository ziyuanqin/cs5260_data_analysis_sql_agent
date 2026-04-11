from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from langgraph.graph import StateGraph as _StateGraph  # noqa: F401

    HAS_LANGGRAPH = True
except ModuleNotFoundError:
    HAS_LANGGRAPH = False

from app.agent.general_graph import build_general_agent_graph


@unittest.skipUnless(HAS_LANGGRAPH, "langgraph is required for graph unit tests")
class GeneralGraphTests(unittest.TestCase):
    def test_review_retry_cap_finishes_loop(self):
        calls: list[str] = []

        def run_stage(stage: str, _session_id: str, prompt: str):
            calls.append(stage)
            if stage == "planner":
                if "Return JSON array only" in prompt:
                    return ('[{"task":"执行唯一步骤"}]', "planner-model")
                return ("修订后的唯一步骤", "planner-model")
            if stage == "executor":
                return ("执行结果不充分", "executor-model")
            if stage == "reviewer":
                return ("FAIL: 结果不满足目标", "reviewer-model")
            if stage == "summarizer":
                return ("已在最大重试后给出最终总结", "summarizer-model")
            return ("", "unknown-model")

        graph = build_general_agent_graph(run_stage)
        initial_state = {
            "question": "请给出执行结果",
            "session_id": "s-graph",
            "task_list": [],
            "current_step_index": 0,
            "review_feedback": "",
            "review_retries": 0,
            "max_review_retries": 1,
            "review_decision": "next_step",
            "response": "",
            "selected_models": {},
        }

        result = graph.invoke(initial_state)
        self.assertEqual(result["response"], "已在最大重试后给出最终总结")
        self.assertEqual(calls.count("reviewer"), 2)
        self.assertEqual(calls.count("planner"), 2)
        self.assertEqual(calls.count("executor"), 2)


if __name__ == "__main__":
    unittest.main()
