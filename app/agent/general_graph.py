"""General-mode LangGraph workflow for the Manus-like loop."""

from __future__ import annotations

import json
from typing import Any, Callable, Literal, TypedDict

try:
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    END = START = object()  # type: ignore[assignment]
    StateGraph = Any  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR = exc


class TaskItem(TypedDict):
    """Single planned task."""

    task: str
    status: str
    result: str


class GeneralAgentState(TypedDict):
    """State contract shared by all general-mode agent nodes."""

    question: str
    session_id: str
    task_list: list[TaskItem]
    current_step_index: int
    review_feedback: str
    review_retries: int
    max_review_retries: int
    review_decision: Literal["next_step", "replan", "finish"]
    response: str
    selected_models: dict[str, str]
    last_tool_meta: dict[str, Any] | None


def _extract_json_array(text: str) -> list[dict]:
    """Best-effort extraction of a JSON array from model output."""

    if not text:
        return []

    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end < start:
        return []

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _normalize_tasks(raw_items: list[dict], question: str) -> list[TaskItem]:
    """Convert arbitrary parsed JSON items into stable task items."""

    tasks: list[TaskItem] = []
    for item in raw_items:
        task_text = item.get("task")
        if not isinstance(task_text, str):
            continue
        normalized = task_text.strip()
        if not normalized:
            continue
        tasks.append({"task": normalized, "status": "pending", "result": ""})

    if tasks:
        return tasks

    # Fallback: keep the workflow alive with one direct step.
    return [{"task": f"Directly answer the user request: {question}", "status": "pending", "result": ""}]


RunStageResult = tuple[str, str] | tuple[str, str, dict[str, Any] | None]


def build_general_agent_graph(
    run_stage: Callable[[str, str, str], RunStageResult],
) -> Any:
    """Build and compile the general-mode graph.

    `run_stage(stage, session_id, prompt)` returns `(text, selected_model)` or
    `(text, selected_model, tool_meta)` when tool usage occurs.
    """

    def planner_node(state: GeneralAgentState) -> dict:
        selected_models = dict(state.get("selected_models", {}))
        tasks = list(state.get("task_list", []))
        session_id = state["session_id"]

        if not tasks:
            prompt = (
                "You are a planner. Break the request into 1-5 concrete steps.\n"
                "Return JSON array only. Each item must be an object with key 'task'.\n"
                f"Request: {state['question']}"
            )
            response_text, model_name = run_stage("planner", session_id, prompt)
            selected_models["planner"] = model_name
            parsed = _extract_json_array(response_text)
            normalized_tasks = _normalize_tasks(parsed, question=state["question"])
            return {
                "task_list": normalized_tasks,
                "current_step_index": 0,
                "review_feedback": "",
                "review_decision": "next_step",
                "selected_models": selected_models,
            }

        idx = min(max(state.get("current_step_index", 0), 0), max(len(tasks) - 1, 0))
        current_task = tasks[idx]["task"]
        feedback = state.get("review_feedback", "")
        prompt = (
            "Rewrite the step to be concrete and executable.\n"
            "Return one sentence only, without bullets.\n"
            f"Current step: {current_task}\n"
            f"Reviewer feedback: {feedback}"
        )
        updated_task, model_name = run_stage("planner", session_id, prompt)
        selected_models["planner"] = model_name
        rewritten = updated_task.strip() or current_task
        updated_tasks = list(tasks)
        updated_tasks[idx] = {"task": rewritten, "status": "pending", "result": ""}
        return {
            "task_list": updated_tasks,
            "review_feedback": "",
            "review_decision": "next_step",
            "selected_models": selected_models,
            "current_step_index": idx,
        }

    def executor_node(state: GeneralAgentState) -> dict:
        selected_models = dict(state.get("selected_models", {}))
        tasks = list(state.get("task_list", []))
        if not tasks:
            return {}

        idx = min(max(state.get("current_step_index", 0), 0), len(tasks) - 1)
        task_text = tasks[idx]["task"]
        session_id = state["session_id"]
        prompt = (
            "Execute this step and produce a concise, useful output.\n"
            "Do not ask follow-up questions.\n"
            f"Step: {task_text}"
        )
        tool_meta = None
        result = run_stage("executor", session_id, prompt)
        if isinstance(result, tuple) and len(result) == 3:
            result_text, model_name, tool_meta = result
        else:
            result_text, model_name = result  # type: ignore[misc]
        selected_models["executor"] = model_name

        updated_tasks = list(tasks)
        updated_tasks[idx] = {
            "task": task_text,
            "status": "done",
            "result": result_text.strip(),
        }
        return {
            "task_list": updated_tasks,
            "selected_models": selected_models,
            "last_tool_meta": tool_meta,
            "current_step_index": idx,
        }

    def reviewer_node(state: GeneralAgentState) -> dict:
        selected_models = dict(state.get("selected_models", {}))
        tasks = list(state.get("task_list", []))
        session_id = state["session_id"]

        if not tasks:
            return {
                "review_feedback": "No task generated; finishing with direct summary.",
                "review_decision": "finish",
                "selected_models": selected_models,
                "task_list": tasks,
                "current_step_index": state.get("current_step_index", 0),
            }

        idx = min(max(state.get("current_step_index", 0), 0), len(tasks) - 1)
        task_text = tasks[idx]["task"]
        task_result = tasks[idx].get("result", "")
        prompt = (
            "Review whether the result satisfies the step objective.\n"
            "Reply exactly in one line:\n"
            "- PASS\n"
            "- FAIL: <reason>\n"
            f"Step objective: {task_text}\n"
            f"Step result: {task_result}"
        )
        review_text, model_name = run_stage("reviewer", session_id, prompt)
        selected_models["reviewer"] = model_name
        review_text = review_text.strip() or "FAIL: empty review output"

        current_retry = int(state.get("review_retries", 0))
        max_retry = max(0, int(state.get("max_review_retries", 0)))
        upper = review_text.upper()
        passed = "PASS" in upper and "FAIL" not in upper

        if passed:
            if idx + 1 < len(tasks):
                return {
                    "current_step_index": idx + 1,
                    "review_feedback": "PASS",
                    "review_retries": 0,
                    "review_decision": "next_step",
                    "selected_models": selected_models,
                    "task_list": tasks,
                }
            return {
                "review_feedback": "PASS",
                "review_retries": 0,
                "review_decision": "finish",
                "selected_models": selected_models,
                "task_list": tasks,
                "current_step_index": idx,
            }

        next_retry = current_retry + 1
        if next_retry > max_retry:
            return {
                "review_feedback": (
                    f"{review_text} | Max review retries reached ({max_retry}). "
                    "Proceeding to summarization."
                ),
                "review_retries": current_retry,
                "review_decision": "finish",
                "selected_models": selected_models,
                "task_list": tasks,
                "current_step_index": idx,
            }

        return {
            "review_feedback": review_text,
            "review_retries": next_retry,
            "review_decision": "replan",
            "selected_models": selected_models,
            "task_list": tasks,
            "current_step_index": idx,
        }

    def summarizer_node(state: GeneralAgentState) -> dict:
        selected_models = dict(state.get("selected_models", {}))
        session_id = state["session_id"]
        tasks = list(state.get("task_list", []))
        lines = []
        for index, item in enumerate(tasks, start=1):
            lines.append(f"{index}. Step: {item['task']}\nResult: {item.get('result', '')}")
        digest = "\n\n".join(lines) if lines else "(no executed task output)"

        prompt = (
            "Create the final user-facing answer based on executed steps.\n"
            "Keep it concise and practical.\n"
            f"Original request: {state['question']}\n"
            f"Execution log:\n{digest}"
        )
        final_text, model_name = run_stage("summarizer", session_id, prompt)
        selected_models["summarizer"] = model_name
        return {"response": final_text.strip(), "selected_models": selected_models}

    def review_router(state: GeneralAgentState) -> str:
        return state.get("review_decision", "finish")

    if _LANGGRAPH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LangGraph is not installed. Install it with `pip install langgraph` before using general agent mode."
        ) from _LANGGRAPH_IMPORT_ERROR

    workflow = StateGraph(GeneralAgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        review_router,
        {
            "next_step": "executor",
            "replan": "planner",
            "finish": "summarizer",
        },
    )
    workflow.add_edge("summarizer", END)
    return workflow.compile()
