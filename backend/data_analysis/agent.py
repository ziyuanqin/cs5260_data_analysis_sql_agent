"""
agent.py — LangGraph pipeline wiring and public API.

Pipeline: load_dataset → infer_types → run_eda → END
Chat:     chat_router  → cleaning | custom_eda | general_chat

Public API:
  run_pipeline(file_path, thread_id)        — run steps 1-3
  handle_human_response(decision, thread_id) — resume after error
  run_chat(user_message, thread_id)         — cleaning / EDA / Q&A
"""

import os
import re
from pathlib import Path
import base64

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.data_analysis.state import AgentState
from backend.data_analysis.shared import log, _df_store, _to_serializable
from backend.data_analysis.processors import (
    DatasetUnderstanding,
    TypeInferencer,
    AutomatedEDA,
    DataCleaningEngine,
    CustomEDAEngine,
)
from backend.data_analysis.html_report import render_eda_html
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 优先使用项目 .env，避免被 shell 中旧变量污染导致鉴权失败。
load_dotenv(override=True)

# Uncomment this line if make it compulsory for user to provide API key, by default the API key will be set in env variable
# llm: ChatOpenAI | None = None
llm = ChatOpenAI(model="deepseek-chat", temperature=0)

def init_llm(api_key: str) -> None:
    """Initialise (or re-initialise) the LLM with the given OpenAI API key."""
    global llm
    #加
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="deepseek-chat", temperature=0, openai_api_key=api_key)
    log.info("[init_llm] LLM ready (deepseek-chat)")

def _require_llm():
    if llm is None:
        raise RuntimeError("LLM not initialised. Call init_llm(api_key) first or POST /api-key.")

_understander = DatasetUnderstanding()
_inferencer   = TypeInferencer()
_eda_runner   = AutomatedEDA()
_cleaner      = DataCleaningEngine()
_custom_eda   = CustomEDAEngine()

def _err(state: AgentState, msg: str, etype: str) -> AgentState:
    log.error("[_err] %s (type=%s)", msg, etype)
    return {**state, "error": msg, "error_type": etype, "step": "error",
            "awaiting_human": True,
            "messages": [AIMessage(content=(
                f"⚠️ **{msg}**\n\n"
                "Options: `retry` · `skip` · `reupload /path/file.csv` · `abort`"
            ))]}


# ══════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════

def node_load_dataset(state: AgentState) -> AgentState:
    fp = state.get("file_path", "")
    log.info("[node_load_dataset] Loading: %s", fp)
    if not fp or not os.path.exists(fp):
        return _err(state, f"File not found: `{fp}`", "load")

    result = _understander.load_and_inspect(fp)
    if result["error"]:
        return _err(state, f"Failed to load file: {result['error']}", "load")

    df     = result["df"]
    schema = result["schema"]
    tname  = re.sub(r'[^a-z0-9_]', '_', Path(fp).stem.lower())
    log.info("[node_load_dataset] Loaded '%s': %d rows × %d cols",
             tname, schema["row_count"], schema["col_count"])
    log.info("[node_load_dataset] Dtypes: %s", schema["dtypes_summary"])

    # Store DataFrames outside state (msgpack cannot serialise them)
    _df_store[tname]              = df
    _df_store[tname + "_original"] = df.copy()

    return {**state,
            "df_key":          tname,
            "df_original_key": tname + "_original",
            "schema_info":     _to_serializable(schema),
            "table_name":      tname,
            "cleaning_log":    [],
            "error":           None,
            "error_type":      None,
            "awaiting_human":  False,
            "step":            "loaded",
            "messages": [AIMessage(content=(
                f"✅ **Loaded** `{tname}` — {schema['row_count']:,} rows × {schema['col_count']} cols\n"
                f"Columns: {', '.join(schema['column_names'][:12])}"
                + (" …" if schema['col_count'] > 12 else "")
            ))]}


def node_infer_types(state: AgentState) -> AgentState:
    df_key = state.get("df_key")
    df     = _df_store.get(df_key) if df_key else None
    log.info("[node_infer_types] df_key=%s, loaded=%s", df_key, df is not None)
    if df is None:
        return _err(state, "No DataFrame to inspect.", "infer")
    try:
        suggestions = _inferencer.infer(df, llm)
        log.info("[node_infer_types] %d suggestion(s) ready", len(suggestions))

        lines = []
        if suggestions:
            lines.append(f"**{len(suggestions)} dtype conversion(s) suggested:**")
            for col, info in suggestions.items():
                icon = "🟡" if info["confidence"] == "low" else "🟢"
                lines.append(f"  {icon} `{col}`: `{info['current']}` → `{info['suggested']}` — {info['reason']}")

        lines.append("\nYou can say things like:")
        lines.append('  - `"convert Age to int64"` · `"drop duplicates"` · `"fill NaN in Cabin with unknown"`')
        lines.append('  - `"skip"` to go straight to EDA')

        return {**state,
                "type_suggestions": _to_serializable(suggestions),
                "step":            "types_inferred",
                "error":           None,
                "error_type":      None,
                "awaiting_human":  False,
                "messages": [AIMessage(content="\n".join(lines) or "✅ All dtypes look correct.")]}
    except Exception as e:
        log.exception("[node_infer_types] Failed")
        return _err(state, f"Type inference failed: {e}", "infer")


def node_run_eda(state: AgentState) -> AgentState:
    df_key = state.get("df_key")
    df     = _df_store.get(df_key) if df_key else None
    log.info("[node_run_eda] df_key=%s, loaded=%s", df_key, df is not None)
    if df is None:
        return _err(state, "No DataFrame for EDA.", "eda")
    try:
        log.info("[node_run_eda] Running on %d rows × %d cols", len(df), len(df.columns))
        eda_report = _eda_runner.run(df)
        log.info("[node_run_eda] Done — %d quality issues, %d high correlations",
                 eda_report["data_quality"]["issue_count"],
                 len(eda_report["correlations"].get("high_correlations", [])))

        eda_html = render_eda_html(
            eda_report,
            state["schema_info"],
            state["table_name"],
            state.get("type_suggestions"),
            state.get("cleaning_log"),
        )
        ov = eda_report["overview"]
        return {**state,
                "eda_report":     _to_serializable(eda_report),
                "eda_html":       eda_html,
                "step":           "eda_complete",
                "error":          None,
                "error_type":     None,
                "awaiting_human": False,
                "messages": [AIMessage(content=(
                    f"✅ **EDA complete** — HTML report ready.\n"
                    f"{ov['rows']:,} rows × {ov['columns']} cols | "
                    f"{ov['total_missing']} missing | "
                    f"{eda_report['data_quality']['issue_count']} quality issues | "
                    f"{len(eda_report['correlations'].get('high_correlations', []))} high correlations\n\n"
                    "Now you can clean the data or ask questions."
                ))]}
    except Exception as e:
        log.exception("[node_run_eda] Failed")
        return _err(state, f"EDA failed: {e}", "eda")


def node_cleaning(state: AgentState) -> AgentState:
    _require_llm()
    df_key = state.get("df_key")
    df     = _df_store.get(df_key) if df_key else None
    si     = state.get("schema_info")
    log.info("[node_cleaning] df_key=%s, loaded=%s", df_key, df is not None)
    if df is None or si is None:
        return {**state, "messages": [AIMessage(content="⚠️ No dataset loaded.")]}

    last = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    log.info("[node_cleaning] Request: %s", last)
    result = _cleaner.apply(df, si, last, llm)

    if result.get("error"):
        log.error("[node_cleaning] Error: %s", result["error"])
        return {**state, "step": "clean_error", "awaiting_human": True,
                "error": result["error"], "error_type": "clean",
                "messages": [AIMessage(content=f"⚠️ Cleaning failed: {result['error']}\n`retry` · `skip`")]}

    new_df = result["df"]
    log.info("[node_cleaning] Applied. Rows: %d → %d", len(df), len(new_df))
    _df_store[df_key] = new_df  # update in-place; key stays the same

    cleaning_log = state.get("cleaning_log") or []
    if result.get("log"):
        cleaning_log = cleaning_log + [result["log"]]

    new_schema = _to_serializable(_understander._inspect(new_df))
    return {**state,
            "df_key":        df_key,
            "schema_info":   new_schema,
            "cleaning_log":  cleaning_log,
            "step":          "clean_complete",
            "awaiting_human": False,
            "messages": [AIMessage(content=result.get("reply", "✅ Cleaning applied."))]}


def node_handle_human_decision(state: AgentState) -> AgentState:
    d  = (state.get("human_decision") or "").strip().lower()
    et = state.get("error_type", "")
    log.info("[node_handle_human_decision] decision='%s' error_type='%s'", d, et)

    if d == "abort":
        return {**state, "step": "aborted", "awaiting_human": False,
                "messages": [AIMessage(content="Pipeline stopped. Upload a new file to restart.")]}
    if d == "skip":
        return {**state, "error": None, "error_type": None,
                "awaiting_human": False, "step": "skipped",
                "messages": [AIMessage(content="Skipped. Continuing.")]}
    if d == "retry":
        return {**state, "error": None, "error_type": None,
                "awaiting_human": False, "step": f"retry_{et}",
                "messages": [AIMessage(content=f"Retrying {et}…")]}
    if d.startswith("reupload"):
        parts    = d.split(maxsplit=1)
        new_path = parts[1].strip() if len(parts) > 1 else None
        if new_path:
            log.info("[node_handle_human_decision] Reupload → %s", new_path)
            return {**state,
                    "error": None, "error_type": None, "awaiting_human": False,
                    "step": "retry_load", "file_path": new_path,
                    "df_key": None, "df_original_key": None,
                    "eda_report": None, "eda_html": None, "cleaning_log": [],
                    "messages": [AIMessage(content=f"Loading `{new_path}`…")]}
        # No path provided — stay interrupted and ask again
        return {**state, "awaiting_human": True,
                "messages": [AIMessage(content="Please provide a path: `reupload /path/file.csv`")]}

    return {**state, "awaiting_human": True,
            "messages": [AIMessage(content="Reply with: `retry` · `skip` · `reupload /path.csv` · `abort`")]}


def _parse_save_path(msg: str, default: str) -> str:
    """
    Extract output path from messages like:
      'save as my_data.csv'  →  'my_data.csv'
      'save to /tmp/out.csv' →  '/tmp/out.csv'
      'save dataset'         →  default
    """
    import re
    m = re.search(r'(?:as|to)\s+(\S+\.csv)', msg, re.IGNORECASE)
    return m.group(1) if m else default


def _detect_special_intent(msg: str) -> str | None:
    """
    Keyword-based detection for rerun_eda and save_csv so we don't
    burn an LLM call on these simple commands.
    Returns a route label or None if no match.
    """
    lower = msg.strip().lower()
    rerun_kw = {"rerun eda", "re-run eda", "refresh eda", "regenerate eda",
                "rerun report", "refresh report", "update eda", "redo eda"}
    save_kw  = {"save", "export", "download", "save dataset", "save csv",
                "export csv", "save cleaned", "export cleaned"}
    if any(lower.startswith(k) or lower == k for k in rerun_kw):
        return "rerun_eda"
    if any(lower.startswith(k) or lower == k for k in save_kw):
        return "save_csv"
    return None


def node_chat_router(state: AgentState) -> AgentState:
    _require_llm()
    last      = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    df_key    = state.get("df_key")
    is_loaded = df_key is not None and df_key in _df_store
    has_eda   = state.get("eda_report") is not None
    log.info("[node_chat_router] msg='%s' is_loaded=%s has_eda=%s", last[:60], is_loaded, has_eda)

    # Check keyword-based special intents first (no LLM call needed)
    special = _detect_special_intent(last)
    if special == "rerun_eda" and is_loaded:
        log.info("[node_chat_router] Detected: rerun_eda")
        return {**state, "step": "route_rerun_eda"}
    if special == "save_csv" and is_loaded:
        log.info("[node_chat_router] Detected: save_csv")
        return {**state, "step": "route_save_csv"}

    resp  = llm.invoke([HumanMessage(content=(
        f'User said: "{last}"\nDataset loaded: {is_loaded}. EDA done: {has_eda}.\n'
        'Classify as ONE of: "custom_analysis" | "cleaning_op" | "general_chat"\n'
        "custom_analysis = plot/regression/kmeans request\n"
        "cleaning_op = drop/convert/rename/fill/remove operation\n"
        "Reply ONLY with the label."
    ))])
    label = resp.content.strip().lower()
    log.info("[node_chat_router] LLM label: '%s'", label)

    if "custom_analysis" in label and is_loaded:
        return {**state, "step": "route_custom"}
    if "cleaning_op" in label and is_loaded:
        return {**state, "step": "route_clean"}

    # General chat — answer directly
    schema_ctx = str(state.get("schema_info", {}).get("dtypes_summary", {})) if state.get("schema_info") else "none"
    cl  = state.get("cleaning_log") or []
    system_prompt = (
        "You are a data analysis assistant with memory of this conversation.\n"
        f"Dataset columns: {schema_ctx}\n"
        f"Cleaning ops applied: {cl}\n"
        "Answer concisely. If no dataset is loaded, guide the user to upload one."
    )
    history = [m for m in state["messages"] if not isinstance(m, SystemMessage)][-10:]
    if history and isinstance(history[0], HumanMessage):
        history = [HumanMessage(content=f"{system_prompt}\n\n{history[0].content}")] + history[1:]
    else:
        history = [HumanMessage(content=system_prompt)] + history
    ans = llm.invoke(history)
    return {**state, "step": "chat_answered",
            "messages": [AIMessage(content=ans.content)]}


def node_custom_eda(state: AgentState) -> AgentState:
    _require_llm()
    df_key = state.get("df_key")
    df     = _df_store.get(df_key) if df_key else None
    si     = state.get("schema_info")
    log.info("[node_custom_eda] df_key=%s, loaded=%s", df_key, df is not None)
    if df is None or si is None:
        return {**state, "messages": [AIMessage(content="⚠️ No dataset loaded.")]}

    last   = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    log.info("[node_custom_eda] Request: %s", last)
    result = _custom_eda.run(df, si, last, llm)

    if result.get("error"):
        log.error("[node_custom_eda] Error: %s", result["error"])
        return {**state, "custom_result": result, "step": "custom_error",
                "awaiting_human": True, "error": result["error"], "error_type": "custom",
                "messages": [AIMessage(content=f"⚠️ Analysis failed: {result['error']}\nTry rephrasing or `skip`.")]}

    log.info("[node_custom_eda] Done: type=%s", result.get("type"))
    txt   = result.get("text_result", "")
    reply = f"✅ **{result['type'].replace('_',' ').title()}** complete." + (f"\n\n{txt}" if txt else "")
    if result.get("plot_b64"):
        img_md = f"\n\n![plot](data:image/png;base64,{result['plot_b64']})"
    reply += img_md
    return {**state, "custom_result": result, "step": "custom_complete",
            "awaiting_human": False,
            "messages": [AIMessage(content=reply)]}


def node_rerun_eda(state: AgentState) -> AgentState:
    """Re-run EDA on the current (possibly cleaned) DataFrame and refresh the HTML report."""
    df_key = state.get("df_key")
    df     = _df_store.get(df_key) if df_key else None
    log.info("[node_rerun_eda] df_key=%s, loaded=%s", df_key, df is not None)
    if df is None:
        return {**state, "messages": [AIMessage(content="⚠️ No dataset loaded.")]}
    try:
        eda_report = _eda_runner.run(df)
        eda_html   = render_eda_html(
            eda_report,
            state["schema_info"],
            state["table_name"],
            state.get("type_suggestions"),
            state.get("cleaning_log"),
        )
        ov = eda_report["overview"]
        log.info("[node_rerun_eda] Done — %d quality issues", eda_report["data_quality"]["issue_count"])
        return {**state,
                "eda_report":     _to_serializable(eda_report),
                "eda_html":       eda_html,
                "step":           "eda_complete",
                "messages": [AIMessage(content=(
                    f"✅ **EDA refreshed** on current dataset.\n"
                    f"{ov['rows']:,} rows × {ov['columns']} cols | "
                    f"{ov['total_missing']} missing | "
                    f"{eda_report['data_quality']['issue_count']} quality issues | "
                    f"{len(eda_report['correlations'].get('high_correlations', []))} high correlations"
                ))]}
    except Exception as e:
        log.exception("[node_rerun_eda] Failed")
        return {**state, "messages": [AIMessage(content=f"⚠️ EDA rerun failed: {e}")]}


def node_save_csv(state: AgentState) -> AgentState:
    """Save the current DataFrame to a CSV file."""
    df_key = state.get("df_key")
    df     = _df_store.get(df_key) if df_key else None
    log.info("[node_save_csv] df_key=%s, loaded=%s", df_key, df is not None)
    if df is None:
        return {**state, "messages": [AIMessage(content="⚠️ No dataset loaded.")]}

    # Parse optional output path from message: "save as output.csv" or "save dataset"
    last     = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    out_path = _parse_save_path(last, default=f"{state.get('table_name', 'dataset')}_cleaned.csv")

    save_dir = "backend/dataset"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pure_filename = os.path.basename(out_path)
    internal_sql_path = os.path.join(save_dir, pure_filename)

    try:
        df.to_csv(out_path, index=False)
        if out_path != internal_sql_path:
            df.to_csv(internal_sql_path, index=False)
        log.info("[node_save_csv] Saved for user: %s | Saved for SQL: %s", out_path, internal_sql_path)
        return {**state, "step": "save_complete",
                "file_name": pure_filename,
                "messages": [AIMessage(content=(
                    f"✅ **Saved** cleaned dataset → `{out_path}`\n"
                    f"{len(df):,} rows × {len(df.columns)} cols\n\n"
                    f"You can click the button below to directly execute SQL queries on the table."
                ))]}
    except Exception as e:
        log.error("[node_save_csv] Failed: %s", e)
        return {**state, "messages": [AIMessage(content=f"⚠️ Save failed: {e}")]}


# ══════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════

def route_after_load(s):
    return "handle_human_decision" if s.get("awaiting_human") else "infer_types"

def route_after_infer(s):
    return "handle_human_decision" if s.get("awaiting_human") else "run_eda"

def route_after_eda(s):
    return "handle_human_decision" if s.get("awaiting_human") else END

def route_after_human(s):
    """Routes forward after handle_human_decision. Never loops back to itself."""
    step = s.get("step", "")
    if step == "aborted":     return END
    if step == "skipped":     return END
    if step == "retry_load":  return "load_dataset"
    if step == "retry_infer": return "infer_types"
    if step == "retry_eda":   return "run_eda"
    return END

def route_chat(s):
    step = s.get("step", "")
    if step == "route_custom":    return "custom_eda"
    if step == "route_clean":     return "cleaning"
    if step == "route_rerun_eda": return "rerun_eda"
    if step == "route_save_csv":  return "save_csv"
    return END


# ══════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════

def _build_pipeline_graph(checkpointer):
    g = StateGraph(AgentState)
    for name, fn in [
        ("load_dataset",          node_load_dataset),
        ("infer_types",           node_infer_types),
        ("run_eda",               node_run_eda),
        ("cleaning",              node_cleaning),
        ("handle_human_decision", node_handle_human_decision),
        ("chat_router",           node_chat_router),
        ("custom_eda",            node_custom_eda),
        ("rerun_eda",             node_rerun_eda),
        ("save_csv",              node_save_csv),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("load_dataset")
    g.add_conditional_edges("load_dataset",  route_after_load,
                            {"handle_human_decision": "handle_human_decision",
                             "infer_types": "infer_types"})
    g.add_conditional_edges("infer_types",   route_after_infer,
                            {"handle_human_decision": "handle_human_decision",
                             "run_eda": "run_eda"})
    g.add_conditional_edges("run_eda",       route_after_eda,
                            {"handle_human_decision": "handle_human_decision",
                             END: END})
    g.add_conditional_edges("handle_human_decision", route_after_human,
                            {"load_dataset": "load_dataset",
                             "infer_types":  "infer_types",
                             "run_eda":      "run_eda",
                             END: END})
    g.add_conditional_edges("chat_router", route_chat,
                            {"custom_eda": "custom_eda", "cleaning": "cleaning",
                             "rerun_eda": "rerun_eda", "save_csv": "save_csv", END: END})
    g.add_edge("custom_eda", END)
    g.add_edge("cleaning",   END)
    g.add_edge("rerun_eda",  END)
    g.add_edge("save_csv",   END)
    return g.compile(checkpointer=checkpointer)


def _build_chat_graph(checkpointer):
    g = StateGraph(AgentState)
    g.add_node("chat_router", node_chat_router)
    g.add_node("custom_eda",  node_custom_eda)
    g.add_node("cleaning",    node_cleaning)
    g.add_node("rerun_eda",   node_rerun_eda)
    g.add_node("save_csv",    node_save_csv)
    g.set_entry_point("chat_router")
    g.add_conditional_edges("chat_router", route_chat,
                            {"custom_eda": "custom_eda", "cleaning": "cleaning",
                             "rerun_eda": "rerun_eda", "save_csv": "save_csv", END: END})
    g.add_edge("custom_eda", END)
    g.add_edge("cleaning",   "save_csv")
    g.add_edge("rerun_eda",  END)
    g.add_edge("save_csv",   END)
    return g.compile(checkpointer=checkpointer)


_memory     = MemorySaver()
_pipeline   = _build_pipeline_graph(checkpointer=_memory)
_chat_graph = _build_chat_graph(checkpointer=_memory)


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

def run_pipeline(file_path: str, thread_id: str = "default") -> AgentState:
    """Run load → type inference → EDA."""
    config = {"configurable": {"thread_id": thread_id}}
    log.info("[run_pipeline] file=%s thread=%s", file_path, thread_id)
    init: AgentState = {
        "messages":        [HumanMessage(content=f"Analyze: {file_path}")],
        "file_path":       file_path,
        "df_key":          None,
        "df_original_key": None,
        "schema_info":     None,
        "type_suggestions": None,
        "table_name":      None,
        "eda_report":      None,
        "eda_html":        None,
        "custom_result":   None,
        "cleaning_log":    [],
        "error":           None,
        "error_type":      None,
        "step":            "start",
        "awaiting_human":  False,
        "human_decision":  None,
    }
    return _pipeline.invoke(init, config=config)


def handle_human_response(decision: str, thread_id: str = "default") -> AgentState:
    """
    Resume after an error pause.
    """
    config = {"configurable": {"thread_id": thread_id}}
    log.info("[handle_human_response] decision='%s' thread=%s", decision, thread_id)

    # Read current saved state
    snapshot = _pipeline.get_state(config)
    sv = dict(snapshot.values) if snapshot and snapshot.values else {}

    # Patch decision into state
    sv["human_decision"] = decision
    sv["messages"] = sv.get("messages", []) + [HumanMessage(content=decision)]

    # Run the handler node directly to compute next state
    next_sv = node_handle_human_decision(sv)

    # Write patched state back to the checkpointer so the next invoke sees it
    _pipeline.update_state(config, next_sv)

    # Now re-invoke — the graph reads from checkpoint, routes via route_after_human
    return _pipeline.invoke(None, config=config)


def run_chat(user_message: str, thread_id: str = "default") -> AgentState:
    """Send a follow-up message (cleaning op, custom EDA, or general question)."""
    config  = {"configurable": {"thread_id": thread_id}}
    current = _pipeline.get_state(config)
    sv      = current.values if current and current.values else {}
    new_state = {
        **sv,
        "messages":       sv.get("messages", []) + [HumanMessage(content=user_message)],
        "step":           "chat",
        "human_decision": None,
    }
    return _chat_graph.invoke(new_state, config=config)


# ══════════════════════════════════════════════════════════════
# CLI DEMO
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    llm = ChatOpenAI(model="deepseek-chat", temperature=0)
    path = sys.argv[1] if len(sys.argv) > 1 else "Iris.csv"
    tid  = "cli-demo"
    print(f"\n▶ Running pipeline on: {path}\n")
    state = run_pipeline(path, thread_id=tid)

    # Print the latest AI message after each step
    def _print_last(s):
        ai_msgs = [m for m in s.get("messages", []) if isinstance(m, AIMessage)]
        if ai_msgs:
            print(f"\n[AGENT]: {ai_msgs[-1].content}\n")

    _print_last(state)

    while state.get("awaiting_human"):
        decision = input("Decision (retry/skip/abort/reupload <path>): ").strip()
        state = handle_human_response(decision, thread_id=tid)
        _print_last(state)

    # if state.get("eda_html"):
    #     p = Path("eda_report.html")
    #     p.write_text(state["eda_html"], encoding="utf-8")
    #     print(f"\n✅ EDA HTML saved → {p.resolve()}")
    
    if state.get("df_key") is not None:
        print("\n─"*50)
        print("Chat mode. Commands:")
        print("  Cleaning : 'drop column ID' · 'drop duplicates' · 'fill NaN in Age with mean'")
        print("  Analysis : 'histogram of Age' · 'scatter PetalLength vs PetalWidth' ")
        print("  EDA      : 'rerun eda'  — regenerates HTML report on current dataset")
        print("  Save     : 'save' or 'save as output.csv'  — exports cleaned CSV")
        print("  Exit     : 'exit'")
        while True:
            msg = input("\nYou: ").strip()
            if msg.lower() in ("exit", "quit"): break
            result = run_chat(msg, thread_id=tid)

            # Only print the LAST AI message
            ai_msgs = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
            if ai_msgs:
                print(f"Agent: {ai_msgs[-1].content}")

            step = result.get("step", "")

            # After cleaning — show dataset preview
            if step == "clean_complete":
                df_key = result.get("df_key")
                df = _df_store.get(df_key)
                if df is not None:
                    print(f"\n📋 Dataset preview ({len(df)} rows × {len(df.columns)} cols):")
                    print(df.head().to_string())
                    print()

            # After rerun EDA — save updated HTML report
            if step == "eda_complete" and result.get("eda_html"):
                p = Path(f"{result.get('table_name', 'dataset')}_eda_report.html")
                p.write_text(result["eda_html"], encoding="utf-8")
                print(f"📄 EDA report saved → {p.resolve()}")

            cr = result.get("custom_result", {})
            if cr and cr.get("plot_b64"):
                img = Path(f"plot_{cr['type']}.png")
                img.write_bytes(base64.b64decode(cr["plot_b64"]))
                print(f"[Plot saved → {img}]")