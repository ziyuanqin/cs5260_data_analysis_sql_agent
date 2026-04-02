"""
api.py — FastAPI backend for the Data Analysis Agent.

Endpoints:
  POST /api-key                    — set OpenAI API key (required before anything else)
  GET  /api-key/status             — check whether a key has been set

  POST /upload                     — upload CSV + run pipeline (load → types → EDA)
  GET  /state/{thread_id}          — current session state snapshot
  GET  /eda-report/{thread_id}     — full HTML EDA report
  GET  /chart/{thread_id}/{name}   — single chart as base64 PNG

  POST /chat                       — cleaning / EDA / Q&A chat message
  POST /human-response             — respond to an error prompt (retry/skip/abort/reupload)

  GET  /download-csv/{thread_id}   — download the current cleaned DataFrame as CSV
  GET  /health                     — health check

Run:
  uvicorn api:app --reload --port 8000
"""

import io
import os
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

import agent
from shared import log, _df_store

# APP SETUP

app = FastAPI(
    title="Data Analysis Agent API",
    description="LangGraph-powered CSV analysis, cleaning, and EDA agent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp directory for uploaded files
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# HELPERS

def _clean(obj):
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v or v == float("inf") or v == float("-inf")) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _state_summary(state: dict) -> dict:
    """Return a JSON-safe snapshot of the session state (no raw DataFrames or HTML)."""
    eda = state.get("eda_report") or {}
    charts = list((eda.get("charts") or {}).keys())
    return _clean({
        "thread_id":       state.get("table_name"),   # convenient alias
        "step":            state.get("step"),
        "awaiting_human":  state.get("awaiting_human", False),
        "error":           state.get("error"),
        "error_type":      state.get("error_type"),
        "table_name":      state.get("table_name"),
        "df_key":          state.get("df_key"),
        "has_eda_html":    bool(state.get("eda_html")),
        "has_eda_report":  bool(state.get("eda_report")),
        "available_charts": charts,
        "cleaning_log":    state.get("cleaning_log") or [],
        "type_suggestions": state.get("type_suggestions") or {},
        "schema_info": {
            "row_count":  (state.get("schema_info") or {}).get("row_count"),
            "col_count":  (state.get("schema_info") or {}).get("col_count"),
            "column_names": (state.get("schema_info") or {}).get("column_names", []),
        },
    })


def _last_ai_message(state: dict) -> str:
    """Extract the most recent AIMessage content from state."""
    from langchain_core.messages import AIMessage
    msgs = state.get("messages", [])
    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""


def _require_key():
    """Raise 401 if the LLM hasn't been initialised yet."""
    if agent.llm is None:
        raise HTTPException(
            status_code=401,
            detail="OpenAI API key not set. POST to /api-key first.",
        )


# REQUEST / RESPONSE MODELS

class ApiKeyRequest(BaseModel):
    api_key: str

class ChatRequest(BaseModel):
    thread_id: str
    message:   str

class HumanResponseRequest(BaseModel):
    thread_id: str
    decision:  str   # retry | skip | abort | reupload <path>


# ── API Key ───────────────────────────────────────────────────

@app.post("/api-key", summary="Set OpenAI API key")
def set_api_key(body: ApiKeyRequest):
    """
    Set the OpenAI API key. Must be called before any analysis.
    The key is stored in memory only — it is never written to disk.
    """
    if not body.api_key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="Invalid API key format (must start with 'sk-').")
    try:
        agent.init_llm(body.api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialise LLM: {e}")
    return {"status": "ok", "message": "API key accepted. LLM ready (gpt-4o-mini)."}


@app.get("/api-key/status", summary="Check if API key is set")
def api_key_status():
    return {"llm_ready": agent.llm is not None}


# ── Upload & Pipeline ─────────────────────────────────────────

@app.post("/upload", summary="Upload CSV and run full pipeline")
async def upload_and_analyze(
    file:      UploadFile = File(...),
    thread_id: Optional[str] = None,
):
    """
    Upload a CSV file and run the full pipeline:
    load → type inference → EDA.

    Returns a thread_id you use for all subsequent calls.
    """
    _require_key()

    thread_id = thread_id or f"session_{uuid.uuid4().hex[:8]}"
    suffix    = Path(file.filename).suffix or ".csv"
    save_path = UPLOAD_DIR / f"{thread_id}{suffix}"

    # Save upload to disk
    contents = await file.read()
    save_path.write_bytes(contents)
    log.info("[/upload] Saved %s → %s", file.filename, save_path)

    try:
        state = agent.run_pipeline(str(save_path), thread_id=thread_id)
    except Exception as e:
        log.exception("[/upload] Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "thread_id":     thread_id,
        "message":       _last_ai_message(state),
        "state":         _state_summary(state),
    }


@app.get("/state/{thread_id}", summary="Get session state snapshot")
def get_state(thread_id: str):
    """Returns a JSON summary of the current session state."""
    _require_key()
    snapshot = agent._pipeline.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"No session found for thread_id='{thread_id}'.")
    return _state_summary(dict(snapshot.values))


@app.get("/eda-report/{thread_id}", response_class=HTMLResponse,
         summary="Get full HTML EDA report")
def get_eda_report(thread_id: str):
    """Returns the full EDA report as an HTML page (render directly in browser)."""
    snapshot = agent._pipeline.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found.")
    html = snapshot.values.get("eda_html")
    if not html:
        raise HTTPException(status_code=404, detail="EDA report not yet generated.")
    return HTMLResponse(content=html)


@app.get("/chart/{thread_id}/{chart_name}", summary="Get a single chart as base64 PNG")
def get_chart(thread_id: str, chart_name: str):
    """
    Returns a specific chart from the EDA report.
    Available names: missing_bar, correlation_heatmap, histograms, categorical_bars.
    """
    snapshot = agent._pipeline.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found.")
    eda = snapshot.values.get("eda_report") or {}
    charts = eda.get("charts") or {}
    b64 = charts.get(chart_name)
    if not b64:
        raise HTTPException(status_code=404, detail=f"Chart '{chart_name}' not found.")
    return {"chart_name": chart_name, "plot_b64": b64}



@app.post("/chat", summary="Send a chat message (cleaning / EDA / Q&A)")
def chat(body: ChatRequest):
    """
    Send a message to the agent. Handles:
    - Cleaning ops: 'drop column Id', 'fill NaN in Age with mean'
    - Custom EDA:   'histogram of Age', 'scatter X vs Y'
    - EDA refresh:  'rerun eda'
    - Save CSV:     'save' or 'save as output.csv'
    - General Q&A:  anything else
    """
    _require_key()
    try:
        state = agent.run_chat(body.message, thread_id=body.thread_id)
    except Exception as e:
        log.exception("[/chat] Error")
        raise HTTPException(status_code=500, detail=str(e))

    response = {
        "thread_id":  body.thread_id,
        "message":    _last_ai_message(state),
        "step":       state.get("step"),
        "state":      _state_summary(state),
    }

    # Include custom result if a plot was produced
    cr = state.get("custom_result") or {}
    if cr.get("plot_b64"):
        response["plot_b64"]    = cr["plot_b64"]
        response["plot_type"]   = cr.get("type")
        response["plot_stats"]  = _clean(cr.get("stats") or {})
        response["text_result"] = cr.get("text_result", "")

    return response



@app.post("/human-response", summary="Respond to an error prompt")
def human_response(body: HumanResponseRequest):
    """
    Resume the pipeline after it paused with awaiting_human=True.

    decision options:
      retry                  — retry the failed step
      skip                   — skip the failed step and continue
      abort                  — stop the pipeline
      reupload /path/to.csv  — load a different file
    """
    _require_key()
    try:
        state = agent.handle_human_response(body.decision, thread_id=body.thread_id)
    except Exception as e:
        log.exception("[/human-response] Error")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "thread_id": body.thread_id,
        "message":   _last_ai_message(state),
        "state":     _state_summary(state),
    }



@app.get("/download-csv/{thread_id}", summary="Download cleaned dataset as CSV")
def download_csv(thread_id: str):
    """
    Stream the current (cleaned) DataFrame as a CSV file download.
    The filename is <table_name>_cleaned.csv.
    """
    snapshot = agent._pipeline.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found.")

    df_key = snapshot.values.get("df_key")
    df     = _df_store.get(df_key) if df_key else None
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded for this session.")

    table_name = snapshot.values.get("table_name", "dataset")
    filename   = f"{table_name}_cleaned.csv"

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "llm_ready": agent.llm is not None}