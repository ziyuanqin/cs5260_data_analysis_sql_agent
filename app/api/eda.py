import io
import os
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, HTMLResponse, StreamingResponse
from pydantic import BaseModel

# 确保路径正确导入你的 agent 和 shared
import backend.data_analysis.agent as agent
from backend.data_analysis.shared import log, _df_store

router = APIRouter(prefix="/api/eda", tags=["eda"])

# --- 模型定义 ---
class ChatRequest(BaseModel):
    thread_id: str
    message:   str

class HumanResponseRequest(BaseModel):
    thread_id: str
    decision:  str
# --- 核心工具函数 (必须包含，否则对话和状态返回会报错) ---

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
    """获取最新 AI 回复"""
    from langchain_core.messages import AIMessage
    msgs = state.get("messages", [])
    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""

# --- 接口实现 ---

@router.post("/chat")
async def eda_chat(body: ChatRequest):
    """
    真正的本地对话接口。
    不再通过 urlrequest 访问 8002，而是直接驱动本地 agent。
    """
    if agent.llm is None:
        agent.init_llm() # 自动从环境变量读取 Key

    try:
        # 1. 调用本地 agent 运行对话逻辑
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

# --- 人机交互接口 ---
@router.post("/human-response")
async def human_response(body: HumanResponseRequest):
    """处理错误暂停后的用户决策"""
    try:
        state = agent.handle_human_response(body.decision, thread_id=body.thread_id)
        return {
            "thread_id": body.thread_id,
            "message":   _last_ai_message(state),
            "state":     _state_summary(state),
        }
    except Exception as e:
        log.exception(f"处理人机交互失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 健康检查接口 ---
@router.get("/health")
async def health():
    return {"status": "ok", "llm_ready": agent.llm is not None}

@router.get("/report/{thread_id}")
async def get_eda_report(thread_id: str):
    """获取报告 HTML"""
    snapshot = agent._pipeline.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="未找到分析记录")

    html = snapshot.values.get("eda_html")
    return HTMLResponse(content=html)

@router.get("/download-csv/{thread_id}")
async def download_csv(thread_id: str):
    """下载清洗后的 CSV"""
    snapshot = agent._pipeline.get_state({"configurable": {"thread_id": thread_id}})
    df_key = snapshot.values.get("df_key")
    df = _df_store.get(df_key)

    if df is None:
        raise HTTPException(status_code=404, detail="数据已失效")

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cleaned_{thread_id}.csv"}
    )

@router.get("/chart/{thread_id}/{chart_name}", summary="Get a single chart as base64 PNG")
async def get_chart(thread_id: str, chart_name: str):
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