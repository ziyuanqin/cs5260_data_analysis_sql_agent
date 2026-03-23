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

# --- 核心工具函数 (必须包含，否则对话和状态返回会报错) ---

def _clean(obj):
    """处理 numpy 类型，防止 JSON 报错"""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v or v == float("inf") or v == float("-inf")) else v
    return obj

def _state_summary(state: dict) -> dict:
    """构造前端需要的状态摘要"""
    eda = state.get("eda_report") or {}
    charts = list((eda.get("charts") or {}).keys())
    return _clean({
        "thread_id":       state.get("table_name"),
        "step":            state.get("step"),
        "awaiting_human":  state.get("awaiting_human", False),
        "table_name":      state.get("table_name"),
        "has_eda_html":    bool(state.get("eda_html")),
        "available_charts": charts,
        "cleaning_log":    state.get("cleaning_log") or [],
        "schema_info":     state.get("schema_info") or {},
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

        # 2. 构造响应
        response = {
            "thread_id":  body.thread_id,
            "message":    _last_ai_message(state),
            "state":      _state_summary(state),
        }

        # 3. 如果对话产生了新图表，也带回去
        cr = state.get("custom_result") or {}
        if cr.get("plot_b64"):
            response["plot_b64"] = cr["plot_b64"]

        return response

    except Exception as e:
        log.exception(f"EDA 对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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