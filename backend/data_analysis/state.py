"""
state.py — LangGraph state definition.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages:          Annotated[list, add_messages]
    file_path:         Optional[str]
    df_key:            Optional[str]       # key into shared._df_store
    df_original_key:   Optional[str]       # key into shared._df_store (immutable original)
    schema_info:       Optional[dict]
    type_suggestions:  Optional[dict]      # {col: {current, suggested, reason, confidence}}
    table_name:        Optional[str]
    eda_report:        Optional[dict]
    eda_html:          Optional[str]
    custom_result:     Optional[dict]
    cleaning_log:      Optional[list]      # ordered list of applied cleaning ops
    error:             Optional[str]
    error_type:        Optional[str]
    step:              Optional[str]
    awaiting_human:    Optional[bool]
    human_decision:    Optional[str]
