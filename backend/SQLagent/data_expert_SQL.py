import os
import pandas as pd
import re
import asyncio
from string import Template
from typing import TypedDict, Annotated, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv
from backend.SQLagent.registry import DOMAIN_REGISTRY

# 优先使用项目 .env，避免被 shell 中旧变量污染导致鉴权失败。
load_dotenv(override=True)

# 1. 定义状态结构
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    db_type: str
    excel_paths: Optional[list[str]]
    # 记录本次对话涉及的表，方便调试
    active_tables: Optional[list[str]]
    sql_query: Optional[str]
    query_result: Any
    analysis: Optional[str]
    error: Optional[str]
    retry_count: int
    schema_info: Optional[str]

# 2. 定义节点逻辑
class SQLExpert:
    def __init__(self):
        # 使用具备长上下文理解能力的模型
        self.llm = ChatOpenAI(model="deepseek-chat", temperature=0)

        # 5大核心领域指标知识库
        self.domain_registry = DOMAIN_REGISTRY

    async def data_ingestion(self, state: AgentState, engine):
        """异步入库逻辑"""
        paths = state.get("excel_paths")
        if not paths: return {}

        # 将同步的 Pandas/SQL 操作封装
        def _sync_ingest():
            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()
            new_tables = []
            for path in paths:
                if os.path.exists(path):
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    target_table = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
                    final_table = target_table
                    counter = 1
                    while final_table in existing_tables:
                        final_table = f"{target_table}_{counter}"
                        counter += 1

                    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
                    df.columns = [c.strip().replace(' ', '_').replace('-', '_') for c in df.columns]
                    # 阻塞 I/O
                    df.to_sql(final_table, engine, if_exists="replace", index=False)
                    existing_tables.append(final_table)
                    new_tables.append(final_table)
            return new_tables

        # 使用 to_thread 运行，不阻塞主线程
        new_added = await asyncio.to_thread(_sync_ingest)
        return {"excel_paths": None, "active_tables": new_added}

    def schema_awareness(self, state: AgentState, engine):
        # 即使 schema_info 已存在也建议重新探测，以支持对话中途新增文件
        try:
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            if not table_names:
                return {"schema_info": "当前数据库为空，请先上传数据。"}

            schema_context = []
            for table in table_names:
                columns = [col['name'] for col in inspector.get_columns(table)]
                # 获取 1 条样例数据帮助 LLM 更有把握
                sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", engine)
                schema_context.append(f"Table: {table}, Columns: {columns}, Sample: {sample.to_dict(orient='records')}")

            return {"schema_info": "\n".join(schema_context), "retry_count": 0}
        except Exception as e:
            return {"error": f"Schema 探测失败: {str(e)}"}

    def _get_relevant_knowledge(self, text: str, schema: str) -> str:
        combined_text = text.lower()
        selected_knowledge = []
        for domain, info in self.domain_registry.items():
            # 改进：不仅匹配 keywords，也匹配领域名本身
            if any(kw in combined_text for kw in info["keywords"]) or domain in combined_text:
                safe_metrics = info['metrics'].replace("%", "%%")
                safe_tips = info['sql_tips'].replace("%", "%%")
                # 结构化输出，方便 LLM 区分
                selected_knowledge.append(f"### 【{domain.upper()}领域规范】\n业务指标:\n{safe_metrics}\nSQL建议:\n{safe_tips}")

        return "\n\n".join(selected_knowledge) if selected_knowledge else "请基于通用统计逻辑进行分析。"

    async def generate_sql(self, state: AgentState):

        # 1. 准备基础数据
        schema = str(state.get('schema_info', "Unknown schema"))
        current_question = state["messages"][-1].content if state.get("messages") else ""
        domain_hint = str(self._get_relevant_knowledge(current_question, schema))
        last_error = state.get('error')

        # 2. 格式化错误信息 - 核心修复：严禁使用 f-string
        if last_error:
            # High-priority feedback to guide the LLM's self-correction
            error_feedback = f"⚠️ WARNING: The previous SQL execution failed with error: {str(last_error)}. Please cross-reference the Schema and fix the syntax."
        else:
            error_feedback = "System status: Healthy."

        dialect_prompt = "Standard MySQL syntax" if state.get('db_type') == 'mysql' else "SQLite-compatible syntax"

        # 3. 格式化对话历史
        history_list = []
        if "messages" in state and len(state["messages"]) > 1:
            for m in state["messages"][:-1]:
                role = "User" if isinstance(m, HumanMessage) else "Assistant"
                # 历史记录也可能包含带 % 的 SQL，必须安全处理
                history_list.append(role + ": " + str(m.content))
        history_context = "\n".join(history_list) if history_list else "Initial turn of conversation."

        # 4. 使用 Template 渲染
        template_str = """You are an expert Data Engineer and Senior SQL Developer. Your goal is to translate natural language questions into high-performance, syntactically correct SQL queries based on the provided context.
        
        - [Database Dialect]: $dialect
        
        - [Database Schema]: $schema
        
        - [Conversation History]: $history
        
        - [Domain-Specific Knowledge]: $domain
        (Note: Use the Domain Business Logic for calculation methods, but STRICTLY map them to the real column names provided in the [Database Schema].)
        
        - [ERROR/FEEDBACK CORRECTION]: $error_msg
        
        - [USER QUESTION]: $question
        
        1. OUTPUT FORMAT: Return ONLY the raw SQL string. Do not include Markdown blocks (```sql), explanations, or any conversational filler.
        2. SCHEMA FIDELITY: Do not hallucinate columns. Use ONLY the columns listed in the [Database Schema]. If a required column is missing, use the most logical substitute or return a comment indicating the missing field.
        3. ALIASING: Always provide clear, English aliases for aggregated or calculated columns (e.g., `SUM(sales) AS total_revenue`).
        4. STRING MATCHING: Do not use the `LIKE` operator unless the user explicitly requests partial matching. Favor exact equality `=` for performance.
        5. NULL HANDLING: Use `COALESCE()` or `IFNULL()` for any arithmetic operations to prevent NULL propagation in results.
        6. JOIN LOGIC: Prefer explicit `JOIN` syntax over implicit comma-separated joins. Always specify the join key.
        
        - PREDICATE PUSHdown: Place filtering conditions in the `WHERE` clause rather than `HAVING` whenever possible to reduce the dataset before grouping.
        - SELECT SPECIFICITY: Avoid `SELECT *`. Explicitly name only the columns required by the user's question.
        - AGGREGATION EFFICIENCY: For "Top N" queries, use `DENSE_RANK()` or `ROW_NUMBER()` over a `LIMIT` if the business logic requires handling ties.
        - TYPE CONSISTENCY: Ensure comparisons match data types (e.g., do not compare a string to an integer without explicit casting if required by the $dialect).
        
        """

        t = Template(template_str)
        prompt = t.safe_substitute(
            dialect=dialect_prompt,
            schema=schema,
            history=history_context,
            domain=domain_hint,
            error_msg=error_feedback,
            question=current_question
        )

        response = await self.llm.ainvoke(prompt)
        clean_sql = re.sub(r'```sql\s*|\s*```', '', response.content).strip().rstrip(';')
        return {"sql_query": clean_sql}


    async def execute_sql(self, state: AgentState, engine):
        """异步执行 SQL"""
        sql = state.get('sql_query')
        if not sql: return {"error": "SQL generation failed."}

        if any(kw in sql.upper() for kw in ["DROP", "DELETE", "UPDATE"]):
            return {"error": "Security Policy: Only SELECT statements are permitted."}

        # 定义同步执行块
        def _run_query():
            return pd.read_sql(sql, engine)

        try:
            # 在线程池中执行查询
            result = await asyncio.to_thread(_run_query)
            clean_result = result.to_dict(orient='records')
            return {"query_result": clean_result, "error": ""}
        except Exception as e:
            return {"error": str(e).replace("%", "%%"), "retry_count": state.get('retry_count', 0) + 1}

    def analyze_result(self, state: AgentState):

        current_question = state["messages"][-1].content if state.get("messages") else "Unknown problem"

        if state.get('error') and state.get('retry_count', 0) >= 3:

            fail_msg = "System reached maximum retries without a successful query. Reason:" + str(state['error']) + "。"
            return {"analysis": fail_msg, "messages": [AIMessage(content=fail_msg)], "retry_count": 0}

        # 处理成功情况

        # 使用 Template 保护分析阶段的 Prompt
        analysis_template = Template(r"""
        ### ROLE
        You are a Senior Strategic Business Consultant and Data Storyteller. Your goal is to transform raw query results into high-impact executive insights.
        
        ### CONTEXT
        - [User's Strategic Question]: $question
        - [Retrieved Data Result]: 
        $result
        
        ### ANALYSIS GUIDELINES
        1. **Mathematical Precision**: 
           - Use LaTeX for all mathematical notations, growth formulas, and statistical summaries (e.g., $$Growth = \frac{V_{current} - V_{past}}{V_{past}}$$).
           - Clearly state the $n$ (sample size) or totals if available in the result.
        
        2. **Data Interpretation**:
           - **Contextualize the numbers**: Don't just list values. Identify trends (e.g., "A $15\%$ MoM increase"), anomalies, or significant concentrations (e.g., "Top 3 SKUs contribute $80\%$ of revenue").
           - **Root Cause Hypothesis**: Briefly suggest a "why" behind the data based on industry benchmarks.
        
        3. **Executive Summary**:
           - Provide a 1-sentence "Bottom Line Up Front" (BLUF) that answers the core question.
        
        ### STRATEGIC RECOMMENDATIONS (3 Pillars)
        Provide exactly three (3) highly targeted, non-generic business recommendations. Each must follow this structure:
        - **Observation**: What does the data say?
        - **Action**: What specific step should the business take?
        - **Expected Impact**: What is the projected ROI or strategic benefit?
        
        ### CONSTRAINTS
        - Avoid corporate jargon; be concise and direct.
        - If the [Retrieved Data Result] is empty or insufficient, state exactly what additional data is needed to provide a valid answer.
        - Do not hallucinate external facts not present in the data.
        """)

        safe_prompt = analysis_template.safe_substitute(
            question=current_question,
            result=str(state.get('query_result')).replace("%", "%%") # 关键加固
        )

        response = self.llm.invoke(safe_prompt)
        return {"analysis": response.content, "messages": [AIMessage(content=response.content)], "retry_count": 0}

# 3. 构建 LangGraph
def create_smart_sql_graph(engine):
    expert = SQLExpert()
    workflow = StateGraph(AgentState)

    # --- 节点包装逻辑 ---

    # 1. ingest 是 async def，需要 await
    async def _ingest(state: AgentState):
        return await expert.data_ingestion(state, engine)

    # 2. schema_awareness 是普通 def，绝对不能 await！
    # 如果想不阻塞，可以用 asyncio.to_thread，或者直接同步调用
    def _detect_schema(state: AgentState):
        return expert.schema_awareness(state, engine)

    # 3. sql_exec 是 async def，需要 await
    async def _sql_exec(state: AgentState):
        return await expert.execute_sql(state, engine)

    # --- 注册节点 ---
    workflow.add_node("ingest", _ingest)
    workflow.add_node("detect_schema", _detect_schema)
    workflow.add_node("sql_gen", expert.generate_sql) # generate_sql 是 async，LangGraph 会自动处理
    workflow.add_node("sql_exec", _sql_exec)
    workflow.add_node("analysis", expert.analyze_result) # analyze_result 是同步，LangGraph 也会处理

    # --- 编排工作流 ---
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "detect_schema")
    workflow.add_edge("detect_schema", "sql_gen")
    workflow.add_edge("sql_gen", "sql_exec")

    def retry_logic(state: AgentState):
        # 这里的 state 是字典，逻辑正确
        if state.get("error") and state.get("retry_count", 0) < 2:
            return "retry"
        return "end"

    workflow.add_conditional_edges("sql_exec", retry_logic, {"retry": "sql_gen", "end": "analysis"})
    workflow.add_edge("analysis", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)