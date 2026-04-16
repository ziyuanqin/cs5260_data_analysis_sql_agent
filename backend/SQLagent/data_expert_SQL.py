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
from backend.data_analysis.agent import init_llm

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
    intent: Optional[str]

# 2. 定义节点逻辑
class SQLExpert:
    def __init__(self):
        # 初始化 LLM 并保存返回值
        self.llm = init_llm()
        if self.llm is None:
            raise RuntimeError("SQLExpert initialization failed: init_llm returned None")

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
                    def clean_column_name(name):
                        # 1. 统一转小写，去除首尾空格
                        name = str(name).strip().lower()
                        # 2. 将空格、短横杠、点号等非法字符全部替换为下划线
                        name = re.sub(r'[\s\-\.]+', '_', name)
                        # 3. 去除重复的下划线 (e.g., "order--date" -> "order_date")
                        name = re.sub(r'_+', '_', name)
                        # 4. 确保不以数字开头（SQL规范）
                        if name[0].isdigit():
                            name = "col_" + name
                        return name

                    df.columns = [clean_column_name(c) for c in df.columns]
                    df.to_sql(final_table, engine, if_exists="replace", index=False)
                    existing_tables.append(final_table)
                    new_tables.append(final_table)
            return new_tables

        # 使用 to_thread 运行，不阻塞主线程
        new_added = await asyncio.to_thread(_sync_ingest)
        return {"excel_paths": None, "active_tables": new_added}

    def schema_awareness(self, state: AgentState, engine):
        try:
            inspector = inspect(engine)
            table_names = inspector.get_table_names()

            full_schema = []
            for table in table_names:
                # 获取列信息：名称和类型
                cols = inspector.get_columns(table)
                col_info = [f"{c['name']} ({c['type']})" for c in cols]

                # 关键：获取主键和外键，帮助 AI 理解字段差异
                pk = inspector.get_pk_constraint(table).get("constrained_columns", [])
                fk = inspector.get_foreign_keys(table)

                schema_str = f"TABLE: {table}\nCOLUMNS: {', '.join(col_info)}\nPRIMARY KEY: {pk}"
                if fk:
                    schema_str += f"\nFOREIGN KEYS: {fk}"

                full_schema.append(schema_str)

            mapping_guide = (
                    "\n" + "="*30 + "\n"
                                    "### COLUMN MAPPING RULES:\n"
                                    "1. All Excel columns have been normalized: spaces (' '), dashes ('-'), and dots ('.') are replaced by underscores ('_').\n"
                                    "2. Example: If a user asks for 'Order Date' or 'Order-Date', look for 'order_date' in the COLUMNS list above.\n"
                                    "3. Case Sensitivity: Always use the exact case shown in the COLUMNS list (mostly lowercase)."
            )
            full_schema.append(mapping_guide)

            return {
                "schema_info": "\n\n".join(full_schema),
                "retry_count": 0,
                "active_tables": table_names
            }
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

        current_question = state["messages"][-1].content
        schema = str(state.get('schema_info', ""))
        last_error = state.get('error')

        domain_hint = str(self._get_relevant_knowledge(current_question, schema))

        # 1. Security Check (Hard Block)
        illegal_pattern = r"\b(DROP|DELETE|UPDATE|TRUNCATE|ALTER|INSERT|GRANT)\b"
        if re.search(illegal_pattern, current_question, re.IGNORECASE):
            return {
                "intent": "violation",
                "error": "Access Denied: You only have READ-ONLY (SELECT) permissions. Structural or data modifications are prohibited."
            }

        # 2. Intent Classification via LLM
        # We ask the LLM to decide if this is a request for SQL or just a request for ideas.
        intent_prompt = f"""
        Analyze the user's request: "{current_question}"
        Based on the context, categorize the intent as:
        - 'advice': User is ONLY asking what CAN be done, seeking suggestions, or brainstorming.
        - 'query': User is asking for specific data, numbers, or has accepted a previous suggestion and wants the result.
        Return ONLY the word 'advice' or 'query'.
        """
        intent_res = self.llm.invoke(intent_prompt).content.lower().strip()

        if 'advice' in intent_res:
            return {"intent": "advice", "sql_query": "SKIP"}

        error_feedback = f"Feedback: {str(last_error)}" if last_error else "System status: Healthy."
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
        
        ### CRITICAL RULES FOR COLUMN NAMES
        - RULE 1: [Database Schema] is the ONLY source of truth. 
        - RULE 2: If the user asks for 'User ID' but Table A has 'uid' and Table B has 'userid', you MUST use the exact name specified in the [Database Schema] for that specific table.
        - RULE 3: DO NOT use column names from [Conversation History] if they are not present in the current [Database Schema]. The database environment may have changed.
        - RULE 4: Always prefix columns with table names (e.g., `table_a.uid`) to avoid ambiguity.
        
        PRIORITY: If the [Database Schema] conflicts with [Conversation History], ALWAYS follow the [Database Schema]. The database structure has changed.
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
        return {"sql_query": clean_sql, "intent": "query"}


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

        """
        Final node that handles English communication for three scenarios:
        1. Security violations (Read-only enforcement).
        2. Strategic suggestions (Advice mode without SQL).
        3. Data storytelling (Interpreting SQL query results).
        """
        intent = state.get("intent")
        error = state.get('error')
        current_question = state["messages"][-1].content if state.get("messages") else "N/A"

        # --- Scenario 1: Security or Max Retry Failures ---
        if intent == "violation" or (error and state.get('retry_count', 0) >= 3):
            fail_msg = (
                f"I cannot proceed with this request. "
                f"Reason: {error if error else 'Security Policy Violation - Only SELECT operations are allowed.'}"
            )
            return {"analysis": fail_msg, "messages": [AIMessage(content=fail_msg)], "retry_count": 0}

        # --- Scenario 2: Strategic Advice Mode (No SQL was run) ---
        if intent == "advice":
            advice_template = Template(r"""
            ### ROLE
            You are a Senior Business Intelligence Consultant. You excel at turning raw data schemas into strategic business insights.
            
            ### OBJECTIVE
            Based on the provided schema, suggest 3-4 distinct, high-value business analyses. 
            For each suggestion, you MUST follow this exact Markdown structure:
            
            **[Number]. [Analysis Title]**
            - **Analysis**: [Describe what will be examined and which data dimensions will be correlated, without using technical table names.]
            - **Business Value**: [Explain the strategic impact, e.g., ROI, cost reduction, or customer retention.]
            
            ### CONTEXT
            - [User Question]: $question
            - [Available Schema]: $schema
            
            ### CONSTRAINTS
            - **STRICTLY PROHIBITED**: Do not show any SQL code, actual table names, or technical parameters (like JOINs or data types).
            - **NO PROSE**: Do not include introductory or concluding remarks like "Here is the analysis...". Start directly with the suggestions.
            - **LANGUAGE**: Must be English.
            """)

            prompt = advice_template.safe_substitute(
                question=current_question,
                schema=state.get('schema_info', 'No schema provided')
            )
            response = self.llm.invoke(prompt)
            return {"analysis": response.content, "messages": [AIMessage(content=response.content)]}

        # --- Scenario 3: Data Interpretation Mode (After successful SQL execution)
        analysis_template = Template(r"""
        ### ROLE
        You are a Senior Strategic Business Consultant and Data Storyteller. Your goal is to transform raw query results into high-impact executive insights.
        
        ###Constraints (must be observed)
        1. It is strictly prohibited to output any SQL code, technical parameter or database table name.
        2. Only output business conclusions and action suggestions.
        3. If the data is empty, please politely inform the user that the relevant record is not found.
        
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
        Provide exactly three (3) highly targeted, non-generic business recommendations. 
        **CRITICAL: You must use explicit numbering 1, 2, and 3.**
        
        1. **Strategic Pillar A: [Insert High-Level Title]**
           - **Observation**: What does the data say?
           - **Action**: What specific step should the business take?
           - **Expected Impact**: What is the projected ROI or strategic benefit?
        
        2. **Strategic Pillar B: [Insert High-Level Title]**
           - **Observation**: ...
           - **Action**: ...
           - **Expected Impact**: ...
        
        3. **Strategic Pillar C: [Insert High-Level Title]**
           - **Observation**: ...
           - **Action**: ...
           - **Expected Impact**: ...
        
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

    # 1. Intent Router after SQL Generation
    def intent_router(state: AgentState):
        """
        Determines if we should execute SQL or skip directly to analysis.
        """
        intent = state.get("intent")
        # If it's a violation (DROP) or just advice, BYPASS sql_exec
        if intent in ["violation", "advice"]:
            return "skip_to_analysis"
        return "continue_to_exec"

    # Add the conditional edge for sql_gen
    workflow.add_conditional_edges(
        "sql_gen",
        intent_router,
        {
            "skip_to_analysis": "analysis",
            "continue_to_exec": "sql_exec"
        }
    )

    # 2. Retry Logic after SQL Execution
    def retry_logic(state: AgentState):
        # Only retry if it's a genuine execution error, not a policy violation
        if state.get("error") and state.get("retry_count", 0) < 2:
            return "retry"
        return "end"

    workflow.add_conditional_edges(
        "sql_exec",
        retry_logic,
        {"retry": "sql_gen", "end": "analysis"}
    )

    workflow.add_edge("analysis", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)