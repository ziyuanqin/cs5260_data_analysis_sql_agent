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

load_dotenv()

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
        self.domain_registry = {
            "retail": {
                "keywords": ["sales", "order", "product", "inventory", "customer"],
                "metrics": "电商/零售领域指标：\n- RFM模型: R(最近消费), F(频率), M(总额)。\n- 连带率: count(distinct product_id) / count(distinct order_id)。",
                "sql_tips": "使用 SQLite 计算复购：`SELECT user_id FROM t GROUP BY user_id HAVING COUNT(order_id) > 1`。"
            },
            "finance": {
                "keywords": ["loan", "balance", "interest", "credit", "transaction"],
                "metrics": "金融风控指标：\n- 逾期率: 逾期金额 / 总待还金额。\n- ROI: (收益 - 成本) / 成本。",
                "sql_tips": "风控分析常需处理空值：使用 `COALESCE(amount, 0)` 防止计算出错。"
            },
            "product": {
                "keywords": ["user_id", "active", "session", "click", "login"],
                "metrics": "互联网运营指标：\n- 次日留存率: 第N天登录人数 / 第一天新增人数。\n- DAU/MAU 活跃比率。",
                "sql_tips": "留存分析模板：`LEFT JOIN` 同一张表并判断 `date_diff = 1`。"
            },
            "manufacturing": {
                "keywords": ["sensor", "machine", "output", "failure", "efficiency"],
                "metrics": "制造/供应链指标：\n- OEE(设备综合效率): 可用率 × 表现指数 × 质量指数。\n- 库存周转率。",
                "sql_tips": "时序数据处理：使用 `LAG()` 比较当前批次与上一批次的良率。"
            },
            "healthcare": {
                "keywords": ["patient", "drug", "diagnosis", "treatment", "test_result"],
                "metrics": "医疗健康指标：\n- 治愈率: 治愈人数 / 确诊人数。\n- 相关性: P-value 显著性检验。",
                "sql_tips": "患者隐私保护：查询时避免暴露敏感字段名，必要时进行脱敏处理。"
            }
        }

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
        combined_text = text.lower() # 只匹配问题
        selected_knowledge = []
        for domain, info in self.domain_registry.items():
            if any(kw in combined_text for kw in info["keywords"]):
                # 这里注入时，确保把内容里的 % 预先转义为 %%
                safe_metrics = info['metrics'].replace("%", "%%")
                safe_tips = info['sql_tips'].replace("%", "%%")
                # 使用 + 号拼接，绝对安全
                selected_knowledge.append("---【" + domain + "领域建议】---\n" + safe_metrics + "\n" + safe_tips)
        return "\n".join(selected_knowledge) if selected_knowledge else "请基于通用统计逻辑进行分析。"

    async def generate_sql(self, state: AgentState):

        # 1. 准备基础数据
        schema = str(state.get('schema_info', "未知架构"))
        current_question = state["messages"][-1].content if state.get("messages") else ""
        domain_hint = str(self._get_relevant_knowledge(current_question, schema))
        last_error = state.get('error')

        # 2. 格式化错误信息 - 核心修复：严禁使用 f-string
        if last_error:
            # 使用字符串拼接，绝对安全
            error_feedback = "⚠️ 注意：上一次生成的 SQL 报错：" + str(last_error) + "。请核对 Schema 修正。"
        else:
            error_feedback = "状态正常。"

        dialect_prompt = "使用 MySQL 语法" if state.get('db_type') == 'mysql' else "使用 SQLite 语法"

        # 3. 格式化对话历史
        history_list = []
        if "messages" in state and len(state["messages"]) > 1:
            for m in state["messages"][:-1]:
                role = "User" if isinstance(m, HumanMessage) else "Assistant"
                # 历史记录也可能包含带 % 的 SQL，必须安全处理
                history_list.append(role + ": " + str(m.content))
        history_context = "\n".join(history_list) if history_list else "这是首轮对话。"

        # 4. 使用 Template 渲染
        template_str = """你是一个专业的数据分析师。请根据以下上下文生成 SQL 语句。
        
        [数据库环境]: $dialect
        [Schema 信息]:
        $schema
        
        [对话历史记录]:
        $history
        
        [领域知识注入]:
        $domain
        
        [反馈/错误修复]:
        $error_msg
        
        [当前用户需求]: $question
        
        要求：
        1. 直接返回 SQL，严禁包含任何 Markdown 格式（如 ```sql）。
        2. 严禁猜测字段，必须使用 [Schema 信息] 中存在的列。
        3. 聚合字段必须使用英文别名。
        4. 如果用户没有指明在SQL里面使用like，则不要使用。
        """

        t = Template(template_str)
        prompt = t.safe_substitute(
            dialect=dialect_prompt,
            schema=schema,
            history=history_context,
            domain=domain_hint,
            error_msg=error_feedback, # 传入预先拼接好的纯字符串
            question=current_question
        )

        response = await self.llm.ainvoke(prompt)
        clean_sql = re.sub(r'```sql\s*|\s*```', '', response.content).strip().rstrip(';')
        return {"sql_query": clean_sql}


    async def execute_sql(self, state: AgentState, engine):
        """异步执行 SQL"""
        sql = state.get('sql_query')
        if not sql: return {"error": "未生成 SQL"}

        if any(kw in sql.upper() for kw in ["DROP", "DELETE", "UPDATE"]):
            return {"error": "只允许执行 SELECT 语句。"}

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
        """生成带商业洞察的最终回复（安全拼接版）"""
        current_question = state["messages"][-1].content if state.get("messages") else "未知问题"

        # 处理失败情况
        if state.get('error') and state.get('retry_count', 0) >= 3:
            # 使用 + 拼接，绝对不会解析内部的 % 或 '
            fail_msg = "抱歉，经过多次重试仍无法成功查询。错误原因：" + str(state['error']) + "。"
            return {"analysis": fail_msg, "messages": [AIMessage(content=fail_msg)], "retry_count": 0}

        # 处理成功情况

        # 使用 Template 保护分析阶段的 Prompt
        analysis_template = Template("""
        你是一个商业分析师。请基于以下查询结果回答用户问题：
        [当前问题]: $question
        [查询结果]: $result
        
        要求：
        1. 使用 LaTeX 渲染公式。
        2. 提供 3 条针对性的商业建议。
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