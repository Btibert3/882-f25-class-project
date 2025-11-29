from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import TypedDict, Any
import duckdb
import pandas as pd
import json

# --- setup ---
PROJECT_ID = "btibert-ba882-fall25"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-flash"

llm = ChatVertexAI(
    model_name=MODEL_ID,
    project=PROJECT_ID,
    location=LOCATION,
)

# --- state ---
class State(TypedDict):
    question: str
    request_chart: bool
    table_records: list[dict]
    col_records: list[dict]
    sql_query: str
    query_results: Any
    validation: str
    answer: str
    judge_evaluation: str
    judge_passed: bool
    needs_retry: bool
    retry_count: int

# --- structured output for SQL ---
class SQLQuery(BaseModel):
    sql: str

# --- helpers ---
def get_schema_context(md):
    """Get database schema for SQL generation."""
    # get all tables first to see what we have
    tables_query = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    ORDER BY table_schema, table_name
    LIMIT 100
    """
    schema_tables = md.sql(tables_query).df()
    
    # filter for nfl schemas
    if not schema_tables.empty:
        nfl_schemas = ['stage', 'gold', 'raw', 'mlops', 'ai_datasets']
        schema_tables = schema_tables[schema_tables['table_schema'].isin(nfl_schemas)]
    
    table_records = schema_tables.to_dict(orient="records")
    
    # get columns
    cols_query = """
    SELECT table_schema, table_name, column_name, data_type, ordinal_position
    FROM information_schema.columns
    ORDER BY table_name, ordinal_position
    LIMIT 500
    """
    schema_cols = md.sql(cols_query).df()
    
    # filter for nfl schemas
    if not schema_cols.empty:
        nfl_schemas = ['stage', 'gold', 'raw', 'mlops', 'ai_datasets']
        schema_cols = schema_cols[schema_cols['table_schema'].isin(nfl_schemas)]
    
    col_records = schema_cols.to_dict(orient='records')
    
    return table_records, col_records

@tool
def execute_sql(query: str, md_token: str) -> str:
    """Execute SQL query and return results as JSON."""
    md = duckdb.connect(f'md:nfl?motherduck_token={md_token}')
    try:
        df = md.sql(query).df()
        if df.empty:
            return json.dumps({"status": "success", "rows": 0, "data": []})
        result = df.to_dict(orient='records')
        return json.dumps({"status": "success", "rows": len(result), "data": result})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

# --- nodes ---
def schema_context_node(state: State, md_token: str) -> State:
    """Get database schema context."""
    md = duckdb.connect(f'md:nfl?motherduck_token={md_token}')
    table_records, col_records = get_schema_context(md)
    return {**state, "table_records": table_records, "col_records": col_records}

def chart_detection_node(state: State) -> State:
    """Detect if user wants a chart."""
    question = state["question"]
    keywords = ["show", "plot", "chart", "graph", "visualize", "display"]
    request_chart = any(k in question.lower() for k in keywords)
    
    if not request_chart:
        resp = llm.invoke(f"Does this question request a chart? {question} Answer yes or no.")
        request_chart = "yes" in resp.content.lower()
    
    return {**state, "request_chart": request_chart}

def sql_generation_node(state: State) -> State:
    """Generate SQL from question."""
    question = state["question"]
    table_records = state.get("table_records", [])
    col_records = state.get("col_records", [])
    previous_sql = state.get("sql_query", "")
    validation_feedback = state.get("validation", "")
    
    prompt = f"""### Tables in the database
{table_records}

### Column level detail in the database
{col_records}

### User prompt
{question}"""
    
    if validation_feedback and "No results" in validation_feedback:
        prompt += f"""

### Previous SQL attempt (returned no results):
{previous_sql}

### Feedback:
{validation_feedback}

Generate a corrected SQL query:"""
    else:
        prompt += """

### SQL query to answer the question above based on the database schema"""
    
    # use structured output to get clean SQL string
    structured_llm = llm.with_structured_output(SQLQuery)
    resp = structured_llm.invoke(prompt)
    sql_query = resp.sql.strip()
    
    # clean up any remaining backticks or markdown
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    
    return {**state, "sql_query": sql_query}

def sql_execution_node(state: State, md_token: str) -> State:
    """Execute SQL query."""
    sql_query = state.get("sql_query", "")
    if not sql_query:
        return state
    
    result_json = execute_sql.invoke({"query": sql_query, "md_token": md_token})
    result = json.loads(result_json)
    
    if result["status"] == "error":
        return state
    
    df = pd.DataFrame(result["data"]) if result["data"] else pd.DataFrame()
    return {**state, "query_results": df}

def validation_node(state: State) -> State:
    """Validate query results and determine if retry needed."""
    question = state["question"]
    sql_query = state.get("sql_query", "")
    query_results = state.get("query_results")
    retry_count = state.get("retry_count", 0)
    
    if query_results is None or query_results.empty:
        validation = "No results returned. SQL may need correction."
        needs_retry = retry_count < 2  # limit to 2 retries
    else:
        prompt = f"""Validate if this SQL answers the question.

Question: {question}
SQL: {sql_query}
Rows: {len(query_results)}
Columns: {', '.join(query_results.columns.tolist())}
Sample: {query_results.head(2).to_string()}

Brief validation:"""
        resp = llm.invoke(prompt)
        validation = resp.content
        needs_retry = False
    
    return {**state, "validation": validation, "needs_retry": needs_retry, "retry_count": retry_count + 1 if needs_retry else retry_count}

def answer_generation_node(state: State) -> State:
    """Generate answer from results."""
    question = state["question"]
    sql_query = state.get("sql_query", "")
    query_results = state.get("query_results")
    validation = state.get("validation", "")
    
    if query_results is None or query_results.empty:
        answer = "No data found. Try rephrasing your question."
    else:
        results_str = query_results.to_string()
        
        prompt = f"""Answer this question based on the results:

Question: {question}
SQL: {sql_query}
Validation: {validation}
Results ({len(query_results)} rows):
{results_str}

Provide a clear answer with all data points:"""
        resp = llm.invoke(prompt)
        answer = resp.content
    
    return {**state, "answer": answer}

def judge_node(state: State) -> State:
    """Judge if answer aligns with question."""
    question = state["question"]
    answer = state.get("answer", "")
    
    prompt = f"""Evaluate if this answer addresses the question.

Question: {question}
Answer: {answer}

Respond with "PASS" or "FAIL" and brief explanation:"""
    
    resp = llm.invoke(prompt)
    evaluation = resp.content
    judge_passed = "PASS" in evaluation.upper()
    
    return {**state, "judge_evaluation": evaluation, "judge_passed": judge_passed}

# --- build graph ---
def create_workflow(md_token: str):
    """Create and return compiled LangGraph workflow."""
    
    def schema_node(state: State) -> State:
        return schema_context_node(state, md_token)
    
    def exec_node(state: State) -> State:
        return sql_execution_node(state, md_token)
    
    graph = StateGraph(State)
    graph.add_node("schema_context", schema_node)
    graph.add_node("chart_detection", chart_detection_node)
    graph.add_node("sql_generation", sql_generation_node)
    graph.add_node("sql_execution", exec_node)
    graph.add_node("validation", validation_node)
    graph.add_node("answer_generation", answer_generation_node)
    graph.add_node("judge", judge_node)
    
    def should_retry(state: State) -> str:
        """Check if SQL needs to be regenerated."""
        return "sql_generation" if state.get("needs_retry", False) else "answer_generation"
    
    graph.set_entry_point("schema_context")
    graph.add_edge("schema_context", "chart_detection")
    graph.add_edge("chart_detection", "sql_generation")
    graph.add_edge("sql_generation", "sql_execution")
    graph.add_edge("sql_execution", "validation")
    graph.add_conditional_edges("validation", should_retry)
    graph.add_edge("answer_generation", "judge")
    graph.add_edge("judge", END)
    
    return graph.compile()

