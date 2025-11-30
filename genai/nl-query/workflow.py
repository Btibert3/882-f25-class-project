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
    chart_type: str  # 'line', 'bar', etc.
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
    conversation_history: list[dict]  # Previous Q&A pairs
    previous_query_results: Any  # Previous query results DataFrame
    previous_sql: str  # Previous SQL query

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
    """Detect if user wants a chart and chart type modifications."""
    question = state["question"].lower()
    previous_results = state.get("previous_query_results")
    
    # Check for chart modification requests (bar, line, pie, etc.)
    chart_type = None
    if "bar chart" in question or "bar graph" in question:
        chart_type = "bar"
    elif "line chart" in question or "line graph" in question:
        chart_type = "line"
    elif "pie chart" in question:
        chart_type = "pie"
    elif "scatter" in question:
        chart_type = "scatter"
    
    # Check if this is a visualization modification request
    modification_keywords = ["make", "change", "convert", "switch", "show as", "display as"]
    is_modification = any(k in question for k in modification_keywords) and chart_type is not None
    
    # If it's a modification request and we have previous results, we can reuse them
    if is_modification and previous_results is not None and not previous_results.empty:
        request_chart = True
        return {**state, "request_chart": request_chart, "chart_type": chart_type}
    
    # Otherwise, detect if user wants a chart
    keywords = ["show", "plot", "chart", "graph", "visualize", "display"]
    request_chart = any(k in question for k in keywords)
    
    if not request_chart:
        resp = llm.invoke(f"Does this question request a chart? {state['question']} Answer yes or no.")
        request_chart = "yes" in resp.content.lower()
    
    # Detect chart type from question if not already set
    if request_chart and chart_type is None:
        if "bar" in question:
            chart_type = "bar"
        elif "line" in question:
            chart_type = "line"
        else:
            chart_type = "line"  # default
    
    return {**state, "request_chart": request_chart, "chart_type": chart_type or "line"}

def sql_generation_node(state: State) -> State:
    """Generate SQL from question with conversation context."""
    question = state["question"]
    table_records = state.get("table_records", [])
    col_records = state.get("col_records", [])
    previous_sql = state.get("sql_query", "")
    validation_feedback = state.get("validation", "")
    conversation_history = state.get("conversation_history", [])
    previous_results = state.get("previous_query_results")
    
    # Check if this is just a visualization modification request
    question_lower = question.lower()
    modification_keywords = ["make", "change", "convert", "switch", "show as", "display as"]
    is_modification = any(k in question_lower for k in modification_keywords) and (
        "chart" in question_lower or "graph" in question_lower
    )
    
    # If it's just a chart type modification and we have previous results, reuse previous SQL
    if is_modification and previous_results is not None and not previous_results.empty:
        prev_sql = state.get("previous_sql", "")
        if prev_sql:
            return {**state, "sql_query": prev_sql}
    
    # Build prompt with conversation history
    prompt = f"""### Tables in the database
{table_records}

### Column level detail in the database
{col_records}"""
    
    # Add conversation history context
    if conversation_history:
        prompt += "\n\n### Previous conversation context:"
        for i, msg in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"\n\nUser question {i}: {content}"
            elif role == "assistant":
                prompt += f"\n\nAssistant answer {i}: {content[:500]}"  # Truncate long answers
        
        # Add previous SQL if available
        if state.get("previous_sql"):
            prompt += f"\n\nPrevious SQL query: {state.get('previous_sql')}"
        
        # Add information about previous results if available
        if previous_results is not None and not previous_results.empty:
            prompt += f"""
Previous query returned {len(previous_results)} rows with columns: {', '.join(previous_results.columns.tolist()[:5])}
Sample data: {previous_results.head(3).to_string()}"""
    
    prompt += f"""

### Current user prompt
{question}"""
    
    if validation_feedback and "No results" in validation_feedback:
        prompt += f"""

### Previous SQL attempt (returned no results):
{previous_sql}

### Feedback:
{validation_feedback}

Generate a corrected SQL query:"""
    else:
        if conversation_history:
            prompt += """

IMPORTANT: This is a follow-up question. Consider the previous conversation context when generating SQL.
If the user is asking about the same data from a previous query, you may need to reuse or modify the previous SQL."""
        prompt += """

### SQL query to answer the question above based on the database schema"""
    
    # use structured output to get clean SQL string
    try:
        structured_llm = llm.with_structured_output(SQLQuery)
        resp = structured_llm.invoke(prompt)
        if resp is None or not hasattr(resp, 'sql'):
            # fallback to regular LLM if structured output fails
            resp_text = llm.invoke(prompt)
            sql_query = resp_text.content.strip()
        else:
            sql_query = resp.sql.strip()
    except Exception as e:
        # fallback to regular LLM if structured output fails
        resp_text = llm.invoke(prompt)
        sql_query = resp_text.content.strip()
    
    # clean up any remaining backticks or markdown
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    
    return {**state, "sql_query": sql_query}

def sql_execution_node(state: State, md_token: str) -> State:
    """Execute SQL query, or reuse previous results if this is a visualization-only request."""
    sql_query = state.get("sql_query", "")
    previous_results = state.get("previous_query_results")
    question = state.get("question", "").lower()
    
    # Check if this is a visualization modification request and we have previous results
    modification_keywords = ["make", "change", "convert", "switch", "show as", "display as"]
    is_viz_modification = any(k in question for k in modification_keywords) and (
        "chart" in question or "graph" in question
    )
    
    # If it's just a visualization change and we have previous results, reuse them
    if is_viz_modification and previous_results is not None and not previous_results.empty:
        return {**state, "query_results": previous_results}
    
    # Otherwise, execute the SQL query
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
    chart_type = state.get("chart_type", "line")
    
    # Check if this is a visualization modification request
    question_lower = question.lower()
    modification_keywords = ["make", "change", "convert", "switch", "show as", "display as"]
    is_viz_modification = any(k in question_lower for k in modification_keywords) and (
        "chart" in question_lower or "graph" in question_lower
    )
    
    if query_results is None or query_results.empty:
        answer = "No data found. Try rephrasing your question."
    else:
        results_str = query_results.to_string()
        
        if is_viz_modification:
            # For visualization modifications, provide a simpler answer
            chart_type_name = chart_type.replace("_", " ").title()
            answer = f"Updated the visualization to a {chart_type_name} chart. The data is displayed below."
        else:
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

