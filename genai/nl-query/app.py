import streamlit as st
import pandas as pd
from google.cloud import secretmanager
import langsmith as ls
from langsmith import uuid7
from workflow import create_workflow, State
import os

# --- setup ---
PROJECT_ID = "btibert-ba882-fall25"
SECRET_ID_LANGSMITH = 'LangSmith'
SECRET_ID_MOTHERDUCK = 'MotherDuck'
version_id = 'latest'

sm = secretmanager.SecretManagerServiceClient()

# langsmith
name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID_LANGSMITH}/versions/{version_id}"
response = sm.access_secret_version(request={"name": name})
ls_token = response.payload.data.decode("UTF-8")

os.environ["LANGSMITH_TRACING"] = 'true'
os.environ["LANGSMITH_API_KEY"] = ls_token

# motherduck
name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID_MOTHERDUCK}/versions/{version_id}"
response = sm.access_secret_version(request={"name": name})
md_token = response.payload.data.decode("UTF-8")

# --- Streamlit UI ---
st.set_page_config(page_title="NL Query", layout="wide")
st.title("Natural Language Database Query")
st.caption("Ask questions about our in-class project data (the NFL season) in Natural Language!")

st.sidebar.header("Note")
st.sidebar.markdown("""
This application uses a more "complex" LangGraph pipeline.

It is far from perfect, but aims to highlight:

1.  We _can_ use advanced logic to talk to our internal systems and reason about answers.
1.  Evaluations are important, and how we wire up these pipelines can have an impact on the end user experience (UX)
1.  Consider breaking down the problems into smaller units such that a given agent in the pipeline does one thing, and one thing well (e.g. plotting)

> NOTE:  This pipeline also is logging traces to Langsmith for observability.  We can also use these traces to build datasets for further evaluation!.

""")

if "workflow" not in st.session_state:
    st.session_state.workflow = create_workflow(md_token)

if "messages" not in st.session_state:
    st.session_state.messages = []

# store previous query results for multi-turn conversations
if "previous_query_results" not in st.session_state:
    st.session_state.previous_query_results = None
if "previous_sql" not in st.session_state:
    st.session_state.previous_sql = ""

# cache schema context for multi-turn conversations
if "schema_context" not in st.session_state:
    from workflow import get_schema_context
    import duckdb
    md = duckdb.connect(f'md:nfl?motherduck_token={md_token}')
    table_records, col_records = get_schema_context(md)
    st.session_state.schema_context = {"table_records": table_records, "col_records": col_records}

# replay messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# react to user input
if prompt := st.chat_input("Ask a question about the database..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # Build conversation history from messages (last 6 messages = 3 exchanges)
                conversation_history = []
                for msg in st.session_state.messages[-6:]:
                    conversation_history.append({"role": msg["role"], "content": msg["content"]})
                
                initial_state: State = {
                    "question": prompt,
                    "request_chart": False,
                    "chart_type": "line",
                    "table_records": st.session_state.schema_context["table_records"],
                    "col_records": st.session_state.schema_context["col_records"],
                    "sql_query": "",
                    "query_results": pd.DataFrame(),
                    "validation": "",
                    "answer": "",
                    "judge_evaluation": "",
                    "judge_passed": False,
                    "needs_retry": False,
                    "retry_count": 0,
                    "conversation_history": conversation_history,
                    "previous_query_results": st.session_state.previous_query_results,
                    "previous_sql": st.session_state.previous_sql
                }
                
                with ls.tracing_context(enabled=True, run_id=str(uuid7())):
                    result = st.session_state.workflow.invoke(initial_state)
                
                # answer
                st.markdown(result["answer"])
                
                # SQL
                with st.expander("Generated SQL"):
                    st.code(result["sql_query"], language="sql")
                
                # Get query results from the workflow result
                query_results = result.get("query_results")
                
                # chart/data if requested
                if result.get("request_chart") and query_results is not None and not query_results.empty:
                    st.dataframe(query_results, use_container_width=True)
                    if len(query_results.columns) >= 2:
                        chart_type = result.get("chart_type", "line")
                        df_indexed = query_results.set_index(query_results.columns[0])
                        
                        if chart_type == "bar":
                            st.bar_chart(df_indexed)
                        elif chart_type == "scatter":
                            st.scatter_chart(df_indexed)
                        else:  # default to line
                            st.line_chart(df_indexed)
                
                # Store results for next turn
                if query_results is not None and not query_results.empty:
                    st.session_state.previous_query_results = query_results
                    if result.get("sql_query"):
                        st.session_state.previous_sql = result["sql_query"]
                
                # debug mode
                # if debug_mode:
                #     st.divider()
                #     st.subheader("Debug Information")
                #     st.write("**Judge Evaluation:**")
                #     st.info(result.get("judge_evaluation", "N/A"))
                #     st.write(f"**Passed:** {result.get('judge_passed', False)}")
                #     with st.expander("Full State"):
                #         state_dict = {k: str(v) if isinstance(v, pd.DataFrame) else v for k, v in result.items()}
                #         st.json(state_dict)
                #     st.write("**LangSmith Tracing:**")
                #     st.info("Check LangSmith dashboard for detailed workflow tracing")
        
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    except Exception as e:
        error_msg = f"Error processing query: {e}"
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
