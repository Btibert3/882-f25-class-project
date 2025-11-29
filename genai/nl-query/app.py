import streamlit as st
import pandas as pd
from google.cloud import secretmanager
import langsmith as ls
from langsmith import uuid7
from workflow import create_workflow, State
import streamlit_mermaid as stmd
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
st.caption("Ask questions about your NFL data in plain English")

debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# workflow graph in sidebar
st.sidebar.header("Workflow Graph")
if "workflow" not in st.session_state:
    st.session_state.workflow = create_workflow(md_token)

try:
    graph_mermaid = st.session_state.workflow.get_graph().draw_mermaid()
    stmd.st_mermaid(graph_mermaid)
except:
    st.sidebar.write("Graph not available")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the database..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            initial_state: State = {
                "question": prompt,
                "request_chart": False,
                "schema_context": "",
                "sql_query": "",
                "query_results": pd.DataFrame(),
                "validation": "",
                "answer": "",
                "judge_evaluation": "",
                "judge_passed": False
            }
            
            with ls.tracing_context(enabled=True, run_id=str(uuid7())):
                result = st.session_state.workflow.invoke(initial_state)
            
            # answer
            st.markdown(result["answer"])
            
            # SQL
            with st.expander("Generated SQL"):
                st.code(result["sql_query"], language="sql")
            
            # chart/data if requested
            query_results = result.get("query_results")
            if result.get("request_chart") and query_results is not None and not query_results.empty:
                st.dataframe(query_results, use_container_width=True)
                if len(query_results.columns) >= 2:
                    st.line_chart(query_results.set_index(query_results.columns[0]))
            
            # debug mode
            if debug_mode:
                st.divider()
                st.subheader("Debug Information")
                
                st.write("**Judge Evaluation:**")
                st.info(result.get("judge_evaluation", "N/A"))
                st.write(f"**Passed:** {result.get('judge_passed', False)}")
                
                with st.expander("Full State"):
                    state_dict = {k: str(v) if isinstance(v, pd.DataFrame) else v for k, v in result.items()}
                    st.json(state_dict)
                
                st.write("**Workflow Graph:**")
                st.info("See sidebar for interactive graph visualization")
                
                st.write("**LangSmith Tracing:**")
                st.info("Check LangSmith dashboard for detailed workflow tracing")
        
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
