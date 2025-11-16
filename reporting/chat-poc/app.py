import streamlit as st

st.set_page_config(page_title="POC GenAI Data Apps", layout="wide")

pg = st.navigation([
    st.Page("assistant.py", title="Simple Chat Assistant", icon=":material/chat:"), 
    st.Page("chunking.py", title="LlamaIndex Intuition", icon=":material/text_snippet:"),
    st.Page("doc-compare.py", title="Document Comparison", icon=":material/assignment:")
    ])
pg.run()

