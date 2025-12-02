# imports
import streamlit as st
import requests

############################################## streamlit setup

st.image("https://questromworld.bu.edu/ftmba/wp-content/uploads/sites/42/2021/11/Questrom-1-1.png")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App layout
st.title("Streamlit as a GenAI Interface")
st.subheader("Great for Prototypes and POCs")

# Sidebar (just for demo text)
st.sidebar.header("Note")
st.sidebar.markdown("""
This application takes the LangGraph pipeline and puts an interface on top.  

It is a simple RAG application but highlights that:

1.  LangGraph can be run a in serverless environment.
1.  Our UI layer can be a simple Streamlit app sitting on top, also running serverless.
1.  Both are running on our GCP project.

> You could imagine using this sidebar to define use-cases, control the model(s) used, show past conversations, etc.

""")

############################################## Cloud Function setup

API_URL = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/agent-poc"


def call_agent(question: str) -> str:
    """Call the Cloud Function LangGraph endpoint and return the answer text."""
    resp = requests.post(API_URL, json={"question": question}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Expecting {"question": "...", "answer": "..."}
    return data.get("answer", "")


########################################################################
# Streamlit App - Simple Conversational Agent backed by Cloud Function
########################################################################

st.markdown("---")

# Replay existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat
    st.chat_message("user").markdown(prompt)
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call your Cloud Function / LangGraph agent
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = call_agent(prompt)
                st.markdown(answer)
        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        error_msg = f"Error calling Cloud Function: {e}"
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.messages.append({"role": 'assistant', 'content': error_msg})
