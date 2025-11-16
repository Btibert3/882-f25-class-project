# imports
import streamlit as st
from google import genai

############################################## streamlit setup


st.image("https://questromworld.bu.edu/ftmba/wp-content/uploads/sites/42/2021/11/Questrom-1-1.png")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App layout
st.title("Streamlit as a GenAI Interface")
st.subheader("Great for Prototypes and POCs")

# Sidebar filters for demo, this is not functional
st.sidebar.header("Inputs")
st.sidebar.markdown("One option is to use sidebars for inputs")

############################################## project setup
GCP_PROJECT = 'btibert-ba882-fall25'
GCP_REGION = "us-central1"


client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT,
    location=GCP_REGION
)


######################################################################## Streamlit App 1 - Simple Conversational Agent


# that chat model
if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(model='gemini-2.5-flash')

# helper to grab the response
def get_chat_response(chat, prompt: str) -> str:
    text_response = []
    for chunk in chat.send_message_stream(prompt):
        if chunk.text:
            text_response.append(chunk.text)
    return "".join(text_response)

st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # get a response from the chat session with GCP
    response = get_chat_response(st.session_state.chat, prompt)
    # playback the response
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})