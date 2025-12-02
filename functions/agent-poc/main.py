import functions_framework
from langgraph.graph import StateGraph
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END
import langsmith as ls
from typing import TypedDict, Literal
from google.cloud import secretmanager
import os 

# --- Vertex AI setup ---
PROJECT_ID = "btibert-ba882-fall25"   # <-- update to your project
LOCATION = "us-central1"              # as needed
MODEL_ID = "gemini-2.5-flash"
SECRET_ID_LANGSMITH = 'LangSmith'   #<---------- this is the name of the secret you created
version_id = 'latest'

# instantiate the services 
sm = secretmanager.SecretManagerServiceClient()
name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID_LANGSMITH}/versions/{version_id}"
response = sm.access_secret_version(request={"name": name})
ls_token = response.payload.data.decode("UTF-8")

os.environ["LANGSMITH_TRACING"] = 'true'
os.environ["LANGSMITH_API_KEY"] = ls_token


llm = ChatVertexAI(
    model_name=MODEL_ID,
    project=PROJECT_ID,
    location=LOCATION,
)

# --- LangGraph state + node(s) ---

# Define state
class State(TypedDict):
    question: str
    response: str

# Create a simple node
def chat_node(state: State):
    response = llm.invoke(state["question"])
    return {"response": response.content}

# Build graph
graph = StateGraph(State)
graph.add_node("question", chat_node)
graph.set_entry_point("question")
graph.add_edge("question", END)

# Compile
app = graph.compile()


# --- Cloud Function HTTP entrypoint ---

@functions_framework.http
def task(request):
    # Handle simple GET / health / config probes
    if request.method == "GET":
        return ("ok", 200)

    # Only POST should trigger the LangGraph workflow
    _input = request.get_json(silent=True) or {}

    print("=== Received input payload ===")
    print(_input)
    print("================================")
    print(_input.keys())

    question = _input.get("question")

    if not question:
        # Don't try to call the LLM with None
        return ({"error": "Missing 'question' in request body"}, 400)

    with ls.tracing_context(enabled=True):
        result = app.invoke({"question": question})

    print("==================== result")
    print(result)

    return (
        {
            "question": result.get("question", question),
            "answer": result.get("response"),
        },
        200,
    )







