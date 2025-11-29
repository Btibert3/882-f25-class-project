import functions_framework
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END
import langsmith as ls
from typing import TypedDict, Literal
from google.cloud import secretmanager
from google import genai
from google.genai import types
from pinecone import Pinecone
import os 

# --- setup ---
PROJECT_ID = "btibert-ba882-fall25"   # <-- update to your project
LOCATION = "us-central1"              # as needed
MODEL_ID = "gemini-2.5-flash"
SECRET_ID_LANGSMITH = 'LangSmith'   #<---------- this is the name of the secret you created
version_id = 'latest'
VECTOR_INDEX = 'nfl-articles'
VEC_SIZE = 768
NUM_RESULTS = 25

# instantiate the secret manager service
sm = secretmanager.SecretManagerServiceClient()

# langsmith
name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID_LANGSMITH}/versions/{version_id}"
response = sm.access_secret_version(request={"name": name})
ls_token = response.payload.data.decode("UTF-8")

# pinecone secret
name = f"projects/{PROJECT_ID}/secrets/Pinecone/versions/{version_id}"
response = sm.access_secret_version(request={"name": name})
pinecone_token = response.payload.data.decode("UTF-8")

os.environ["LANGSMITH_TRACING"] = 'true'
os.environ["LANGSMITH_API_KEY"] = ls_token

llm = ChatVertexAI(
    model_name=MODEL_ID,
    project=PROJECT_ID,
    location=LOCATION,
)

client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )


pc = Pinecone(api_key=pinecone_token)
index = pc.Index(VECTOR_INDEX)

# --- LangGraph state + node(s) ---
class State(TypedDict):
    question: str
    contexts: list[dict]      # retrieved chunks: {article_id, chunk_text, score, id}
    articles: list[int]       # article_ids from the retrieved chunks (for eval)
    response: str             # final LLM answer

def retrieve_node(state: State) -> State:
    question = state["question"]

    # embed
    embed_resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[question],
        config=types.EmbedContentConfig(
            output_dimensionality=VEC_SIZE,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    embedding = embed_resp.embeddings[0].values

    # pinecone
    res = index.query(
        vector=embedding,
        top_k=NUM_RESULTS,
        include_metadata=True,
    )

    contexts = []
    articles = []
    for match in res.matches:
        md = match.metadata or {}
        aid = md.get("article_id")
        chunk_text = md.get("chunk_text", "")

        contexts.append(
            {
                "id": match.id,
                "score": match.score,
                "article_id": aid,
                "chunk_text": chunk_text,
            }
        )
        if aid is not None:
            articles.append(aid)

    return {
        **state,
        "contexts": contexts,
        "articles": articles,
    }

def answer_node(state: State) -> State:
    question = state["question"]
    contexts = state.get("contexts", [])

    context_text = "\n\n".join(
        f"[article_id={c.get('article_id')}] {c.get('chunk_text', '')}"
        for c in contexts
    )

    prompt = f"""
    You are an NFL analyst. Use ONLY the context below to answer the question.
    If the context is insufficient or does not match the question, say so.

    Question:
    {question}

    Context (snippets from articles):
    {context_text}
    """.strip()

    resp = llm.invoke(prompt)
    answer = resp.content

    return {
        **state,
        "response": answer,
    }


graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

app = graph.compile()

# --- Cloud Function HTTP entrypoint ---

@functions_framework.http
def task(request):
    if request.method == "GET":
        return ("ok", 200)

    _input = request.get_json(silent=True) or {}
    question = _input.get("question")

    if not question:
        return ({"error": "Missing 'question' in request body"}, 400)

    initial_state: State = {
        "question": question,
        "contexts": [],
        "articles": [],
        "response": "",
    }

    with ls.tracing_context(enabled=True):
        result = app.invoke(initial_state)

    return (
        {
            "question": result["question"],
            "answer": result["response"],
            "articles": result["articles"],  
            "contexts": result["contexts"],     # for evaluation later
        },
        200,
    )






