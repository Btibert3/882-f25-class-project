# imports
import streamlit as st 
import json
from google.cloud import secretmanager
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
import duckdb
from typing import List, Literal
from pydantic import BaseModel
from annotated_text import annotated_text


GCP_PROJECT = 'btibert-ba882-fall25'
GCP_REGION = "us-central1"
version_id = 'latest'
vector_secret = "Pinecone"
vector_index = 'nfl-articles'
VEC_SIZE = 768
EMBEDDING_MODEL = "gemini-embedding-001"

# secret manager pinecone
sm = secretmanager.SecretManagerServiceClient()
vector_name = f"projects/{GCP_PROJECT}/secrets/{vector_secret}/versions/{version_id}"
response = sm.access_secret_version(request={"name": vector_name})
pinecone_token = response.payload.data.decode("UTF-8")
pc = Pinecone(api_key=pinecone_token)
index = pc.Index(vector_index)
# print(f"index stats: {index.describe_index_stats()}")

# secret manager motherduck
db_name = f"projects/{GCP_PROJECT}/secrets/MotherDuck/versions/{version_id}"
response = sm.access_secret_version(request={"name": db_name})
md_token = response.payload.data.decode("UTF-8")
md = duckdb.connect(f'md:?motherduck_token={md_token}')

# vertex ai
client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT,
    location=GCP_REGION,
)


############################################## streamlit setup


st.image("https://questromworld.bu.edu/ftmba/wp-content/uploads/sites/42/2021/11/Questrom-1-1.png")
st.title("RAG: Article Retrieval")


################### sidebar <---- I know, we can do other layouts, easy for POCs

st.sidebar.title("Search the Game Articles")

search_query = st.sidebar.text_area("What are you looking to learn?")

top_k = st.sidebar.slider("Top K", 1, 15, step=1)

search_button = st.sidebar.button("Run the Retrieval pipeline")


################### main

# Main action: Handle search
if search_button:
    if search_query.strip():
        # Get embedding
        response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=[search_query],
            config=types.EmbedContentConfig(output_dimensionality=VEC_SIZE,
                                            task_type="RETRIEVAL_QUERY"),
        )
        embedding = response.embeddings[0].values

        # search pincone
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )

        # answer the question
        chunks = [r.metadata['chunk_text'] for r in results.matches]
        # print(results)
        context = "\n".join(chunks)

        prompt_template = f"""
            You are an AI assistant trained to provide detailed and accurate answers based on the provided context.
            Use the context below to respond to the query.
            If the context does not contain sufficient information to answer, state that explicitly and avoid making up an answer.

            ### Context:
            {context}

            ### Query:
            {search_query}

            ### Your Response:
        """

        response = resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_template
        )
        print(resp.text)

        # Display the results
        st.write(response.text)

        # return the full document from just the first entry - 
        top_article = results.matches[0]['metadata']['article_id']
        story = results.matches[0]['metadata']['story']
        headline = results.matches[0]['metadata']['headline']

        # the full blog post
        st.markdown("## The 'Top Ranked Article' from Pinecone")
        st.info("This is the parent article for the top chunk")
        st.markdown(f"### {headline}")
        # st.pills('Tags', options=page.tags.to_list()[0].split(','))
        st.markdown(story, unsafe_allow_html=True)

        # now lets pull out entities
        class Person(BaseModel):
            name: str
            team: str
            sentiment: Literal['Very Good Game','Good Game', 'Netural', 'Bad Game', 'Very Bad Game','Not Applicable']
            role: Literal['Player', 'Coach', 'Referee', 'Other']

        class EntityExtractionResult(BaseModel):
            entities: List[Person]

        system = """ 
        You are an expert at Named Entity Recognition for sports, in particular, the NFL. Retrieve the entities from the aritle provided and associate each eith a team.
        If the entity is not part of a team, you can use NO_TEAM.
        In addition, for each entity extracted, estimate the sentiment of their peformance based on the article.
        """
        ner_resp = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=story,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type='application/json',
                response_schema=EntityExtractionResult,
            ),
        )

        ents = json.loads(ner_resp.text).get('entities')

        sentiment_colors = {
            "Very Good Game": "#1a7f37",   # dark green
            "Good Game": "#4caf50",        # medium green
            "Neutral": "#9e9e9e",          # gray
            "Netural": "#9e9e9e",          # fix typo from model
            "Bad Game": "#ef5350",         # red
            "Very Bad Game": "#b71c1c",    # dark red
            "Not Applicable": "#bdbdbd",   # light gray
        }

        st.markdown("## Entity Highlight View")

        if not ents:
            st.info("No entities were extracted from this article.")
        else:
            blocks = []

            for e in ents:
                name = e.get("name", "Unknown")
                team = e.get("team", "NO_TEAM")
                sentiment = e.get("sentiment", "Not Applicable")
                role = e.get("role", "Other")

                # Normalize typo
                if sentiment == "Netural":
                    sentiment = "Neutral"

                color = sentiment_colors.get(sentiment, "#bdbdbd")

                label = f"{team} – {sentiment}"

                # Each annotated chunk: (text, label, background_color)
                blocks.append((name, label, color))
                # Add a separator so it's readable
                blocks.append("  \n  ")

            # Remove trailing separator if present
            if blocks and isinstance(blocks[-1], str):
                blocks = blocks[:-1]

            annotated_text(*blocks)


    else:
        st.warning("Please enter a search query!")