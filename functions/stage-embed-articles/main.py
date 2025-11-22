# this function will get unprocessed articles from the stage table and embed them into pinecone

import functions_framework
from google.cloud import secretmanager
import duckdb
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
from bs4 import BeautifulSoup
import re
import json

# settings
project_id = 'btibert-ba882-fall25'
secret_id = 'MotherDuck'   #<---------- this is the name of the secret you created
version_id = 'latest'

# db setup
db = 'nfl'
schema = "stage"
db_schema = f"{db}.{schema}"

@functions_framework.http
def task(request):

    # instantiate the services 
    sm = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version in order to get the MotherDuck token
    response = sm.access_secret_version(request={"name": name})
    md_token = response.payload.data.decode("UTF-8")

    # initiate the MotherDuck connection through an access token through
    # this syntax lets us connect to our motherduck cloud warehouse and execute commands via the duckdb library

    # md = duckdb.connect(f'md:?motherduck_token={md_token}') 

    # pinecone secret
    name = f"projects/{project_id}/secrets/Pinecone/versions/{version_id}"
    response = sm.access_secret_version(request={"name": name})
    pinecone_token = response.payload.data.decode("UTF-8")


    ##################################################### connect to vertexai
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location="us-central1",
    )


    ##################################################### take the input
    payload = request.get_json(silent=True) or {}

    print("=== Received article payload ===")
    print(payload)
    print("Type:", type(payload))
    print("Keys:", list(payload.keys()))
    print("================================")

    # quick sanity response
    return (
        json.dumps({"ok": True, "received_keys": list(payload.keys())}),
        200,
        {"Content-Type": "application/json"},
    )


    ##################################################### 

    # # get the headlines
    # headlines = articles.headline.tolist()

    # response = client.models.embed_content(
    #     model='gemini-embedding-001',
    #     contents=headlines,
    #     config=types.EmbedContentConfig(output_dimensionality=768,
    #                                     task_type="RETRIEVAL_DOCUMENT"),
    # )
    # headline_embeddings = [e.values for e in response.embeddings]

    # # helper to clean the stories
    # def clean_html_and_whitespace(text):
    #     if not isinstance(text, str):
    #         return text
    #     # Strip HTML
    #     txt = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    #     # Collapse repeated dashes and whitespace
    #     txt = re.sub(r"\s+", " ", txt).strip()
    #     return txt

    # articles['story'] = articles["story"].apply(clean_html_and_whitespace)
    # articles_list = articles.to_dict(orient="records")

    # # chunk the stories
    # chunk_docs = []
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=350,
    #     chunk_overlap=75,
    #     length_function=len,
    #     is_separator_regex=False,
    # )

    # for each doc, create chunks, and create a set of embeddings with metadata
    # https://github.com/Btibert3/BA882-Fall24-InClass-Project/blob/main/genai/pipeline/functions/ingestor/main.py#L78




    

    # return a dictionary/json entry, its blank because are not returning data, 200 for success
    # return {}, 200