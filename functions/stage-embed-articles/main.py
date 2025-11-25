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
import pandas as pd

# settings
project_id = 'btibert-ba882-fall25'
secret_id = 'MotherDuck'   #<---------- this is the name of the secret you created
version_id = 'latest'

# db setup
db = 'nfl'
schema = "stage"
db_schema = f"{db}.{schema}"
tbl_name = "article_embeddings"

# vector db setup
vector_db = 'nfl-articles'
VEC_SIZE = 768

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
    md = duckdb.connect(f'md:?motherduck_token={md_token}') 

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
    ## this expects to be posted a dictionary of k/v pairs, where the dictionary is a single record
    article = request.get_json(silent=True) or {}

    print("=== Received article payload ===")
    print(article)
    print("================================")
    print(article.keys())

    def clean_html_and_whitespace(text):
        if not isinstance(text, str):
            return text
        # Strip HTML
        txt = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        # Collapse repeated dashes and whitespace
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    story = clean_html_and_whitespace(article.get('story'))

    # chunk the story
    chunk_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=75,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([story])
    print(f"Number of chunks: {len(chunks)}   ======================")

    # generate the chunk dataset
    _id = article.get('id')
    for cid, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        resp = client.models.embed_content(
            model='gemini-embedding-001',
            contents=chunk_text,
            config=types.EmbedContentConfig(output_dimensionality=VEC_SIZE,
                                            task_type="RETRIEVAL_DOCUMENT"),
        )
        e = resp.embeddings[0].values
        chunk_doc = {
            'id': str(_id) + '_' + str(cid),
            'values': e,
            'metadata': {
                'published': article.get('published'),
                'chunk_index': cid,
                'article_id': _id,
                'chunk_text': chunk_text,
                'headline': article.get('headline'),
                'story': story
            }
        }
        chunk_docs.append(chunk_doc)

    # flatten to dataframe for the ingestion
    # this is the format pinecone is looking for, the id, the embedding, and the metadata
    # these three columns are the core items needed, and it takes advantage of the object type to store the metadata.  objects can be more than strings!
    chunk_df = pd.DataFrame(chunk_docs)

    # connect to pinecone
    pc = Pinecone(api_key=pinecone_token)
    if not pc.has_index(vector_db):
        pc.create_index(
            name=vector_db,
            dimension=VEC_SIZE,  
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', # <- gcp is not part of the free usage
                region='us-east-1' # <- us-central1 <- not part of free
            )
        )
    index = pc.Index(vector_db)
    print("#==============================================")
    print(f"index stats: {index.describe_index_stats()}")
    print("#==============================================")

    
    # upsert to pinecone in batches from a dataframe
    # we _should_ log this below to GCS, there are a few corners cut here in order to reduce the surface area of the intuition
    index.upsert_from_dataframe(chunk_df, batch_size=100)

    # log this to motherduck -- not the exact best use case for an OLAP store
    sql = f"""
    INSERT OR REPLACE INTO {db_schema}.{tbl_name} (article_id)
    VALUES (?)
    """
    md.execute(sql, [article["id"]] )
    print("finsihed adding record to Motherduck, this is not the best approach but it slides by for this domain")
    print("THOUGHT EXERCISE FOR BA882: why is writing a single record to a cloud data warehouse table like this not ideal?")

    # metadata could/should be added here
    return {}, 200
