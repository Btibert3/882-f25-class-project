from airflow.decorators import dag, task
from datetime import datetime, timedelta
from pathlib import Path
import duckdb
import os
from ba882 import utils
import requests

# paths, as the airflow project is a project we deploy to astronomer
BASE_DIR = Path(os.environ.get("AIRFLOW_HOME", "/usr/local/airflow"))
SQL_DIR = BASE_DIR / "include" / "sql"


@dag(
    schedule=None,       
    start_date=datetime(2025, 9, 4),
    catchup=False,  # if True, when the DAG is activated, Airflow will coordinate backfills
    max_active_runs = 1,  # if backfilling, this will seralize the work 1x per backfill date/job
    tags=["stage", "genai", "pinecone"]
)
def nfl_article_processing():
    
    @task
    def setup_article_schema():
        s = utils.read_sql(SQL_DIR / "stage-article-proccessing.sql")
        utils.run_execute(s)

    @task
    def setup_image_schema():
        s = utils.read_sql(SQL_DIR / "stage-article-image-processing.sql")
        utils.run_execute(s)
    
    @task(retries=1, retry_delay=timedelta(seconds=30))
    def get_new_articles():
        # retrieve articles that aren't in the processed table
        s = utils.read_sql(SQL_DIR / "unprocessed-articles.sql")
        print(s)
        articles = utils.get_df(s)
        print(f"articles shape: {articles.shape}")
        # Convert datetime columns to strings for XCom JSON serialization
        datetime_cols = articles.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        for col in datetime_cols:
            articles[col] = articles[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return articles.head(10).to_dict(orient='records') # I know that our domain will always have a small amount of data, so keeping it in XCOM but we **should** write this to gcs for playback

    @task(retries=1, retry_delay=timedelta(seconds=30), max_active_tis_per_dag=5)
    def parse_article(article):
        URL = "https://genai-ingest-articles-7qsihlqmoa-uc.a.run.app"
        
        # Send the article as JSON
        resp = requests.post(URL, json=article)
        resp.raise_for_status()

        print("Cloud Run response status:", resp.status_code)
        print("Cloud Run response body:", resp.text)

        return {"status_code": resp.status_code}

    
    # wiring
    _setup_article = setup_article_schema()
    _setup_article_images = setup_image_schema()
    article_list = get_new_articles()

    _setup_article >> _setup_article_images >> article_list
    
    # map over the articles by calling a function processor
    mapped_parse = parse_article.expand(article=article_list)

    article_list >> mapped_parse

    
    

    
nfl_article_processing()