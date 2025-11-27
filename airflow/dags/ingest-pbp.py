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
    catchup=False,  
    max_active_runs = 1,  
    tags=["stage", "genai", "pinecone"]
)
def pbp_ingestion():

    @task
    def setup_schema():
        s = utils.read_sql(SQL_DIR / "pbp-schemas.sql")
        utils.run_execute(s)

pbp_ingestion()