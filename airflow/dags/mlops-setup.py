from datetime import datetime
from airflow.sdk import dag, task    
from pathlib import Path
import duckdb
import os
from ba882 import utils

# paths, as the airflow project is a project we deploy to astronomer
BASE_DIR = Path(os.environ.get("AIRFLOW_HOME", "/usr/local/airflow"))
SQL_DIR = BASE_DIR / "include" / "sql"

@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "setup"]
)
def mlops_setup():

    @task
    def setup_schema():
        s = utils.read_sql(SQL_DIR / "mlops-schema-setup.sql")
        utils.run_execute(s)
    
    setup_schema()

mlops_setup()