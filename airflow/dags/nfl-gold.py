from datetime import datetime
from airflow.sdk import dag, task    
from pathlib import Path
import duckdb
import os
from ba882 import utils


# paths, as the airflow project is a project we deploy to astronomer
BASE_DIR = Path(os.environ.get("AIRFLOW_HOME", "/usr/local/airflow"))
SQL_DIR = BASE_DIR / "include" / "sql"

# ---------------------------------------------------------------------
#  DAG Definition
# ---------------------------------------------------------------------
@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold"]
)
def gold():

    @task
    def setup_schema():
        s = utils.read_sql(SQL_DIR / "gold-schema-setup.sql")
        utils.run_execute(s)

    @task
    def incremental():
        s = utils.read_sql(SQL_DIR / "gold-incremental.sql")
        utils.run_execute(s)

    setup_schema() >> incremental()


gold()
