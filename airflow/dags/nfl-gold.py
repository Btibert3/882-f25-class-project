from datetime import datetime
from airflow.sdk import dag, task    
from pathlib import Path
import duckdb
import os


# paths, as the airflow project is a project we deploy to astronomer
BASE_DIR = Path(os.environ.get("AIRFLOW_HOME", "/usr/local/airflow"))
SQL_DIR = BASE_DIR / "include" / "sql"


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def read_sql(path: Path) -> str:
    """Read a .sql file and return its contents."""
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


def run_execute(SQL: str):
    """Execute multiple SQL statements in a single transaction."""
    md = duckdb.connect(f"md:nfl?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}")
    print("starting transaction ...")
    try:
        md.execute("BEGIN")
        for stmt in [s for s in SQL.split(";") if s.strip()]:
            print("running statement --------------------")
            print(stmt)
            md.execute(stmt)
        md.execute("COMMIT")
    except Exception as e:
        print(f"Error: {e}")
        try:
            md.execute("ROLLBACK")
        except Exception:
            print("Rollback failed")
        raise
    finally:
        md.close()


def run_sql(SQL: str):
    """Execute a single SQL script."""
    md = duckdb.connect(f"md:nfl?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}")
    print("running SQL ...")
    try:
        md.sql(SQL)
    finally:
        md.close()


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
        s = read_sql(SQL_DIR / "gold-schema-setup.sql")
        run_sql(s)

    @task
    def incremental():
        s = read_sql(SQL_DIR / "gold-incremental.sql")
        run_execute(s)

    setup_schema() >> incremental()


gold()
