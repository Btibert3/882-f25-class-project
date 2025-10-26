from pathlib import Path
import duckdb
import os

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
        x = md.sql(SQL).fetchall()
    finally:
        md.close()
    return x