from datetime import datetime
from airflow.sdk import dag, task    
from pathlib import Path
import duckdb
import os
from ba882 import utils
from jinja2 import Template

# paths, as the airflow project is a project we deploy to astronomer
BASE_DIR = Path(os.environ.get("AIRFLOW_HOME", "/usr/local/airflow"))
SQL_DIR = BASE_DIR / "include" / "sql"

# static values that are why this dag/workflow exists
model_vals = {
  "model_id": "model-fantasy-predictor",
  "name": "Player Fantasy Points Predictor",
  "business_problem": "Predict weekly fantasy football points for player lineup optimization",
  "ticket_number": "BA882-25",
  "owner": "analytics_team"
}

@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "model"]
)
def player_points_prediction():

    @task
    def register_model():
        s = utils.read_sql(SQL_DIR / "mlops-model-registry.sql")
        template = Template(s)
        sql = template.render(**model_vals)  # Unpack dictionary values kwargs into template
        print(sql)
        utils.run_sql(sql)

    @task
    def create_dataset():
        sql = utils.read_sql(SQL_DIR / "ai_datasets" / "player-fantasy-points.sql")
        utils.run_sql(sql)
        
    register_model() >> create_dataset()

player_points_prediction()