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
        utils.run_execute(sql)

    @task
    def create_dataset():
        sql = utils.read_sql(SQL_DIR / "ai_datasets" / "player-fantasy-points.sql")
        utils.run_execute(sql)

        # Query to get metadata about what we just created
        result = utils.run_sql("""
            SELECT 
                '2025_w' || LPAD(MAX(week)::VARCHAR, 2, '0') as data_version,
                COUNT(*) as row_count,
                COUNT(DISTINCT athlete_id) as unique_players
            FROM nfl.ai_datasets.player_fantasy_features
        """)
        print(result)

        # the utility returns a list of tuples based on what we created, and uses data in our warehouse to form the entry
        metadata = {
            "data_version": result[0][0],  # e.g., "2025_w08"
            "dataset_id": f"ds-player-fantasy-{result[0][0]}",
            "row_count": result[0][1],
            "unique_players": result[0][2]
        }
        print(f"Dataset created: {metadata}")
        return metadata  # This gets passed via XCom in Airflow

        
    register_model() >> create_dataset()

player_points_prediction()