from datetime import datetime
from airflow.sdk import dag, task    
from pathlib import Path
# import duckdb
import json
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
                COUNT(DISTINCT athlete_id) as unique_players,
                (SELECT COUNT(*) 
                 FROM information_schema.columns 
                 WHERE table_schema = 'ai_datasets' 
                  AND table_name = 'player_fantasy_features'
                  AND column_name LIKE 'avg_%') as feature_count
            FROM nfl.ai_datasets.player_fantasy_features
        """)
        print(result)

        # the utility returns a list of tuples based on what we created, and uses data in our warehouse to form the entry
        metadata = {
            "data_version": result[0][0],  # e.g., "2025_w08"
            "dataset_id": f"ds-player-fantasy-{result[0][0]}",
            "row_count": result[0][1],
            "unique_players": result[0][2],
            "feature_count": result[0][3]
        }
        print(f"Dataset created: {metadata}")
        return metadata  # This gets passed via XCom in Airflow

    @task
    def register_dataset(dataset_metadata):
        s = utils.read_sql(SQL_DIR / "mlops-dataset-registry.sql")
        template = Template(s)
        sql = template.render(
            model_id=model_vals["model_id"],  # Add model_id from model_vals
            **dataset_metadata  # Unpack dataset metadata
        )
        print(sql)
        utils.run_execute(sql)
    
    @task
    def register_output_table():
        SQL = """
        CREATE TABLE IF NOT EXISTS nfl.model_outputs.predictions_fantasy (
            run_id TEXT NOT NULL,
            game_id INTEGER NOT NULL,
            athlete_id TEXT NOT NULL,
            athlete_name TEXT,
            season INTEGER,
            week INTEGER,
            game_date TIMESTAMP,
            split TEXT,
            actual_ppr DECIMAL(18,4),
            predicted_ppr DECIMAL(18,4),
            actual_standard DECIMAL(18,4),
            predicted_standard DECIMAL(18,4),
            created_at TIMESTAMP DEFAULT NOW(),
        
            -- Composite primary key: run + season + game + athlete
            PRIMARY KEY (run_id, season, game_id, athlete_id)
        );
        """
        print(SQL)
        utils.run_execute(SQL)

    @task
    def create_predictions(dataset_metadata, **context):
        # Generate clean run_id
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"run_{dataset_metadata['data_version']}_{timestamp}"
        airflow_run_id = context['dag_run'].run_id
        print(f"Generating predictions for run: {run_id}")

        # Apply SQL model to generate predictions
        sql = utils.read_sql(SQL_DIR / "model_outputs" / "player-fantasy-points-model.sql")
        template = Template(sql)
        prediction_sql = template.render(run_id=run_id)
        utils.run_execute(prediction_sql)

        # metrics_result
        metrics_result = utils.run_sql(f"""
            WITH test_predictions AS (
                SELECT actual_ppr, predicted_ppr, actual_standard, predicted_standard
                FROM nfl.model_outputs.predictions_fantasy
                WHERE run_id = '{run_id}' AND split = 'test' AND actual_ppr IS NOT NULL
            )
            SELECT 
                COUNT(*) as test_count,
                ROUND(SQRT(AVG(POWER(actual_ppr - predicted_ppr, 2))), 2) as rmse_ppr,
                ROUND(AVG(ABS(actual_ppr - predicted_ppr)), 2) as mae_ppr,
                ROUND(CORR(actual_ppr, predicted_ppr), 3) as corr_ppr
            FROM test_predictions
        """)

        # Package metadata for next task
        run_metadata = {
            "run_id": run_id,
            "airflow_run_id": airflow_run_id,
            "dataset_id": dataset_metadata['dataset_id'],
            "test_count": int(metrics_result[0][0]) if metrics_result[0][0] else 0,
            "rmse_ppr": float(metrics_result[0][1]) if metrics_result[0][1] else None,
            "mae_ppr": float(metrics_result[0][2]) if metrics_result[0][2] else None,
            "corr_ppr": float(metrics_result[0][3]) if metrics_result[0][3] else None,
        }
        
        print(f"Test Metrics: RMSE={run_metadata['rmse_ppr']}, MAE={run_metadata['mae_ppr']}, Corr={run_metadata['corr_ppr']}")
        return run_metadata
    
    @task
    def register_training_run(run_metadata):
        """Register the training run in mlops.training_run"""
        
        # Build params JSON (your SQL model configuration)
        params = {
            "model_type": "sql_weighted_average",
            "lookback_window": 3,
            "scoring_system": "ppr_and_standard",
            "scoring_rules": {
                "pass_yard_points": 0.04,
                "pass_td_points": 4,
                "rush_yard_points": 0.1,
                "rush_td_points": 6,
                "reception_points_ppr": 1,
                "receiving_yard_points": 0.1,
                "receiving_td_points": 6,
                "interception_penalty": -2,
                "fumble_lost_penalty": -2,
                "field_goal_points": 3,
                "extra_point_points": 1
            }
        }
        
        # Build metrics JSON
        metrics = {
            "ppr": {
                "test_rmse": run_metadata['rmse_ppr'],
                "test_mae": run_metadata['mae_ppr'],
                "test_correlation": run_metadata['corr_ppr'],
                "test_count": run_metadata['test_count']
            }
        }
        
        # Insert into mlops.training_run
        sql = f"""
        INSERT INTO nfl.mlops.training_run (
            run_id,
            model_id,
            dataset_id,
            params,
            metrics,
            artifact,
            status,
            created_at
        )
        VALUES (
            '{run_metadata['run_id']}',
            '{model_vals['model_id']}',
            '{run_metadata['dataset_id']}',
            '{json.dumps(params)}',
            '{json.dumps(metrics)}',
            'nfl.model_outputs.predictions_fantasy WHERE run_id = ''{run_metadata['run_id']}''',
            'completed',
            NOW()
        )
        """
        
        print(sql)
        utils.run_execute(sql)
        print(f"Training run registered: {run_metadata['run_id']}")

    # Wire up dependencies
    dataset_meta = create_dataset()
    run_meta = create_predictions(dataset_meta)
    
    register_model() >> dataset_meta >> register_dataset(dataset_meta) >> register_output_table() >> run_meta >> register_training_run(run_meta)


player_points_prediction()