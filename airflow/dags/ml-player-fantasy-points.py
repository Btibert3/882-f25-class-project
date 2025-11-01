from datetime import datetime
from airflow.sdk import dag, task    
from pathlib import Path
from airflow.operators.python import get_current_context
# import duckdb
import json
import os
import requests
from ba882 import utils
from jinja2 import Template

# paths, as the airflow project is a project we deploy to astronomer
BASE_DIR = Path(os.environ.get("AIRFLOW_HOME", "/usr/local/airflow"))
SQL_DIR = BASE_DIR / "include" / "sql"

# helper
def invoke_function(url, params={}) -> dict:
    """
    Invoke our cloud function url and optionally pass data for the function to use
    """
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# static values that are why this dag/workflow exists
model_vals = {
  "model_id": "model-fantasy-predictor",
  "name": "Player Fantasy Points Predictor",
  "business_problem": "Predict weekly fantasy football points for player lineup optimization",
  "ticket_number": "BA882-25",
  "owner": "analytics_team"
}

# model configurations for parallel training
model_configs = [
    {
        "algorithm": "linear_regression",
        "hyperparameters": {}
    },
    {
        "algorithm": "random_forest",
        "hyperparameters": {"n_estimators": 100, "max_depth": 10}
    },
    {
        "algorithm": "random_forest",
        "hyperparameters": {"n_estimators": 200, "max_depth": 15}
    },
    {
        "algorithm": "gradient_boosting",
        "hyperparameters": {"n_estimators": 100, "learning_rate": 0.1}
    }
]

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
    def train_python_model(model_config: dict, dataset_metadata: dict, **context):
        """Invoke Cloud Function to train/evaluate a Python model"""
        # Generate unique run_id for this model config
        ctx = get_current_context()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        algo_short = model_config["algorithm"].replace("_", "")[:8]  # e.g., "linearre", "randomfo"
        # Include map_index to ensure uniqueness across parallel tasks
        map_index = ctx.get("ti", {}).get("map_index", 0)
        run_id = f"run_{dataset_metadata['data_version']}_{algo_short}_{map_index}_{timestamp}"
        
        url = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/ml-train-model"
        params = {
            "algorithm": model_config["algorithm"],
            "hyperparameters": json.dumps(model_config["hyperparameters"]),
            "run_id": run_id,
            "dataset_id": dataset_metadata["dataset_id"],
            "model_id": model_vals["model_id"]
        }
        print(f"Training {model_config['algorithm']} model with run_id: {run_id}")
        result = invoke_function(url, params=params)
        # Add hyperparameters to result for registration task
        result["hyperparameters"] = model_config["hyperparameters"]
        print(f"Training completed: {result}")
        return result

    @task
    def register_training_run_python(model_result: dict, dataset_metadata: dict, **context):
        """Register the Python model training run in mlops.training_run"""
        
        # Build params JSON from model result and config
        params = {
            "algorithm": model_result["algorithm"],
            "hyperparameters": model_result.get("hyperparameters", {}),
            "model_type": "python_sklearn"
        }
        
        # Build metrics JSON from Cloud Function response
        metrics = {
            "ppr": {
                "test_rmse": model_result["metrics"]["test_rmse"],
                "test_mae": model_result["metrics"]["test_mae"],
                "test_correlation": model_result["metrics"]["test_correlation"],
                "test_count": model_result["metrics"]["test_count"]
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
            '{model_result['run_id']}',
            '{model_vals['model_id']}',
            '{dataset_metadata['dataset_id']}',
            '{json.dumps(params)}',
            '{json.dumps(metrics)}',
            '{model_result['gcs_path']}',
            'completed',
            NOW()
        )
        """
        
        print(sql)
        utils.run_execute(sql)
        print(f"Training run registered: {model_result['run_id']}")

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

    @task
    def find_best_model(dataset_metadata):
        """Find the best model by comparing all training runs for this dataset, using MAE as the metric"""
        
        # Query all training runs for this dataset/model and extract MAE from metrics JSON
        sql = f"""
        SELECT 
            run_id,
            params,
            metrics,
            artifact
        FROM nfl.mlops.training_run
        WHERE model_id = '{model_vals['model_id']}'
          AND dataset_id = '{dataset_metadata['dataset_id']}'
          AND status = 'completed'
        ORDER BY CAST(JSON_EXTRACT(metrics, '$.ppr.test_mae') AS DECIMAL(10,4)) ASC
        LIMIT 1
        """
        
        result = utils.run_sql(sql)
        
        if not result or len(result) == 0:
            return {"error": "No completed training runs found"}
        
        best_run = result[0]
        run_id = best_run[0]
        params_json = best_run[1]
        metrics_json = best_run[2]
        artifact = best_run[3]
        
        # Parse metrics to get MAE
        metrics = json.loads(metrics_json)
        best_mae = metrics["ppr"]["test_mae"]
        
        print(f"Best model found: run_id={run_id}, MAE={best_mae}, artifact={artifact}")
        
        return {
            "run_id": run_id,
            "dataset_id": dataset_metadata['dataset_id'],
            "artifact": artifact,
            "mae": best_mae,
            "metrics": metrics
        }

    # To me, this is more pythonic, but it's not the only way to wire up our workflow
    model_task = register_model()
    dataset_task = create_dataset()
    register_dataset_task = register_dataset(dataset_task)
    table_task = register_output_table()
    
    # SQL baseline path (existing)
    prediction_task = create_predictions(dataset_task)
    training_run_task = register_training_run(prediction_task)
    
    # Python models path (parallel training)
    python_training_results = train_python_model.partial(
        dataset_metadata=dataset_task
    ).expand(
        model_config=model_configs
    )
    python_registration_tasks = register_training_run_python.partial(
        dataset_metadata=dataset_task
    ).expand(
        model_result=python_training_results
    )

    # Convergence: find best model after all training runs complete
    find_best = find_best_model(dataset_task)
    
    # Flow
    model_task >> dataset_task >> register_dataset_task >> table_task
    
    # Parallel paths after setup
    table_task >> prediction_task >> training_run_task
    table_task >> python_training_results >> python_registration_tasks
    
    # Convergence: find best after all runs complete
    [training_run_task, python_registration_tasks] >> find_best


player_points_prediction()