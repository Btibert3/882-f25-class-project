# this function trains a machine learning model and evaluates it
# accepts algorithm type and hyperparameters via request params

import functions_framework
from google.cloud import secretmanager
from google.cloud import storage
import duckdb
import pandas as pd
import json
import joblib
from datetime import datetime

# settings
project_id = 'btibert-ba882-fall25'
secret_id = 'MotherDuck'
version_id = 'latest'
bucket_name = 'btibert-ba882-fall25-nfl'

@functions_framework.http
def task(request):
    """
    Train a model based on algorithm and hyperparameters provided.
    
    Expected request params:
    - algorithm: "linear_regression" | "random_forest" | "gradient_boosting"
    - hyperparameters: JSON string of hyperparameters dict
    - run_id: training run identifier from Airflow
    - dataset_id: dataset identifier
    - model_id: model identifier
    """
    
    # Get parameters from request
    algorithm = request.args.get("algorithm")
    hyperparams_json = request.args.get("hyperparameters", "{}")
    run_id = request.args.get("run_id")
    dataset_id = request.args.get("dataset_id")
    model_id = request.args.get("model_id")
    
    if not all([algorithm, run_id, dataset_id, model_id]):
        return {"error": "Missing required parameters: algorithm, run_id, dataset_id, model_id"}, 400
    
    print(f"Training model: algorithm={algorithm}, run_id={run_id}, dataset_id={dataset_id}")
    
    # Parse hyperparameters
    try:
        hyperparams = json.loads(hyperparams_json)
    except json.JSONDecodeError:
        return {"error": "Invalid hyperparameters JSON"}, 400
    
    # Connect to Motherduck
    sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = sm.access_secret_version(request={"name": name})
    md_token = response.payload.data.decode("UTF-8")
    md = duckdb.connect(f'md:?motherduck_token={md_token}')
    
    # TODO: Load training data from nfl.ai_datasets.player_fantasy_features
    # TODO: Extract features (avg_* columns) and target (target_fantasy_ppr)
    # TODO: Train model based on algorithm
    # TODO: Evaluate on test set
    # TODO: Serialize model to GCS
    # TODO: Return metrics and GCS path
    
    # Placeholder return
    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "gcs_path": f"gs://{bucket_name}/models/model_id={model_id}/dataset_id={dataset_id}/run_id={run_id}/model.pkl",
        "metrics": {
            "test_rmse": None,
            "test_mae": None,
            "test_correlation": None,
            "test_count": 0
        }
    }, 200

