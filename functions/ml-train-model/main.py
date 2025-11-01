# this function trains a machine learning model and evaluates it
# accepts algorithm type and hyperparameters via request params

import functions_framework
from google.cloud import secretmanager
from google.cloud import storage
import duckdb
import pandas as pd
import json
import joblib
import io
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Helper function to read SQL files
def read_sql(filename: str) -> str:
    """Read a SQL file from the sql/ directory."""
    sql_dir = Path(__file__).parent / "sql"
    sql_path = sql_dir / filename
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")
    return sql_path.read_text(encoding="utf-8")

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
    
    # Load SQL queries from files
    train_query = read_sql("load-train-data.sql")
    test_query = read_sql("load-test-data.sql")
    
    print("Loading training data...")
    train_df = md.sql(train_query).df()
    print(f"Loaded {len(train_df)} training samples")
    
    print("Loading test data...")
    test_df = md.sql(test_query).df()
    print(f"Loaded {len(test_df)} test samples")
    
    if len(train_df) == 0:
        return {"error": "No training data found"}, 400
    if len(test_df) == 0:
        return {"error": "No test data found"}, 400
    
    # Extract features (all avg_* columns + is_home)
    feature_cols = [col for col in train_df.columns if col.startswith('avg_') or col == 'is_home']
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target_fantasy_ppr'].fillna(0)
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['target_fantasy_ppr'].fillna(0)
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train model based on algorithm
    print(f"Training {algorithm} model with hyperparameters: {hyperparams}")
    
    if algorithm == "linear_regression":
        model = LinearRegression(**hyperparams)
    elif algorithm == "random_forest":
        model = RandomForestRegressor(**hyperparams, random_state=42)
    elif algorithm == "gradient_boosting":
        model = GradientBoostingRegressor(**hyperparams, random_state=42)
    else:
        return {"error": f"Unknown algorithm: {algorithm}"}, 400
    
    print("Fitting model...")
    model.fit(X_train, y_train)
    print("Model training completed")
    
    # Evaluate on test set
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    
    print(f"Test Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Correlation: {correlation:.4f}")
    
    # Serialize model to GCS
    print("Serializing model to GCS...")
    
    # Serialize model to bytes using joblib
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)
    
    # Build GCS path with Hive partitioning
    gcs_path = f"models/model_id={model_id}/dataset_id={dataset_id}/run_id={run_id}/model.pkl"
    
    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_file(model_bytes, content_type='application/octet-stream')
    
    gcs_full_path = f"gs://{bucket_name}/{gcs_path}"
    print(f"Model saved to: {gcs_full_path}")
    
    # Return metrics and GCS path
    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "gcs_path": gcs_full_path,
        "metrics": {
            "test_rmse": round(float(rmse), 4),
            "test_mae": round(float(mae), 4),
            "test_correlation": round(float(correlation), 4),
            "test_count": len(test_df)
        }
    }, 200

