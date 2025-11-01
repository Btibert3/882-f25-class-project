# this function serves predictions from deployed ML models
# supports both SQL-based models and Python/sklearn models
# determines model type by checking artifact_path format

import functions_framework
from google.cloud import secretmanager
from google.cloud import storage
import duckdb
import pandas as pd
import json
import joblib
import io

# settings
project_id = 'btibert-ba882-fall25'
secret_id = 'MotherDuck'
version_id = 'latest'
bucket_name = 'btibert-ba882-fall25-nfl'

@functions_framework.http
def task(request):
    """
    Serve predictions from the currently deployed model.
    
    Expected request params:
    - athlete_id: (optional) filter predictions by athlete
    - season: (optional) filter by season
    - week: (optional) filter by week
    - game_id: (optional) filter by game_id
    
    Returns predictions based on the deployed model type:
    - SQL models: execute SQL query from artifact_path
    - Python models: load .pkl from GCS and run predict()
    """
    
    # Get optional filter parameters
    athlete_id = request.args.get("athlete_id")
    season = request.args.get("season")
    week = request.args.get("week")
    game_id = request.args.get("game_id")
    
    # Connect to Motherduck
    sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = sm.access_secret_version(request={"name": name})
    md_token = response.payload.data.decode("UTF-8")
    md = duckdb.connect(f'md:?motherduck_token={md_token}')
    
    # Look up current deployment (only active deployments with traffic_split > 0)
    deployment_query = """
        SELECT 
            d.model_version_id,
            d.endpoint_url,
            mv.artifact_path
        FROM nfl.mlops.deployment d
        JOIN nfl.mlops.model_version mv ON d.model_version_id = mv.model_version_id
        WHERE d.traffic_split > 0
        ORDER BY d.deployed_at DESC, d.traffic_split DESC
        LIMIT 1
    """
    
    try:
        deployment_result = md.sql(deployment_query).df()
        if len(deployment_result) == 0:
            return {
                "error": "No deployment found",
                "details": "No model has been deployed yet. Run the training pipeline first."
            }, 404
        
        model_version_id = deployment_result.iloc[0]['model_version_id']
        artifact_path = deployment_result.iloc[0]['artifact_path']
        endpoint_url = deployment_result.iloc[0]['endpoint_url']
        
        print(f"Using model version: {model_version_id}")
        print(f"Artifact path: {artifact_path}")
        
    except Exception as e:
        return {
            "error": "Could not lookup deployment",
            "details": str(e)
        }, 500
    
    # Determine model type based on artifact_path format
    if artifact_path.startswith("gs://"):
        # Python/sklearn model - load from GCS
        print("Detected Python model - loading from GCS...")
        try:
            # Parse GCS path: gs://bucket/path/to/model.pkl
            path_parts = artifact_path.replace("gs://", "").split("/", 1)
            if len(path_parts) != 2:
                return {"error": "Invalid GCS path format"}, 400
            
            gcs_bucket = path_parts[0]
            gcs_blob_path = path_parts[1]
            
            # Load model from GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(gcs_bucket)
            blob = bucket.blob(gcs_blob_path)
            
            model_bytes = io.BytesIO()
            blob.download_to_file(model_bytes)
            model_bytes.seek(0)
            
            model = joblib.load(model_bytes)
            print("Model loaded successfully")
            
            # Load features from Motherduck
            # Use the same feature set as training, but also include actuals for comparison
            feature_query = """
                SELECT 
                    athlete_id,
                    game_id,
                    athlete_name,
                    season,
                    week,
                    game_date,
                    target_fantasy_ppr as actual_ppr,
                    target_fantasy_standard as actual_standard,
                    avg_pass_completions_3w,
                    avg_pass_attempts_3w,
                    avg_pass_yards_3w,
                    avg_pass_touchdowns_3w,
                    avg_interceptions_3w,
                    avg_sacks_3w,
                    avg_qb_rating_3w,
                    avg_rush_attempts_3w,
                    avg_rush_yards_3w,
                    avg_rush_touchdowns_3w,
                    avg_receptions_3w,
                    avg_receiving_targets_3w,
                    avg_receiving_yards_3w,
                    avg_receiving_touchdowns_3w,
                    avg_fumbles_3w,
                    avg_fumbles_lost_3w,
                    avg_field_goals_made_3w,
                    avg_field_goal_attempts_3w,
                    avg_extra_points_made_3w,
                    avg_extra_point_attempts_3w,
                    avg_total_tackles_3w,
                    avg_sacks_def_3w,
                    avg_interceptions_def_3w,
                    avg_fumbles_recovered_3w,
                    avg_defensive_touchdowns_3w,
                    is_home
                FROM nfl.ai_datasets.player_fantasy_features
                WHERE target_fantasy_ppr IS NOT NULL
            """
            
            # Add filters if provided
            conditions = []
            if athlete_id:
                conditions.append(f"athlete_id = '{athlete_id}'")
            if season:
                conditions.append(f"season = {season}")
            if week:
                conditions.append(f"week = {week}")
            if game_id:
                conditions.append(f"game_id = {game_id}")
            
            if conditions:
                feature_query += " AND " + " AND ".join(conditions)
            
            # Load features
            feature_df = md.sql(feature_query).df()
            
            if len(feature_df) == 0:
                return {
                    "error": "No data found",
                    "details": "No player data matching the provided filters"
                }, 404
            
            # Prepare features for prediction (same order as training)
            feature_columns = [
                'avg_pass_completions_3w', 'avg_pass_attempts_3w', 'avg_pass_yards_3w',
                'avg_pass_touchdowns_3w', 'avg_interceptions_3w', 'avg_sacks_3w',
                'avg_qb_rating_3w', 'avg_rush_attempts_3w', 'avg_rush_yards_3w',
                'avg_rush_touchdowns_3w', 'avg_receptions_3w', 'avg_receiving_targets_3w',
                'avg_receiving_yards_3w', 'avg_receiving_touchdowns_3w', 'avg_fumbles_3w',
                'avg_fumbles_lost_3w', 'avg_field_goals_made_3w', 'avg_field_goal_attempts_3w',
                'avg_extra_points_made_3w', 'avg_extra_point_attempts_3w', 'avg_total_tackles_3w',
                'avg_sacks_def_3w', 'avg_interceptions_def_3w', 'avg_fumbles_recovered_3w',
                'avg_defensive_touchdowns_3w', 'is_home'
            ]
            
            X = feature_df[feature_columns].fillna(0)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Combine predictions with metadata and actuals
            result_df = feature_df[['athlete_id', 'game_id', 'athlete_name', 'season', 'week', 'game_date']].copy()
            result_df['predicted_ppr'] = predictions
            result_df['actual_ppr'] = feature_df['actual_ppr']
            result_df['actual_standard'] = feature_df['actual_standard']
            
            # Calculate prediction error for easy display
            result_df['error'] = result_df['predicted_ppr'] - result_df['actual_ppr']
            result_df['abs_error'] = result_df['error'].abs()
            
            # Return as JSON
            return {
                "model_version_id": model_version_id,
                "model_type": "python_sklearn",
                "predictions": result_df.to_dict(orient='records'),
                "count": len(result_df)
            }, 200
            
        except Exception as e:
            return {
                "error": "Error loading or using Python model",
                "details": str(e)
            }, 500
    
    else:
        # SQL model - execute SQL query
        print("Detected SQL model - executing query...")
        try:
            # Build SQL query from artifact_path
            # artifact_path format: "nfl.model_outputs.predictions_fantasy WHERE run_id = '...'"
            # We'll extract the table and WHERE clause, then add our filters
            
            # Parse artifact_path: should be like "table_name WHERE conditions"
            if "WHERE" in artifact_path:
                table_part, where_part = artifact_path.split("WHERE", 1)
                table_name = table_part.strip()
                base_where = where_part.strip()
            else:
                table_name = artifact_path.strip()
                base_where = ""
            
            # Build query
            query = f"SELECT * FROM {table_name}"
            
            # Combine base WHERE clause with optional filters
            where_conditions = []
            if base_where:
                where_conditions.append(f"({base_where})")
            if athlete_id:
                # Quote athlete_id as it's a string/VARCHAR
                where_conditions.append(f"athlete_id = '{athlete_id}'")
            if season:
                where_conditions.append(f"season = {season}")
            if week:
                where_conditions.append(f"week = {week}")
            if game_id:
                where_conditions.append(f"game_id = {game_id}")
            
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            
            # Execute query
            print(f"Executing SQL: {query}")
            result_df = md.sql(query).df()
            
            if len(result_df) == 0:
                return {
                    "error": "No predictions found",
                    "details": "No predictions matching the provided filters"
                }, 404
            
            # Calculate prediction error for easy display (if actuals are available)
            if 'actual_ppr' in result_df.columns and 'predicted_ppr' in result_df.columns:
                result_df['error'] = result_df['predicted_ppr'] - result_df['actual_ppr']
                result_df['abs_error'] = result_df['error'].abs()
            
            # Return as JSON
            return {
                "model_version_id": model_version_id,
                "model_type": "sql_weighted_average",
                "predictions": result_df.to_dict(orient='records'),
                "count": len(result_df)
            }, 200
            
        except Exception as e:
            return {
                "error": "Error executing SQL model",
                "details": str(e)
            }, 500

