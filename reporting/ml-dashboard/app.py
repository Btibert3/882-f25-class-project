# MLOps Fantasy Points Dashboard
# This app demonstrates how to:
# 1. Query ML model registry and metrics from MotherDuck
# 2. Lookup players and their historical performance
# 3. Interact with deployed ML models via Cloud Function endpoint
# 4. Compare predictions vs actuals to understand model performance

import os
import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import secretmanager
import requests
import json
from datetime import datetime

# ------------------------------------------------------------------------------
# Configuration & Connection Setup
# ------------------------------------------------------------------------------
project_id = 'btibert-ba882-fall25'
secret_id = 'MotherDuck'
version_id = 'latest'
prediction_endpoint = 'https://us-central1-btibert-ba882-fall25.cloudfunctions.net/ml-predict-fantasy'

# Default player
DEFAULT_PLAYER_ID = '4431452'  # Drake Maye

# Page config
st.set_page_config(
    page_title="MLOps Fantasy Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
@st.cache_resource
def get_motherduck_connection():
    """Establish and cache MotherDuck connection"""
    sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = sm.access_secret_version(request={"name": name})
    md_token = response.payload.data.decode("UTF-8")
    return duckdb.connect(f'md:?motherduck_token={md_token}')

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_df(query: str, params: tuple | None = None) -> pd.DataFrame:
    """Execute SQL query and return DataFrame"""
    md = get_motherduck_connection()
    if params:
        return md.execute(query, params).df()
    return md.sql(query).df()

def call_prediction_endpoint(params: dict) -> dict:
    """
    Call the ML prediction Cloud Function endpoint.
    This demonstrates how to interact with deployed models.
    """
    try:
        response = requests.get(prediction_endpoint, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "details": "Failed to call prediction endpoint"}

# Initialize connection
md = get_motherduck_connection()

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.title("🏈 MLOps Fantasy Points Dashboard")
st.caption("""
    **Learn how MLOps components work together:**
    - View model metrics and training runs from the ML model registry
    - Lookup players and their historical performance
    - **Interact with deployed models via Cloud Function API** 
""")

# ------------------------------------------------------------------------------
# Main Tabs
# ------------------------------------------------------------------------------
tab_metrics, tab_player, tab_predictions = st.tabs([
    "📊 Metrics Dashboard",
    "👤 Player Lookup",
    "🔮 Live Predictions"
])

# ============================================================================
# TAB 1: Metrics Dashboard
# ============================================================================
with tab_metrics:
    st.header("MLOps Model Registry & Metrics")
    st.markdown("""
    This section queries the **ML model registry** (`nfl.mlops` schema) to show:
    - Registered models and their versions
    - Training run metrics (RMSE, MAE, Correlation)
    - Deployment status
    """)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_count = fetch_df("""
            SELECT COUNT(DISTINCT model_id) as n
            FROM nfl.mlops.model
        """)
        count = model_count["n"].iat[0] if not model_count.empty else 0
        st.metric("Total Models", count)
    
    with col2:
        version_count = fetch_df("""
            SELECT COUNT(*) as n
            FROM nfl.mlops.model_version
            WHERE status = 'approved'
        """)
        count = version_count["n"].iat[0] if not version_count.empty else 0
        st.metric("Approved Versions", count)
    
    with col3:
        deployment_count = fetch_df("""
            SELECT COUNT(*) as n
            FROM nfl.mlops.deployment
            WHERE traffic_split > 0
        """)
        count = deployment_count["n"].iat[0] if not deployment_count.empty else 0
        st.metric("Active Deployments", count)
    
    with col4:
        run_count = fetch_df("""
            SELECT COUNT(*) as n
            FROM nfl.mlops.training_run
            WHERE status = 'completed'
        """)
        count = run_count["n"].iat[0] if not run_count.empty else 0
        st.metric("Completed Runs", count)
    
    st.divider()
    
    # Model Versions Table
    st.subheader("📋 Model Versions Registry")
    
    versions_df = fetch_df("""
        SELECT 
            mv.model_version_id,
            mv.model_id,
            m.name as model_name,
            mv.status,
            CAST(JSON_EXTRACT(mv.metrics_json, '$.ppr.test_rmse') AS DECIMAL(10,4)) as test_rmse,
            CAST(JSON_EXTRACT(mv.metrics_json, '$.ppr.test_mae') AS DECIMAL(10,4)) as test_mae,
            CAST(JSON_EXTRACT(mv.metrics_json, '$.ppr.test_correlation') AS DECIMAL(10,4)) as test_corr,
            CAST(JSON_EXTRACT(mv.metrics_json, '$.ppr.test_count') AS INTEGER) as test_count,
            mv.created_at
        FROM nfl.mlops.model_version mv
        JOIN nfl.mlops.model m ON mv.model_id = m.model_id
        ORDER BY mv.created_at DESC
    """)
    
    if not versions_df.empty:
        st.dataframe(versions_df, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ No model versions found. Run the training pipeline to register models.")
    
    st.divider()
    
    # Training Runs Table
    st.subheader("🏃 Recent Training Runs")
    st.markdown("""
    Each training run represents one model experiment. The registry tracks:
    - Algorithm type (linear_regression, random_forest, sql_weighted_average, etc.)
    - Hyperparameters used
    - Evaluation metrics on test set
    """)
    
    runs_df = fetch_df("""
        SELECT 
            tr.run_id,
            tr.model_id,
            tr.dataset_id,
            CAST(JSON_EXTRACT(tr.metrics, '$.ppr.test_rmse') AS DECIMAL(10,4)) as test_rmse,
            CAST(JSON_EXTRACT(tr.metrics, '$.ppr.test_mae') AS DECIMAL(10,4)) as test_mae,
            CAST(JSON_EXTRACT(tr.metrics, '$.ppr.test_correlation') AS DECIMAL(10,4)) as test_corr,
            CAST(JSON_EXTRACT(tr.params, '$.algorithm') AS VARCHAR) as algorithm,
            CAST(JSON_EXTRACT(tr.params, '$.model_type') AS VARCHAR) as model_type,
            tr.status,
            tr.created_at
        FROM nfl.mlops.training_run tr
        ORDER BY tr.created_at DESC
        LIMIT 30
    """)
    
    if not runs_df.empty:
        st.dataframe(runs_df, use_container_width=True, hide_index=True)
        
        # Algorithm comparison
        if runs_df['algorithm'].notna().any() or runs_df['model_type'].notna().any():
            st.subheader("🔬 Algorithm Performance Comparison")
            
            # Combine algorithm and model_type for grouping
            algo_df = runs_df.copy()
            algo_df['algorithm_type'] = algo_df.apply(
                lambda row: row['algorithm'] if pd.notna(row['algorithm']) else row['model_type'],
                axis=1
            )
            
            algo_agg = algo_df.groupby('algorithm_type').agg({
                'test_mae': 'mean',
                'test_rmse': 'mean',
                'test_corr': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=algo_agg['algorithm_type'],
                y=algo_agg['test_mae'],
                name='Mean MAE',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title="Average Test MAE by Algorithm/Model Type",
                xaxis_title="Algorithm",
                yaxis_title="Mean Absolute Error",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ No training runs found. Models need to be trained first.")

# ============================================================================
# TAB 2: Player Lookup
# ============================================================================
with tab_player:
    st.header("Player Lookup & Historical Analysis")
    st.markdown("""
    This section demonstrates:
    1. Querying player historical data from the **gold schema** (all games)
    2. Visualizing performance trends by last 4 games
    3. Viewing 3-week moving average features used for predictions
    """)
    
    # Get list of players
    players_df = fetch_df("""
        SELECT DISTINCT 
            athlete_id,
            athlete_name
        FROM nfl.gold.player_game_stats
        WHERE athlete_name IS NOT NULL
        ORDER BY athlete_name
    """)
    
    if players_df.empty:
        st.warning("No players found in dataset.")
    else:
        # Player selection with default to Drake Maye
        player_options = dict(zip(
            players_df['athlete_name'] + " (" + players_df['athlete_id'] + ")",
            players_df['athlete_id']
        ))
        
        # Find Drake Maye index
        default_index = 0
        if DEFAULT_PLAYER_ID in players_df['athlete_id'].values:
            drake_options = [(name, aid) for name, aid in player_options.items() if aid == DEFAULT_PLAYER_ID]
            if drake_options:
                default_index = list(player_options.keys()).index(drake_options[0][0])
        
        selected_player_display = st.selectbox(
            "Select Player",
            options=list(player_options.keys()),
            index=default_index
        )
        
        selected_player_id = player_options[selected_player_display]
        player_name = players_df[players_df['athlete_id'] == selected_player_id]['athlete_name'].iloc[0]
        
        st.subheader(f"📊 Analysis for {player_name}")
        
        # Historical performance from GOLD SCHEMA (all games, including gaps)
        hist_df = fetch_df(f"""
            SELECT 
                game_id,
                season,
                week,
                game_date,
                fantasy_points_ppr as actual_ppr,
                fantasy_points_standard as actual_standard
            FROM nfl.gold.player_game_stats
            WHERE athlete_id = ?
              AND fantasy_points_ppr IS NOT NULL
            ORDER BY game_date
        """, (selected_player_id,))
        
        if not hist_df.empty:
            # Get last 4 games
            last_4_df = hist_df.tail(4).copy().reset_index(drop=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_ppr = hist_df['actual_ppr'].mean()
                st.metric("Avg PPR Points", f"{avg_ppr:.2f}")
            
            with col2:
                std_ppr = hist_df['actual_ppr'].std()
                st.metric("Std Deviation", f"{std_ppr:.2f}")
            
            with col3:
                max_ppr = hist_df['actual_ppr'].max()
                st.metric("Best Game", f"{max_ppr:.2f}")
            
            with col4:
                total_games = len(hist_df)
                st.metric("Total Games", total_games)
            
            st.divider()
            
            # Historical performance chart - Last 4 Games with categorical x-axis
            st.subheader("📈 Last 4 Games Fantasy Points")
            
            # Create categorical labels for x-axis
            last_4_df['game_label'] = last_4_df.apply(
                lambda row: f"Week {row['week']}\n{row['game_date'].strftime('%m/%d') if pd.notna(row['game_date']) else ''}",
                axis=1
            )
            
            fig = px.bar(
                last_4_df,
                x='game_label',
                y='actual_ppr',
                title=f"{player_name} - Fantasy Points (Last 4 Games)",
                labels={'actual_ppr': 'PPR Points', 'game_label': 'Game'},
                color='actual_ppr',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # 3-Week Moving Average Features Section
            st.subheader("📊 3-Week Moving Average Features")
            st.markdown("""
            **The model uses 3-week moving averages as features.** Below shows the most recent features
            for this player, which would be used to predict the NEXT game:
            """)
            
            # Get latest features from ai_datasets for this player
            features_df = fetch_df(f"""
                SELECT 
                    season,
                    week,
                    game_date,
                    avg_receptions_3w,
                    avg_receiving_yards_3w,
                    avg_receiving_touchdowns_3w,
                    avg_rush_attempts_3w,
                    avg_rush_yards_3w,
                    avg_rush_touchdowns_3w,
                    avg_pass_yards_3w,
                    avg_pass_touchdowns_3w,
                    avg_interceptions_3w,
                    is_home
                FROM nfl.ai_datasets.player_fantasy_features
                WHERE athlete_id = ?
                ORDER BY game_date DESC
                LIMIT 10
            """, (selected_player_id,))
            
            if not features_df.empty:
                # Show latest features prominently
                st.markdown("**Most Recent Features Used for Predictions:**")
                latest_features = features_df.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("3-Week Avg Receptions", f"{latest_features.get('avg_receptions_3w', 0):.2f}")
                    st.metric("3-Week Avg Rec Yards", f"{latest_features.get('avg_receiving_yards_3w', 0):.2f}")
                    st.metric("3-Week Avg Rec TDs", f"{latest_features.get('avg_receiving_touchdowns_3w', 0):.2f}")
                
                with col2:
                    st.metric("3-Week Avg Rush Attempts", f"{latest_features.get('avg_rush_attempts_3w', 0):.2f}")
                    st.metric("3-Week Avg Rush Yards", f"{latest_features.get('avg_rush_yards_3w', 0):.2f}")
                    st.metric("3-Week Avg Rush TDs", f"{latest_features.get('avg_rush_touchdowns_3w', 0):.2f}")
                
                with col3:
                    st.metric("3-Week Avg Pass Yards", f"{latest_features.get('avg_pass_yards_3w', 0):.2f}")
                    st.metric("3-Week Avg Pass TDs", f"{latest_features.get('avg_pass_touchdowns_3w', 0):.2f}")
                    st.metric("3-Week Avg INTs", f"{latest_features.get('avg_interceptions_3w', 0):.2f}")
                
            else:
                st.info("No feature data available for this player yet (need at least 3 games played).")
        else:
            st.info(f"No historical data found for {player_name}")

# ============================================================================
# TAB 3: Live Predictions
# ============================================================================
with tab_predictions:
    st.header("🔮 Generate Live Predictions")
    st.markdown("""
    **This section demonstrates how to use the deployed ML model:**
    - Select a player
    - View their features and get a prediction for the NEXT game
    - The dataset shows week 8, but the prediction is for week 9
    """)
    
    # Get list of players
    players_df = fetch_df("""
        SELECT DISTINCT 
            athlete_id,
            athlete_name
        FROM nfl.ai_datasets.player_fantasy_features
        WHERE athlete_name IS NOT NULL
        ORDER BY athlete_name
        LIMIT 500
    """)
    
    if players_df.empty:
        st.warning("No players found in dataset.")
    else:
        # Player selection with default to Drake Maye
        player_options = dict(zip(
            players_df['athlete_name'] + " (" + players_df['athlete_id'] + ")",
            players_df['athlete_id']
        ))
        
        # default to drake maye
        default_index = 0
        if DEFAULT_PLAYER_ID in players_df['athlete_id'].values:
            drake_options = [(name, aid) for name, aid in player_options.items() if aid == DEFAULT_PLAYER_ID]
            if drake_options:
                default_index = list(player_options.keys()).index(drake_options[0][0])
        
        selected_player_display = st.selectbox(
            "Select Player",
            options=list(player_options.keys()),
            index=default_index
        )
        
        selected_player_id = player_options[selected_player_display]
        player_name = players_df[players_df['athlete_id'] == selected_player_id]['athlete_name'].iloc[0]
        
        if st.button("🚀 Get Prediction", type="primary", use_container_width=True):
            # Get the player's latest features from ai_datasets
            with st.spinner("Fetching player features and generating prediction..."):
                # Get latest features
                features_df = fetch_df(f"""
                    SELECT 
                        season,
                        week,
                        game_date,
                        is_home,
                        avg_receptions_3w,
                        avg_receiving_yards_3w,
                        avg_receiving_touchdowns_3w,
                        avg_rush_attempts_3w,
                        avg_rush_yards_3w,
                        avg_rush_touchdowns_3w,
                        avg_pass_yards_3w,
                        avg_pass_touchdowns_3w,
                        avg_interceptions_3w,
                        avg_receiving_targets_3w,
                        avg_rush_attempts_3w
                    FROM nfl.ai_datasets.player_fantasy_features
                    WHERE athlete_id = ?
                    ORDER BY game_date DESC
                    LIMIT 1
                """, (selected_player_id,))
                
                if not features_df.empty:
                    latest_features = features_df.iloc[0]
                    
                    # Display the features being used
                    st.success(f"✅ Found data for {player_name}")
                    
                    # Show prediction context
                    st.subheader("📋 Prediction Context")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Last Known Week", f"Week {latest_features['week']}")
                        st.metric("Prediction For", f"Week {int(latest_features['week']) + 1}")
                    with col2:
                        st.metric("Season", latest_features['season'])
                    with col3:
                        st.metric("Home Game?", "Yes" if latest_features.get('is_home', False) else "No")
                    
                    # Show features
                    st.subheader("📊 Features Used (Last 3 Games)")
                    feature_cols = [
                        'avg_receptions_3w', 'avg_receiving_yards_3w', 'avg_receiving_touchdowns_3w',
                        'avg_rush_attempts_3w', 'avg_rush_yards_3w', 'avg_rush_touchdowns_3w',
                        'avg_pass_yards_3w', 'avg_pass_touchdowns_3w', 'avg_interceptions_3w'
                    ]
                    
                    features_display = latest_features[feature_cols]
                    st.dataframe(features_display.T, use_container_width=True)
                    
                    # Now call the prediction endpoint
                    pred_response = call_prediction_endpoint({"athlete_id": selected_player_id})
                    
                    if "error" in pred_response:
                        st.error(f"❌ Error: {pred_response.get('details', pred_response.get('error'))}")
                        if "404" in str(pred_response.get('error', '')):
                            st.info("💡 Tip: Make sure a model has been deployed. Check the Metrics Dashboard tab.")
                    elif "predictions" in pred_response:
                        pred_df = pd.DataFrame(pred_response['predictions'])
                        
                        st.subheader("🔮 Prediction Results")
                        
                    
                        if not pred_df.empty:
                            # Get the most recent prediction
                            latest_pred = pred_df.iloc[-1]

                            
                            with col1:
                                st.metric(
                                    "Predicted PPR Points",
                                    f"{latest_pred['predicted_ppr']:.2f}"
                                )
                        
                            # Model info
                            with st.expander("📋 Model Information", expanded=True):
                                st.json({
                                    "model_version_id": pred_response.get('model_version_id', 'N/A'),
                                    "model_type": pred_response.get('model_type', 'N/A'),
                                    "endpoint": prediction_endpoint
                                })
                            
                            
                        else:
                            st.info("No predictions returned.")
                else:
                    st.error(f"No feature data found for {player_name} in the AI datasets.")
        else:
            st.info("Click the button above to generate a prediction for the selected player.")

# ------------------------------------------------------------------------------
# Footer with teaching notes
# ------------------------------------------------------------------------------
with st.expander("📚 Learning Objectives & Architecture Notes"):
    st.markdown("""
    ### This Dashboard Demonstrates:
    
    1. **MLOps Model Registry**: 
       - Models, versions, and training runs are tracked in `nfl.mlops` schema
       - Each training run records metrics, hyperparameters, and artifacts
       - Model versions can be approved, candidate, or archived
    
    2. **Deployment Management**:
       - Active deployments are tracked in `nfl.mlops.deployment`
       - Traffic splitting allows A/B testing
       - The prediction endpoint automatically uses the active deployment
    
    3. **Model Serving via Cloud Functions**:
       - Models are deployed as serverless HTTP endpoints
       - Supports both SQL-based and Python/sklearn models
       - Same endpoint can serve multiple model types
    
    4. **End-to-End Integration**:
       - Registry → Training → Deployment → Serving
       - Dashboards query registry for metrics
       - Applications call endpoints for predictions
       - Everything connected via shared data warehouse (MotherDuck)
    
    5. **Data Schema Understanding**:
       - **Gold Schema** (`nfl.gold.player_game_stats`): Raw historical data per game
       - **AI Datasets** (`nfl.ai_datasets.player_fantasy_features`): Engineered features for ML
       - Features use 3-week moving averages to predict the next game
    
    6. **Prediction Context**:
       - Dataset shows week 8 = features calculated THROUGH week 8
       - Prediction is FOR week 9 (next game)
       - Model uses last 3 games to predict the 4th
    
    ### Architecture Flow:
    ```
    Training Pipeline (Airflow)
        ↓
    Model Registry (MotherDuck mlops schema)
        ↓
    Model Version & Deployment Registration
        ↓
    Cloud Function Deployment (GCP)
        ↓
    Dashboard/Applications (Streamlit, APIs, etc.)
    ```
    
    ### Data Flow for Predictions:
    ```
    Gold Schema (raw stats) 
        ↓ [3-week moving averages]
    AI Datasets (features)
        ↓ [ML model]
    Predictions (next game)
    ```
    
    ### Key Files in This Project:
    - `airflow/dags/ml-player-fantasy-points.py`: Training pipeline
    - `functions/ml-predict-fantasy/main.py`: Prediction endpoint
    - `airflow/include/sql/mlops-schema-setup.sql`: Registry schema
    - `airflow/include/sql/ai_datasets/player-fantasy-points.sql`: Feature engineering
    """)
