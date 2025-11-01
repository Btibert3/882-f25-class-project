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

# Page config
st.set_page_config(
    page_title="MLOps Fantasy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
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
    - Compare predictions vs actuals to evaluate model performance
""")

# ------------------------------------------------------------------------------
# Sidebar Filters
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Filters & Settings")
    
    # Season filter
    seasons_df = fetch_df("""
        SELECT DISTINCT season 
        FROM nfl.ai_datasets.player_fantasy_features
        WHERE season IS NOT NULL
        ORDER BY season DESC
    """)
    seasons = seasons_df['season'].tolist() if not seasons_df.empty else [2025]
    season = st.selectbox("Season", seasons, index=0 if seasons else None)
    
    # Week filter (optional)
    if season:
        weeks_df = fetch_df("""
            SELECT DISTINCT week
            FROM nfl.ai_datasets.player_fantasy_features
            WHERE season = ?
            ORDER BY week
        """, (int(season),))
        weeks = weeks_df['week'].tolist() if not weeks_df.empty else []
        week = st.selectbox("Week (optional)", [None] + weeks, index=0)
    else:
        week = None
    
    st.divider()
    st.markdown("""
    **💡 Teaching Notes:**
    - This dashboard queries the `nfl.mlops` schema for model metadata
    - Predictions are generated via the Cloud Function endpoint
    - The prediction function automatically selects the active deployed model
    """)

# ------------------------------------------------------------------------------
# Main Tabs
# ------------------------------------------------------------------------------
tab_metrics, tab_player, tab_predictions, tab_compare = st.tabs([
    "📊 Metrics Dashboard",
    "👤 Player Lookup",
    "🔮 Live Predictions",
    "⚖️ Model Comparison"
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
    - Performance trends over time
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
        
        # Performance Trends Chart
        st.subheader("📈 Performance Trends Over Time")
        chart_df = versions_df.sort_values('created_at').copy()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df['created_at'],
            y=chart_df['test_mae'],
            mode='lines+markers',
            name='Mean Absolute Error (MAE)',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=chart_df['created_at'],
            y=chart_df['test_rmse'],
            mode='lines+markers',
            name='Root Mean Squared Error (RMSE)',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Model Performance Metrics Over Time (Lower is Better)",
            xaxis_title="Model Version Creation Date",
            yaxis_title="Error Metric",
            hovermode='x unified',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation trend
        if chart_df['test_corr'].notna().any():
            fig2 = px.line(
                chart_df,
                x='created_at',
                y='test_corr',
                title="Test Correlation Over Time (Higher is Better)",
                labels={'test_corr': 'Correlation', 'created_at': 'Date'},
                markers=True
            )
            fig2.update_traces(line=dict(color='#2ca02c', width=3), marker=dict(size=8))
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ℹ️ No model versions found. Run the training pipeline to register models.")
    
    st.divider()
    
    # Training Runs Table
    st.subheader("🏃 Recent Training Runs")
    st.markdown("""
    Each training run represents one model experiment. The registry tracks:
    - Algorithm type (linear_regression, random_forest, etc.)
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
        if runs_df['algorithm'].notna().any():
            st.subheader("🔬 Algorithm Performance Comparison")
            algo_df = runs_df.groupby('algorithm').agg({
                'test_mae': 'mean',
                'test_rmse': 'mean',
                'test_corr': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=algo_df['algorithm'],
                y=algo_df['test_mae'],
                name='Mean MAE',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title="Average Test MAE by Algorithm",
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
    2. Visualizing performance trends by game sequence (Last N games)
    3. **Calling the prediction endpoint** to get model predictions
    4. Comparing predictions vs actuals for a specific player
    5. Viewing 3-week moving average features used for predictions
    """)
    
    # Get list of players (from either gold or ai_datasets)
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
        # Player selection
        player_options = dict(zip(
            players_df['athlete_name'] + " (" + players_df['athlete_id'] + ")",
            players_df['athlete_id']
        ))
        
        selected_player_display = st.selectbox(
            "Select Player",
            options=list(player_options.keys()),
            index=0
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
            # Add game sequence (Last N games) instead of using dates
            hist_df = hist_df.copy()
            hist_df['game_sequence'] = range(1, len(hist_df) + 1)
            
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
            
            # Historical performance chart by SEQUENCE (not date)
            st.subheader("📈 Historical Fantasy Points (Last N Games)")
            fig = px.line(
                hist_df,
                x='game_sequence',
                y='actual_ppr',
                title=f"{player_name} - Fantasy Points by Game Sequence (Not Date)",
                labels={'actual_ppr': 'PPR Points', 'game_sequence': 'Game # (Last N Games)'},
                markers=True
            )
            fig.update_traces(line=dict(color='#2ca02c', width=2), marker=dict(size=4))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Model Predictions Section
            st.subheader("🤖 Model Predictions vs Actuals")
            st.markdown("""
            **This section demonstrates calling the prediction Cloud Function endpoint:**
            - The endpoint uses `ai_datasets.player_fantasy_features` (3-week lookback features)
            - Each row represents a prediction where we used last 3 games to predict current game
            - We compare predictions against actual values
            """)
            
            with st.spinner("Fetching predictions from Cloud Function endpoint..."):
                pred_response = call_prediction_endpoint({"athlete_id": selected_player_id})
            
            if "error" in pred_response:
                st.error(f"❌ Error: {pred_response.get('details', pred_response.get('error'))}")
            elif "predictions" in pred_response:
                pred_df = pd.DataFrame(pred_response['predictions'])
                
                st.success(f"✅ Retrieved {pred_response['count']} predictions using model: **{pred_response.get('model_version_id', 'N/A')}**")
                st.caption(f"Model Type: {pred_response.get('model_type', 'N/A')}")
                
                if not pred_df.empty:
                    # Merge with actuals for comparison
                    comparison_df = pd.merge(
                        pred_df[['game_date', 'season', 'week', 'predicted_ppr']],
                        hist_df[['season', 'week', 'actual_ppr', 'game_sequence']],
                        on=['season', 'week'],
                        how='inner'
                    )
                    
                    if not comparison_df.empty:
                        comparison_df['error'] = comparison_df['predicted_ppr'] - comparison_df['actual_ppr']
                        comparison_df['abs_error'] = comparison_df['error'].abs()
                        
                        # Comparison chart by SEQUENCE
                        st.subheader("📊 Predictions vs Actuals by Game Sequence")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=comparison_df['game_sequence'],
                            y=comparison_df['predicted_ppr'],
                            mode='lines+markers',
                            name='Predicted (from deployed model)',
                            line=dict(color='blue', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                        fig.add_trace(go.Scatter(
                            x=comparison_df['game_sequence'],
                            y=comparison_df['actual_ppr'],
                            mode='lines+markers',
                            name='Actual',
                            line=dict(color='green', width=2),
                            marker=dict(size=6)
                        ))
                        fig.update_layout(
                            title=f"{player_name} - Predictions vs Actuals by Game Sequence",
                            xaxis_title="Game # (Last N Games)",
                            yaxis_title="PPR Points",
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            mae = comparison_df['abs_error'].mean()
                            st.metric("Mean Absolute Error", f"{mae:.2f}")
                        with col2:
                            rmse = (comparison_df['error']**2).mean()**0.5
                            st.metric("RMSE", f"{rmse:.2f}")
                        with col3:
                            corr = comparison_df['predicted_ppr'].corr(comparison_df['actual_ppr'])
                            st.metric("Correlation", f"{corr:.3f}")
                        
                        # Detailed comparison table
                        st.subheader("📋 Detailed Predictions Table")
                        display_cols = ['game_sequence', 'game_date', 'season', 'week', 'predicted_ppr', 'actual_ppr', 'error', 'abs_error']
                        st.dataframe(
                            comparison_df[display_cols].sort_values('game_sequence'),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No overlapping games between predictions and historical data.")
                else:
                    st.info("No predictions returned for this player.")
            else:
                st.error("Unexpected response format from prediction endpoint")
            
            st.divider()
            
            # 3-Week Moving Average Features Section
            st.subheader("📊 3-Week Moving Average Features")
            st.markdown("""
            **The model uses 3-week moving averages as features.** Below shows the most recent features
            for this player, which would be used to predict the NEXT game:
            """)
            
            # Get latest features from ai_datasets for this player (where target is NULL = future prediction)
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
                
                # Visualize feature trends over time
                st.subheader("📈 Feature Trends Over Time")
                fig = go.Figure()
                
                if 'avg_receptions_3w' in features_df.columns:
                    fig.add_trace(go.Scatter(
                        x=features_df['game_date'],
                        y=features_df['avg_receptions_3w'],
                        mode='lines+markers',
                        name='Avg Receptions',
                        line=dict(color='blue')
                    ))
                
                if 'avg_rush_yards_3w' in features_df.columns:
                    fig.add_trace(go.Scatter(
                        x=features_df['game_date'],
                        y=features_df['avg_rush_yards_3w'],
                        mode='lines+markers',
                        name='Avg Rush Yards',
                        line=dict(color='green')
                    ))
                
                if 'avg_receiving_yards_3w' in features_df.columns:
                    fig.add_trace(go.Scatter(
                        x=features_df['game_date'],
                        y=features_df['avg_receiving_yards_3w'],
                        mode='lines+markers',
                        name='Avg Rec Yards',
                        line=dict(color='purple')
                    ))
                
                fig.update_layout(
                    title=f"{player_name} - 3-Week Moving Average Features",
                    xaxis_title="Game Date",
                    yaxis_title="Average Value (Last 3 Games)",
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # All features table
                st.markdown("**All Recent Feature Values:**")
                st.dataframe(features_df, use_container_width=True, hide_index=True)
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
    1. Set filters for predictions (player, season, week)
    2. Call the Cloud Function endpoint with parameters
    3. Display predictions returned by the active deployed model
    
    This is how other applications would integrate with your ML models!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Parameters")
        
        # Player selection (optional)
        players_df = fetch_df("""
            SELECT DISTINCT 
                athlete_id,
                athlete_name
            FROM nfl.ai_datasets.player_fantasy_features
            WHERE athlete_name IS NOT NULL
            ORDER BY athlete_name
            LIMIT 100
        """)
        
        player_options = ["All Players"] + [
            f"{row['athlete_name']} ({row['athlete_id']})" 
            for _, row in players_df.iterrows()
        ]
        
        selected_player_display = st.selectbox(
            "Player (optional)",
            options=player_options,
            index=0
        )
        
        athlete_id = None
        if selected_player_display != "All Players":
            # Extract athlete_id from selection
            athlete_id = selected_player_display.split("(")[-1].rstrip(")")
        
        # Season and week
        pred_season = st.number_input(
            "Season (optional)",
            value=season if season else 2025,
            min_value=2020,
            max_value=2030,
            step=1
        )
        
        pred_week = st.number_input(
            "Week (optional)",
            value=int(week) if week else None,
            min_value=1,
            max_value=18,
            step=1
        )
        
        st.markdown("---")
        
        if st.button("🚀 Get Predictions", type="primary", use_container_width=True):
            # Build parameters
            params = {}
            if athlete_id:
                params["athlete_id"] = athlete_id
            if pred_season:
                params["season"] = int(pred_season)
            if pred_week:
                params["week"] = int(pred_week)
            
            with st.spinner("Calling prediction Cloud Function..."):
                pred_response = call_prediction_endpoint(params)
            
            if "error" in pred_response:
                st.error(f"❌ Error: {pred_response.get('details', pred_response.get('error'))}")
                if "404" in str(pred_response.get('error', '')):
                    st.info("💡 Tip: Make sure a model has been deployed. Check the Metrics Dashboard tab.")
            elif "predictions" in pred_response:
                st.success(f"✅ Success! Retrieved {pred_response['count']} predictions")
                
                # Model info
                with st.expander("📋 Model Information", expanded=True):
                    st.json({
                        "model_version_id": pred_response.get('model_version_id', 'N/A'),
                        "model_type": pred_response.get('model_type', 'N/A'),
                        "prediction_count": pred_response.get('count', 0),
                        "endpoint": prediction_endpoint,
                        "parameters_used": params
                    })
                
                pred_df = pd.DataFrame(pred_response['predictions'])
                
                if not pred_df.empty:
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_pred = pred_df['predicted_ppr'].mean()
                        st.metric("Avg Predicted PPR", f"{avg_pred:.2f}")
                    
                    with col2:
                        max_pred = pred_df['predicted_ppr'].max()
                        st.metric("Max Prediction", f"{max_pred:.2f}")
                    
                    with col3:
                        min_pred = pred_df['predicted_ppr'].min()
                        st.metric("Min Prediction", f"{min_pred:.2f}")
                    
                    with col4:
                        if 'error' in pred_df.columns and pred_df['error'].notna().any():
                            mae = pred_df['abs_error'].mean()
                            st.metric("Mean Absolute Error", f"{mae:.2f}")
                        else:
                            st.metric("Predictions", len(pred_df))
                    
                    # Predictions table
                    st.subheader("📊 Predictions Table")
                    display_cols = ['athlete_name', 'season', 'week', 'game_date', 'predicted_ppr']
                    if 'actual_ppr' in pred_df.columns:
                        display_cols.extend(['actual_ppr', 'error'])
                    
                    st.dataframe(
                        pred_df[display_cols].sort_values(['season', 'week', 'athlete_name']),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Distribution chart
                    if len(pred_df) > 1:
                        st.subheader("📈 Prediction Distribution")
                        fig = px.histogram(
                            pred_df,
                            x='predicted_ppr',
                            nbins=20,
                            title="Distribution of Predicted PPR Points",
                            labels={'predicted_ppr': 'Predicted PPR Points', 'count': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No predictions returned.")
            else:
                st.error("Unexpected response format")
    
    with col2:
        st.subheader("💡 How This Works")
        st.markdown("""
        **API Endpoint Integration:**
        
        This tab demonstrates a real-world integration pattern:
        
        1. **User selects parameters** (player, season, week)
        2. **App calls Cloud Function** via HTTP GET request:
           ```
           GET https://.../ml-predict-fantasy
           ?athlete_id=...&season=...&week=...
           ```
        
        3. **Cloud Function**:
           - Looks up active deployment in `nfl.mlops.deployment`
           - Loads the deployed model (SQL or Python)
           - Generates predictions
           - Returns JSON response
        
        4. **Dashboard displays** predictions interactively
        
        **Key Learning Points:**
        - Models are deployed as Cloud Functions (serverless)
        - The model registry tracks which version is active
        - Predictions can be filtered by player, season, week
        - The same endpoint can be used by other applications
        """)
        
        st.code("""
# Example: How to call the endpoint from Python
import requests

response = requests.get(
    "https://.../ml-predict-fantasy",
    params={
        "athlete_id": "12345",
        "season": 2025,
        "week": 8
    }
)

predictions = response.json()
        """, language="python")

# ============================================================================
# TAB 4: Model Comparison
# ============================================================================
with tab_compare:
    st.header("⚖️ Model Version Comparison")
    st.markdown("""
    Compare different model versions side-by-side to understand:
    - Which algorithms perform best
    - How hyperparameters affect performance
    - Which model versions are approved vs candidates
    """)
    
    # Get available model versions
    versions_list_df = fetch_df("""
        SELECT 
            mv.model_version_id,
            m.name as model_name,
            mv.status,
            CAST(JSON_EXTRACT(mv.metrics_json, '$.ppr.test_mae') AS DECIMAL(10,4)) as test_mae
        FROM nfl.mlops.model_version mv
        JOIN nfl.mlops.model m ON mv.model_id = m.model_id
        ORDER BY mv.created_at DESC
        LIMIT 15
    """)
    
    if versions_list_df.empty:
        st.info("ℹ️ No model versions found for comparison.")
    else:
        version_options = versions_list_df.apply(
            lambda row: f"{row['model_version_id']} (MAE: {row['test_mae']:.2f}, Status: {row['status']})",
            axis=1
        ).tolist()
        
        selected_versions = st.multiselect(
            "Select Model Versions to Compare (choose 2-4 for best visualization)",
            options=version_options,
            default=version_options[:2] if len(version_options) >= 2 else version_options[:1]
        )
        
        if selected_versions:
            # Extract version IDs
            version_ids = [
                opt.split()[0] for opt in selected_versions
            ]
            
            if version_ids:
                versions_str = "', '".join(version_ids)
                comparison_df = fetch_df(f"""
                    SELECT 
                        mv.model_version_id,
                        m.name as model_name,
                        mv.status,
                        CAST(JSON_EXTRACT(mv.metrics_json, '$.ppr.test_rmse') AS DECIMAL(10,4)) as test_rmse,
                        CAST(JSON_EXTRACT(mv.metrics_json, '$.ppr.test_mae') AS DECIMAL(10,4)) as test_mae,
                        CAST(JSON_EXTRACT(mv.metrics_json, '$.ppr.test_correlation') AS DECIMAL(10,4)) as test_corr,
                        CAST(JSON_EXTRACT(mv.metrics_json, '$.ppr.test_count') AS INTEGER) as test_count,
                        mv.created_at
                    FROM nfl.mlops.model_version mv
                    JOIN nfl.mlops.model m ON mv.model_id = m.model_id
                    WHERE mv.model_version_id IN ('{versions_str}')
                """)
                
                if not comparison_df.empty:
                    # Comparison table
                    st.subheader("📋 Metrics Comparison")
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    st.subheader("📊 Side-by-Side Comparison")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['model_version_id'],
                        y=comparison_df['test_mae'],
                        name='MAE',
                        marker_color='lightblue',
                        text=comparison_df['test_mae'].round(2),
                        textposition='outside'
                    ))
                    fig.add_trace(go.Bar(
                        x=comparison_df['model_version_id'],
                        y=comparison_df['test_rmse'],
                        name='RMSE',
                        marker_color='lightcoral',
                        text=comparison_df['test_rmse'].round(2),
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Error Metrics Comparison (Lower is Better)",
                        xaxis_title="Model Version",
                        yaxis_title="Error Metric",
                        barmode='group',
                        height=400,
                        xaxis={'tickangle': -45}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation comparison
                    fig2 = px.bar(
                        comparison_df,
                        x='model_version_id',
                        y='test_corr',
                        title="Correlation Comparison (Higher is Better)",
                        labels={'test_corr': 'Test Correlation', 'model_version_id': 'Model Version'},
                        color='test_corr',
                        color_continuous_scale='Greens'
                    )
                    fig2.update_layout(
                        height=350,
                        xaxis={'tickangle': -45},
                        showlegend=False
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Select at least one model version to compare.")

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
    
    6. **Sequence-Based Visualization**:
       - Historical trends plotted by "Last N Games" instead of dates
       - Handles gaps in game schedule (bye weeks, injuries)
       - Better reflects player performance patterns
    
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
    - This dashboard: Consumer of both registry and endpoint
    """)

