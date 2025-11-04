-- ============================================================================
-- Create Player Fantasy Features Dataset
-- ============================================================================
-- Purpose: Generate ML dataset with 3-week moving average features
--          to predict current week fantasy points (PPR & Standard)
-- Grain: One row per player per game (after 3-week lookback satisfied)
-- Output: nfl.ai_datasets.player_fantasy_features
-- ============================================================================

CREATE OR REPLACE TABLE nfl.ai_datasets.player_fantasy_features AS
WITH player_features AS (
  SELECT 
    -- Identifiers
    game_id,
    athlete_id,
    athlete_name,
    team_id,
    team_abbrev,
    season,
    week,
    game_date,
    
    -- Context features (current week, not averaged)
    is_home,
    opponent_team_id,
    opponent_abbrev,
    
    -- Target variables (current week's actual fantasy points)
    fantasy_points_ppr as target_fantasy_ppr,
    fantasy_points_standard as target_fantasy_standard,
    
    -- Features: 3-week moving averages of all stats
    -- ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING excludes current week
    AVG(pass_completions) OVER w3 as avg_pass_completions_3w,
    AVG(pass_attempts) OVER w3 as avg_pass_attempts_3w,
    AVG(pass_yards) OVER w3 as avg_pass_yards_3w,
    AVG(pass_touchdowns) OVER w3 as avg_pass_touchdowns_3w,
    AVG(interceptions) OVER w3 as avg_interceptions_3w,
    AVG(sacks) OVER w3 as avg_sacks_3w,
    AVG(qb_rating) OVER w3 as avg_qb_rating_3w,
    
    AVG(rush_attempts) OVER w3 as avg_rush_attempts_3w,
    AVG(rush_yards) OVER w3 as avg_rush_yards_3w,
    AVG(rush_touchdowns) OVER w3 as avg_rush_touchdowns_3w,
    
    AVG(receptions) OVER w3 as avg_receptions_3w,
    AVG(receiving_targets) OVER w3 as avg_receiving_targets_3w,
    AVG(receiving_yards) OVER w3 as avg_receiving_yards_3w,
    AVG(receiving_touchdowns) OVER w3 as avg_receiving_touchdowns_3w,
    
    AVG(fumbles) OVER w3 as avg_fumbles_3w,
    AVG(fumbles_lost) OVER w3 as avg_fumbles_lost_3w,
    
    AVG(field_goals_made) OVER w3 as avg_field_goals_made_3w,
    AVG(field_goal_attempts) OVER w3 as avg_field_goal_attempts_3w,
    AVG(extra_points_made) OVER w3 as avg_extra_points_made_3w,
    AVG(extra_point_attempts) OVER w3 as avg_extra_point_attempts_3w,
    
    AVG(total_tackles) OVER w3 as avg_total_tackles_3w,
    AVG(sacks_def) OVER w3 as avg_sacks_def_3w,
    AVG(interceptions_def) OVER w3 as avg_interceptions_def_3w,
    AVG(fumbles_recovered) OVER w3 as avg_fumbles_recovered_3w,
    AVG(defensive_touchdowns) OVER w3 as avg_defensive_touchdowns_3w,
    
    -- Count how many games in the lookback window
    COUNT(*) OVER w3 as lookback_count
    
  FROM nfl.gold.player_game_stats
  
  WINDOW w3 AS (
    PARTITION BY athlete_id, season 
    ORDER BY game_date 
    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
  )
),

valid_predictions AS (
  SELECT *
  FROM player_features
  WHERE lookback_count = 3
    AND target_fantasy_ppr IS NOT NULL
)

SELECT 
  -- Identifiers
  game_id,
  athlete_id,
  athlete_name,
  team_id,
  team_abbrev,
  season,
  week,
  game_date,
  
  -- Context
  is_home,
  opponent_team_id,
  opponent_abbrev,
  
  -- Features (3-week moving averages)
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
  
  -- Targets
  target_fantasy_ppr,
  target_fantasy_standard,
  
  -- Train/Test Split (temporal, based on game_date - first 70% of games are train)
  CASE 
    WHEN PERCENT_RANK() OVER (ORDER BY game_date) < 0.70 THEN 'train'
    ELSE 'test'
  END as split,
  
  -- Metadata
  NOW() as created_at

FROM valid_predictions
ORDER BY athlete_id, game_date;