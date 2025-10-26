INSERT INTO nfl.model_outputs.predictions_fantasy (
  run_id,
  game_id,
  athlete_id,
  athlete_name,
  season,
  week,
  game_date,
  split,
  actual_ppr,
  predicted_ppr,
  actual_standard,
  predicted_standard,
  created_at
)
SELECT 
  '{{ run_id }}' as run_id,
  game_id,
  athlete_id,
  athlete_name,
  season,
  week,
  game_date,
  split,
  
  -- Actual values (ground truth)
  target_fantasy_ppr as actual_ppr,
  target_fantasy_standard as actual_standard,
  
  -- Predictions: Apply fantasy scoring to 3-week averages
  -- PPR Scoring
  COALESCE(
    (avg_pass_yards_3w * 0.04) +
    (avg_pass_touchdowns_3w * 4) +
    (avg_interceptions_3w * -2) +
    (avg_rush_yards_3w * 0.1) +
    (avg_rush_touchdowns_3w * 6) +
    (avg_receptions_3w * 1) +              -- PPR bonus
    (avg_receiving_yards_3w * 0.1) +
    (avg_receiving_touchdowns_3w * 6) +
    (avg_fumbles_lost_3w * -2) +
    (avg_field_goals_made_3w * 3) +
    (avg_extra_points_made_3w * 1),
    0
  ) as predicted_ppr,
  
  -- Standard Scoring (no reception bonus)
  COALESCE(
    (avg_pass_yards_3w * 0.04) +
    (avg_pass_touchdowns_3w * 4) +
    (avg_interceptions_3w * -2) +
    (avg_rush_yards_3w * 0.1) +
    (avg_rush_touchdowns_3w * 6) +
    (avg_receiving_yards_3w * 0.1) +
    (avg_receiving_touchdowns_3w * 6) +
    (avg_fumbles_lost_3w * -2) +
    (avg_field_goals_made_3w * 3) +
    (avg_extra_points_made_3w * 1),
    0
  ) as predicted_standard,
  
  NOW() as created_at
  
FROM nfl.ai_datasets.player_fantasy_features
WHERE target_fantasy_ppr IS NOT NULL;