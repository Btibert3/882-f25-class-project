-- Load training data from nfl.ai_datasets.player_fantasy_features
-- Returns all features (avg_* columns + is_home) and target (target_fantasy_ppr)
SELECT 
    athlete_id,
    game_id,
    target_fantasy_ppr,
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
WHERE split = 'train'
  AND target_fantasy_ppr IS NOT NULL

