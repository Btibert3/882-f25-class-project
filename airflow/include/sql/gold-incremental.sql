-- ============================================================================
-- Incremental Load for gold.player_game_stats
-- Purpose: Load only NEW data from stage that hasn't been processed yet
-- Run: After stage deduplication completes
-- ============================================================================

-- Insert only NEW player-game combinations that don't exist in gold yet
INSERT INTO nfl.gold.player_game_stats (
    game_id,
    athlete_id,
    athlete_name,
    team_id,
    team_abbrev,
    season,
    week,
    game_date,
    is_home,
    opponent_team_id,
    opponent_abbrev,
    pass_completions,
    pass_attempts,
    pass_yards,
    pass_touchdowns,
    interceptions,
    sacks,
    qb_rating,
    rush_attempts,
    rush_yards,
    rush_touchdowns,
    receptions,
    receiving_targets,
    receiving_yards,
    receiving_touchdowns,
    fumbles,
    fumbles_lost,
    field_goals_made,
    field_goal_attempts,
    extra_points_made,
    extra_point_attempts,
    total_tackles,
    sacks_def,
    interceptions_def,
    fumbles_recovered,
    defensive_touchdowns,
    fantasy_points_ppr,
    fantasy_points_standard,
    loaded_at
)
WITH player_game_base AS (
    SELECT DISTINCT
        fps.game_id,
        fps.athlete_id,
        fps.athlete_name,
        CAST(fps.team_id AS INTEGER) as team_id,
        g.season,
        g.week,
        g.start_date as game_date,
        fgt.home_away,
        CASE WHEN fgt.home_away = 'home' THEN TRUE ELSE FALSE END as is_home
    FROM stage.fact_player_stats fps
    JOIN stage.dim_games g ON fps.game_id = g.id
    JOIN stage.fact_game_team fgt ON fps.game_id = fgt.game_id 
        AND fps.team_id = CAST(fgt.team_id AS VARCHAR)
    -- ONLY process records that don't exist in gold yet
    WHERE NOT EXISTS (
        SELECT 1 
        FROM gold.player_game_stats existing
        WHERE existing.game_id = fps.game_id
          AND existing.athlete_id = fps.athlete_id
    )
),
stats_pivoted AS (
    SELECT
        fps.game_id,
        fps.athlete_id,
        MAX(CASE WHEN fps.stat_key = 'completions/passingAttempts' THEN SPLIT_PART(fps.value_str, '/', 1) END) as pass_completions_str,
        MAX(CASE WHEN fps.stat_key = 'completions/passingAttempts' THEN SPLIT_PART(fps.value_str, '/', 2) END) as pass_attempts_str,
        MAX(CASE WHEN fps.stat_key = 'passingYards' THEN fps.value_str END) as pass_yards_str,
        MAX(CASE WHEN fps.stat_key = 'passingTouchdowns' THEN fps.value_str END) as pass_touchdowns_str,
        MAX(CASE WHEN fps.stat_key = 'interceptions' AND fps.category = 'passing' THEN fps.value_str END) as interceptions_str,
        MAX(CASE WHEN fps.stat_key = 'sacks-sackYardsLost' THEN SPLIT_PART(fps.value_str, '-', 1) END) as sacks_str,
        MAX(CASE WHEN fps.stat_key = 'QBRating' THEN fps.value_str END) as qb_rating_str,
        MAX(CASE WHEN fps.stat_key = 'rushingAttempts' THEN fps.value_str END) as rush_attempts_str,
        MAX(CASE WHEN fps.stat_key = 'rushingYards' THEN fps.value_str END) as rush_yards_str,
        MAX(CASE WHEN fps.stat_key = 'rushingTouchdowns' THEN fps.value_str END) as rush_touchdowns_str,
        MAX(CASE WHEN fps.stat_key = 'receptions' THEN fps.value_str END) as receptions_str,
        MAX(CASE WHEN fps.stat_key = 'receivingTargets' THEN fps.value_str END) as receiving_targets_str,
        MAX(CASE WHEN fps.stat_key = 'receivingYards' THEN fps.value_str END) as receiving_yards_str,
        MAX(CASE WHEN fps.stat_key = 'receivingTouchdowns' THEN fps.value_str END) as receiving_touchdowns_str,
        MAX(CASE WHEN fps.stat_key = 'fumbles' AND fps.category = 'fumbles' THEN fps.value_str END) as fumbles_str,
        MAX(CASE WHEN fps.stat_key = 'fumblesLost' THEN fps.value_str END) as fumbles_lost_str,
        MAX(CASE WHEN fps.stat_key = 'fieldGoalsMade/fieldGoalAttempts' THEN SPLIT_PART(fps.value_str, '/', 1) END) as field_goals_made_str,
        MAX(CASE WHEN fps.stat_key = 'fieldGoalsMade/fieldGoalAttempts' THEN SPLIT_PART(fps.value_str, '/', 2) END) as field_goal_attempts_str,
        MAX(CASE WHEN fps.stat_key = 'extraPointsMade/extraPointAttempts' THEN SPLIT_PART(fps.value_str, '/', 1) END) as extra_points_made_str,
        MAX(CASE WHEN fps.stat_key = 'extraPointsMade/extraPointAttempts' THEN SPLIT_PART(fps.value_str, '/', 2) END) as extra_point_attempts_str,
        MAX(CASE WHEN fps.stat_key = 'totalTackles' THEN fps.value_str END) as total_tackles_str,
        MAX(CASE WHEN fps.stat_key = 'sacks' AND fps.category = 'defensive' THEN fps.value_str END) as sacks_def_str,
        MAX(CASE WHEN fps.stat_key = 'interceptions' AND fps.category = 'interceptions' THEN fps.value_str END) as interceptions_def_str,
        MAX(CASE WHEN fps.stat_key = 'fumblesRecovered' THEN fps.value_str END) as fumbles_recovered_str,
        MAX(CASE WHEN fps.stat_key = 'defensiveTouchdowns' THEN fps.value_str END) as defensive_touchdowns_str
    FROM stage.fact_player_stats fps
    WHERE fps.game_id IN (SELECT game_id FROM player_game_base)
      AND fps.athlete_id IN (SELECT athlete_id FROM player_game_base)
    GROUP BY fps.game_id, fps.athlete_id
)
SELECT
    pgb.game_id,
    pgb.athlete_id,
    pgb.athlete_name,
    pgb.team_id,
    t.abbrev as team_abbrev,
    pgb.season,
    pgb.week,
    pgb.game_date,
    pgb.is_home,
    opp.team_id as opponent_team_id,
    opp.abbrev as opponent_abbrev,
    COALESCE(TRY_CAST(sp.pass_completions_str AS DECIMAL(10,2)), 0) as pass_completions,
    COALESCE(TRY_CAST(sp.pass_attempts_str AS DECIMAL(10,2)), 0) as pass_attempts,
    COALESCE(TRY_CAST(sp.pass_yards_str AS DECIMAL(10,2)), 0) as pass_yards,
    COALESCE(TRY_CAST(sp.pass_touchdowns_str AS DECIMAL(10,2)), 0) as pass_touchdowns,
    COALESCE(TRY_CAST(sp.interceptions_str AS DECIMAL(10,2)), 0) as interceptions,
    COALESCE(TRY_CAST(sp.sacks_str AS DECIMAL(10,2)), 0) as sacks,
    COALESCE(TRY_CAST(sp.qb_rating_str AS DECIMAL(10,2)), 0) as qb_rating,
    COALESCE(TRY_CAST(sp.rush_attempts_str AS DECIMAL(10,2)), 0) as rush_attempts,
    COALESCE(TRY_CAST(sp.rush_yards_str AS DECIMAL(10,2)), 0) as rush_yards,
    COALESCE(TRY_CAST(sp.rush_touchdowns_str AS DECIMAL(10,2)), 0) as rush_touchdowns,
    COALESCE(TRY_CAST(sp.receptions_str AS DECIMAL(10,2)), 0) as receptions,
    COALESCE(TRY_CAST(sp.receiving_targets_str AS DECIMAL(10,2)), 0) as receiving_targets,
    COALESCE(TRY_CAST(sp.receiving_yards_str AS DECIMAL(10,2)), 0) as receiving_yards,
    COALESCE(TRY_CAST(sp.receiving_touchdowns_str AS DECIMAL(10,2)), 0) as receiving_touchdowns,
    COALESCE(TRY_CAST(sp.fumbles_str AS DECIMAL(10,2)), 0) as fumbles,
    COALESCE(TRY_CAST(sp.fumbles_lost_str AS DECIMAL(10,2)), 0) as fumbles_lost,
    COALESCE(TRY_CAST(sp.field_goals_made_str AS DECIMAL(10,2)), 0) as field_goals_made,
    COALESCE(TRY_CAST(sp.field_goal_attempts_str AS DECIMAL(10,2)), 0) as field_goal_attempts,
    COALESCE(TRY_CAST(sp.extra_points_made_str AS DECIMAL(10,2)), 0) as extra_points_made,
    COALESCE(TRY_CAST(sp.extra_point_attempts_str AS DECIMAL(10,2)), 0) as extra_point_attempts,
    COALESCE(TRY_CAST(sp.total_tackles_str AS DECIMAL(10,2)), 0) as total_tackles,
    COALESCE(TRY_CAST(sp.sacks_def_str AS DECIMAL(10,2)), 0) as sacks_def,
    COALESCE(TRY_CAST(sp.interceptions_def_str AS DECIMAL(10,2)), 0) as interceptions_def,
    COALESCE(TRY_CAST(sp.fumbles_recovered_str AS DECIMAL(10,2)), 0) as fumbles_recovered,
    COALESCE(TRY_CAST(sp.defensive_touchdowns_str AS DECIMAL(10,2)), 0) as defensive_touchdowns,
    COALESCE(TRY_CAST(sp.pass_yards_str AS DECIMAL(10,2)), 0) * 0.04 +
    COALESCE(TRY_CAST(sp.pass_touchdowns_str AS DECIMAL(10,2)), 0) * 4 +
    COALESCE(TRY_CAST(sp.interceptions_str AS DECIMAL(10,2)), 0) * -2 +
    COALESCE(TRY_CAST(sp.rush_yards_str AS DECIMAL(10,2)), 0) * 0.1 +
    COALESCE(TRY_CAST(sp.rush_touchdowns_str AS DECIMAL(10,2)), 0) * 6 +
    COALESCE(TRY_CAST(sp.receptions_str AS DECIMAL(10,2)), 0) * 1 +
    COALESCE(TRY_CAST(sp.receiving_yards_str AS DECIMAL(10,2)), 0) * 0.1 +
    COALESCE(TRY_CAST(sp.receiving_touchdowns_str AS DECIMAL(10,2)), 0) * 6 +
    COALESCE(TRY_CAST(sp.fumbles_lost_str AS DECIMAL(10,2)), 0) * -2 +
    COALESCE(TRY_CAST(sp.field_goals_made_str AS DECIMAL(10,2)), 0) * 3 +
    COALESCE(TRY_CAST(sp.extra_points_made_str AS DECIMAL(10,2)), 0) * 1 
    AS fantasy_points_ppr,
    COALESCE(TRY_CAST(sp.pass_yards_str AS DECIMAL(10,2)), 0) * 0.04 +
    COALESCE(TRY_CAST(sp.pass_touchdowns_str AS DECIMAL(10,2)), 0) * 4 +
    COALESCE(TRY_CAST(sp.interceptions_str AS DECIMAL(10,2)), 0) * -2 +
    COALESCE(TRY_CAST(sp.rush_yards_str AS DECIMAL(10,2)), 0) * 0.1 +
    COALESCE(TRY_CAST(sp.rush_touchdowns_str AS DECIMAL(10,2)), 0) * 6 +
    COALESCE(TRY_CAST(sp.receiving_yards_str AS DECIMAL(10,2)), 0) * 0.1 +
    COALESCE(TRY_CAST(sp.receiving_touchdowns_str AS DECIMAL(10,2)), 0) * 6 +
    COALESCE(TRY_CAST(sp.fumbles_lost_str AS DECIMAL(10,2)), 0) * -2 +
    COALESCE(TRY_CAST(sp.field_goals_made_str AS DECIMAL(10,2)), 0) * 3 +
    COALESCE(TRY_CAST(sp.extra_points_made_str AS DECIMAL(10,2)), 0) * 1 
    AS fantasy_points_standard,
    CURRENT_TIMESTAMP as loaded_at
FROM player_game_base pgb
JOIN stats_pivoted sp ON pgb.game_id = sp.game_id AND pgb.athlete_id = sp.athlete_id
LEFT JOIN stage.dim_teams t ON pgb.team_id = t.id
LEFT JOIN LATERAL (
    SELECT fgt2.team_id, t2.abbrev
    FROM stage.fact_game_team fgt2
    JOIN stage.dim_teams t2 ON fgt2.team_id = t2.id
    WHERE fgt2.game_id = pgb.game_id 
      AND fgt2.team_id != pgb.team_id
    LIMIT 1
) opp ON TRUE;