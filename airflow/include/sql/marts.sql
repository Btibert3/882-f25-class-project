-- ensure the schema exists (idempotent)
CREATE SCHEMA IF NOT EXISTS nfl.mart;

-- define the view that lives inside that schema
CREATE OR REPLACE VIEW nfl.mart.v_player_game_base AS
SELECT
  player_id,
  player_name,
  team_id,
  team_abbrev,
  CAST(season AS INTEGER) AS season,
  CAST(week   AS INTEGER) AS week,
  CAST(game_date AS DATE) AS game_date,
  COALESCE(fantasy_points_standard, fantasy_points_ppr) AS fantasy_points
FROM nfl.gold.player_game_stats
WHERE COALESCE(fantasy_points_standard, fantasy_points_ppr) IS NOT NULL;

-- would have expected the grain of one row per player/week/season if they had stats, and the target is fantasy points
-- the features would be the stats (0, no null) with moving agerages of lags, and the target is the players fantasy points
-- this is a something that BI and analysts can review, but will be used to snapshot datsets for (re)training a model.

