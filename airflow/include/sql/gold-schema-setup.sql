    -- ONE-TIME SETUP
    -- Create target + scratch schemas in the *nfl* database
    CREATE SCHEMA IF NOT EXISTS nfl.gold;
    CREATE SCHEMA IF NOT EXISTS nfl.scratch;

    -- Create target table shell if missing
    CREATE TABLE IF NOT EXISTS nfl.gold.player_game_stats AS
    SELECT
    CAST(NULL AS BIGINT)   AS game_id,
    CAST(NULL AS BIGINT)   AS athlete_id,
    CAST(NULL AS TEXT)     AS athlete_name,
    CAST(NULL AS INTEGER)  AS team_id,
    CAST(NULL AS TEXT)     AS team_abbrev,
    CAST(NULL AS INTEGER)  AS season,
    CAST(NULL AS INTEGER)  AS week,
    CAST(NULL AS DATE)     AS game_date,
    CAST(NULL AS BOOLEAN)  AS is_home,
    CAST(NULL AS INTEGER)  AS opponent_team_id,
    CAST(NULL AS TEXT)     AS opponent_abbrev,

    CAST(NULL AS DECIMAL(10,2)) AS pass_completions,
    CAST(NULL AS DECIMAL(10,2)) AS pass_attempts,
    CAST(NULL AS DECIMAL(10,2)) AS pass_yards,
    CAST(NULL AS DECIMAL(10,2)) AS pass_touchdowns,
    CAST(NULL AS DECIMAL(10,2)) AS interceptions,
    CAST(NULL AS DECIMAL(10,2)) AS sacks,
    CAST(NULL AS DECIMAL(10,2)) AS qb_rating,

    CAST(NULL AS DECIMAL(10,2)) AS rush_attempts,
    CAST(NULL AS DECIMAL(10,2)) AS rush_yards,
    CAST(NULL AS DECIMAL(10,2)) AS rush_touchdowns,

    CAST(NULL AS DECIMAL(10,2)) AS receptions,
    CAST(NULL AS DECIMAL(10,2)) AS receiving_targets,
    CAST(NULL AS DECIMAL(10,2)) AS receiving_yards,
    CAST(NULL AS DECIMAL(10,2)) AS receiving_touchdowns,

    CAST(NULL AS DECIMAL(10,2)) AS fumbles,
    CAST(NULL AS DECIMAL(10,2)) AS fumbles_lost,

    CAST(NULL AS DECIMAL(10,2)) AS field_goals_made,
    CAST(NULL AS DECIMAL(10,2)) AS field_goal_attempts,
    CAST(NULL AS DECIMAL(10,2)) AS extra_points_made,
    CAST(NULL AS DECIMAL(10,2)) AS extra_point_attempts,

    CAST(NULL AS DECIMAL(10,2)) AS total_tackles,
    CAST(NULL AS DECIMAL(10,2)) AS sacks_def,
    CAST(NULL AS DECIMAL(10,2)) AS interceptions_def,
    CAST(NULL AS DECIMAL(10,2)) AS fumbles_recovered,
    CAST(NULL AS DECIMAL(10,2)) AS defensive_touchdowns,

    CAST(NULL AS DECIMAL(10,2)) AS fantasy_points_ppr,
    CAST(NULL AS DECIMAL(10,2)) AS fantasy_points_standard
    LIMIT 0;