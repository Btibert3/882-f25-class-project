from datetime import timedelta, datetime
from airflow.sdk import dag, task
import duckdb  
import os

# gold setup
def build_schema():
    SQL = """
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
    """
    return SQL


# incremental
def incremental():
    SQL = """
    -- INCREMENTAL LOAD (rolling window; all writes to nfl.* only)

    BEGIN TRANSACTION;

    -- 1) Recent games window (tune as needed)
    CREATE OR REPLACE TABLE nfl.scratch._recent_games AS
    SELECT g.id AS game_id
    FROM stage.dim_games g
    WHERE g.start_date >= (CURRENT_DATE - INTERVAL 21 DAY);

    -- 2) Facts for those games
    CREATE OR REPLACE TABLE nfl.scratch._delta_fps AS
    SELECT *
    FROM stage.fact_player_stats
    WHERE game_id IN (SELECT game_id FROM nfl.scratch._recent_games);

    -- 3) Keys to upsert
    CREATE OR REPLACE TABLE nfl.scratch._delta_keys AS
    SELECT DISTINCT game_id, athlete_id
    FROM nfl.scratch._delta_fps;

    -- 4) Player/game context
    CREATE OR REPLACE TABLE nfl.scratch._player_game_base AS
    SELECT DISTINCT
    fps.game_id,
    fps.athlete_id,
    fps.athlete_name,
    CAST(fps.team_id AS INTEGER) AS team_id,
    g.season,
    g.week,
    g.start_date AS game_date,
    fgt.home_away,
    CASE WHEN fgt.home_away = 'home' THEN TRUE ELSE FALSE END AS is_home
    FROM nfl.scratch._delta_fps fps
    JOIN stage.dim_games g
    ON fps.game_id = g.id
    JOIN stage.fact_game_team fgt
    ON fgt.game_id = fps.game_id
    AND fgt.team_id = CAST(fps.team_id AS VARCHAR);

    -- 5) Pivot stats
    CREATE OR REPLACE TABLE nfl.scratch._stats_pivoted AS
    SELECT
    game_id,
    athlete_id,

    MAX(CASE WHEN stat_key = 'completions/passingAttempts' THEN split_part(value_str,'/',1) END) AS pass_completions_str,
    MAX(CASE WHEN stat_key = 'completions/passingAttempts' THEN split_part(value_str,'/',2) END) AS pass_attempts_str,
    MAX(CASE WHEN stat_key = 'passingYards' THEN value_str END) AS pass_yards_str,
    MAX(CASE WHEN stat_key = 'passingTouchdowns' THEN value_str END) AS pass_touchdowns_str,
    MAX(CASE WHEN stat_key = 'interceptions' AND category = 'passing' THEN value_str END) AS interceptions_str,
    MAX(CASE WHEN stat_key = 'sacks-sackYardsLost' THEN split_part(value_str,'-',1) END) AS sacks_str,
    MAX(CASE WHEN stat_key = 'QBRating' THEN value_str END) AS qb_rating_str,

    MAX(CASE WHEN stat_key = 'rushingAttempts' THEN value_str END) AS rush_attempts_str,
    MAX(CASE WHEN stat_key = 'rushingYards' THEN value_str END) AS rush_yards_str,
    MAX(CASE WHEN stat_key = 'rushingTouchdowns' THEN value_str END) AS rush_touchdowns_str,

    MAX(CASE WHEN stat_key = 'receptions' THEN value_str END) AS receptions_str,
    MAX(CASE WHEN stat_key = 'receivingTargets' THEN value_str END) AS receiving_targets_str,
    MAX(CASE WHEN stat_key = 'receivingYards' THEN value_str END) AS receiving_yards_str,
    MAX(CASE WHEN stat_key = 'receivingTouchdowns' THEN value_str END) AS receiving_touchdowns_str,

    MAX(CASE WHEN stat_key = 'fumbles' AND category = 'fumbles' THEN value_str END) AS fumbles_str,
    MAX(CASE WHEN stat_key = 'fumblesLost' THEN value_str END) AS fumbles_lost_str,

    MAX(CASE WHEN stat_key = 'fieldGoalsMade/fieldGoalAttempts' THEN split_part(value_str,'/',1) END) AS field_goals_made_str,
    MAX(CASE WHEN stat_key = 'fieldGoalsMade/fieldGoalAttempts' THEN split_part(value_str,'/',2) END) AS field_goal_attempts_str,
    MAX(CASE WHEN stat_key = 'extraPointsMade/extraPointAttempts' THEN split_part(value_str,'/',1) END) AS extra_points_made_str,
    MAX(CASE WHEN stat_key = 'extraPointsMade/extraPointAttempts' THEN split_part(value_str,'/',2) END) AS extra_point_attempts_str,

    MAX(CASE WHEN stat_key = 'totalTackles' THEN value_str END) AS total_tackles_str,
    MAX(CASE WHEN stat_key = 'sacks' AND category = 'defensive' THEN value_str END) AS sacks_def_str,
    MAX(CASE WHEN stat_key = 'interceptions' AND category = 'interceptions' THEN value_str END) AS interceptions_def_str,
    MAX(CASE WHEN stat_key = 'fumblesRecovered' THEN value_str END) AS fumbles_recovered_str,
    MAX(CASE WHEN stat_key = 'defensiveTouchdowns' THEN value_str END) AS defensive_touchdowns_str
    FROM nfl.scratch._delta_fps
    GROUP BY game_id, athlete_id;

    -- 6) Final rows (opponent + casts + fantasy)
    CREATE OR REPLACE TABLE nfl.scratch._final_delta AS
    SELECT
    pgb.game_id,
    pgb.athlete_id,
    pgb.athlete_name,
    pgb.team_id,
    t.abbrev AS team_abbrev,
    pgb.season,
    pgb.week,
    pgb.game_date,
    pgb.is_home,

    CAST(opp_team.team_id AS INTEGER) AS opponent_team_id,
    opp.abbrev AS opponent_abbrev,

    TRY_CAST(sp.pass_completions_str AS DECIMAL(10,2)) AS pass_completions,
    TRY_CAST(sp.pass_attempts_str    AS DECIMAL(10,2)) AS pass_attempts,
    TRY_CAST(sp.pass_yards_str       AS DECIMAL(10,2)) AS pass_yards,
    TRY_CAST(sp.pass_touchdowns_str  AS DECIMAL(10,2)) AS pass_touchdowns,
    TRY_CAST(sp.interceptions_str    AS DECIMAL(10,2)) AS interceptions,
    TRY_CAST(sp.sacks_str            AS DECIMAL(10,2)) AS sacks,
    TRY_CAST(sp.qb_rating_str        AS DECIMAL(10,2)) AS qb_rating,

    TRY_CAST(sp.rush_attempts_str    AS DECIMAL(10,2)) AS rush_attempts,
    TRY_CAST(sp.rush_yards_str       AS DECIMAL(10,2)) AS rush_yards,
    TRY_CAST(sp.rush_touchdowns_str  AS DECIMAL(10,2)) AS rush_touchdowns,

    TRY_CAST(sp.receptions_str       AS DECIMAL(10,2)) AS receptions,
    TRY_CAST(sp.receiving_targets_str AS DECIMAL(10,2)) AS receiving_targets,
    TRY_CAST(sp.receiving_yards_str  AS DECIMAL(10,2)) AS receiving_yards,
    TRY_CAST(sp.receiving_touchdowns_str AS DECIMAL(10,2)) AS receiving_touchdowns,

    TRY_CAST(sp.fumbles_str          AS DECIMAL(10,2)) AS fumbles,
    TRY_CAST(sp.fumbles_lost_str     AS DECIMAL(10,2)) AS fumbles_lost,

    TRY_CAST(sp.field_goals_made_str     AS DECIMAL(10,2)) AS field_goals_made,
    TRY_CAST(sp.field_goal_attempts_str  AS DECIMAL(10,2)) AS field_goal_attempts,
    TRY_CAST(sp.extra_points_made_str    AS DECIMAL(10,2)) AS extra_points_made,
    TRY_CAST(sp.extra_point_attempts_str AS DECIMAL(10,2)) AS extra_point_attempts,

    TRY_CAST(sp.total_tackles_str     AS DECIMAL(10,2)) AS total_tackles,
    TRY_CAST(sp.sacks_def_str         AS DECIMAL(10,2)) AS sacks_def,
    TRY_CAST(sp.interceptions_def_str AS DECIMAL(10,2)) AS interceptions_def,
    TRY_CAST(sp.fumbles_recovered_str AS DECIMAL(10,2)) AS fumbles_recovered,
    TRY_CAST(sp.defensive_touchdowns_str AS DECIMAL(10,2)) AS defensive_touchdowns,

    -- PPR
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

    -- Standard
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
    AS fantasy_points_standard

    FROM nfl.scratch._player_game_base pgb
    JOIN nfl.scratch._stats_pivoted sp USING (game_id, athlete_id)
    LEFT JOIN stage.dim_teams t
    ON pgb.team_id = t.id
    LEFT JOIN stage.fact_game_team opp_team
    ON opp_team.game_id = pgb.game_id
    AND CAST(opp_team.team_id AS INTEGER) <> pgb.team_id
    LEFT JOIN stage.dim_teams opp
    ON opp.id = opp_team.team_id;

    -- 7) Upsert (delete replaced keys, insert fresh rows)
    DELETE FROM nfl.gold.player_game_stats g
    WHERE EXISTS (
    SELECT 1
    FROM nfl.scratch._delta_keys dk
    WHERE dk.game_id = g.game_id
        AND dk.athlete_id = g.athlete_id
    );

    INSERT INTO nfl.gold.player_game_stats
    SELECT * FROM nfl.scratch._final_delta;

    -- 8) (Optional) cleanup scratch
    DROP TABLE IF EXISTS nfl.scratch._final_delta;
    DROP TABLE IF EXISTS nfl.scratch._stats_pivoted;
    DROP TABLE IF EXISTS nfl.scratch._player_game_base;
    DROP TABLE IF EXISTS nfl.scratch._delta_keys;
    DROP TABLE IF EXISTS nfl.scratch._delta_fps;
    DROP TABLE IF EXISTS nfl.scratch._recent_games;

    COMMIT;

    """
    return SQL 


# run a sql command with transactions
def run_execute(SQL):
    # connect to motherduck, set the database
    md = duckdb.connect(f'md:nfl?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}') 
    # get the sql for the schema
    print("starting the SQL transcation")
    try:
        md.execute("BEGIN")
        for stmt in [s for s in SQL.split(";") if s.strip()]:
            print("trying statement --------------------")
            print(stmt)
            md.execute(stmt)
        md.execute("COMMIT")
    except Exception:
        try:
            md.execute("ROLLBACK")
        except Exception:
            print("The Rollback attempt may have fallen over")
            pass
        raise
    finally:
        md.close()


# run a sql command with transactions
def run_sql(SQL):
    # connect to motherduck, set the database
    md = duckdb.connect(f'md:nfl?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}') 
    # get the sql for the schema
    print("starting the SQL transcation")
    try:
        md.sql(SQL)
    except Exception:
        try:
            md.execute("ROLLBACK")
        except Exception:
            print("The Rollback attempt may have fallen over")
            pass
        raise
    finally:
        md.close()


# the dag
@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold"]
)
def gold():

    @task 
    def setup_schema():
        s = build_schema()
        run_sql(s)
    
    @task 
    def incremental():
        s = incremental()
        run_sql(s)

    setup_schema() >> incremental()


gold()
