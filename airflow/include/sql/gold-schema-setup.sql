-- ============================================================================
-- Task: Setup Gold Schema and player_game_stats Table
-- Purpose: Create schema and table structure (idempotent)
-- Run: Once at pipeline initialization, safe to rerun
-- ============================================================================

-- Create gold schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS gold;

-- Create player_game_stats table with primary key
CREATE TABLE IF NOT EXISTS gold.player_game_stats (
    -- Identifiers
    game_id INTEGER NOT NULL,
    athlete_id VARCHAR NOT NULL,
    athlete_name VARCHAR,
    team_id INTEGER,
    team_abbrev VARCHAR,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_date TIMESTAMP,
    is_home BOOLEAN,
    
    -- Opponent info
    opponent_team_id INTEGER,
    opponent_abbrev VARCHAR,
    
    -- Passing stats
    pass_completions DECIMAL(10,2) NOT NULL DEFAULT 0,
    pass_attempts DECIMAL(10,2) NOT NULL DEFAULT 0,
    pass_yards DECIMAL(10,2) NOT NULL DEFAULT 0,
    pass_touchdowns DECIMAL(10,2) NOT NULL DEFAULT 0,
    interceptions DECIMAL(10,2) NOT NULL DEFAULT 0,
    sacks DECIMAL(10,2) NOT NULL DEFAULT 0,
    qb_rating DECIMAL(10,2) NOT NULL DEFAULT 0,
    
    -- Rushing stats
    rush_attempts DECIMAL(10,2) NOT NULL DEFAULT 0,
    rush_yards DECIMAL(10,2) NOT NULL DEFAULT 0,
    rush_touchdowns DECIMAL(10,2) NOT NULL DEFAULT 0,
    
    -- Receiving stats
    receptions DECIMAL(10,2) NOT NULL DEFAULT 0,
    receiving_targets DECIMAL(10,2) NOT NULL DEFAULT 0,
    receiving_yards DECIMAL(10,2) NOT NULL DEFAULT 0,
    receiving_touchdowns DECIMAL(10,2) NOT NULL DEFAULT 0,
    
    -- Fumbles
    fumbles DECIMAL(10,2) NOT NULL DEFAULT 0,
    fumbles_lost DECIMAL(10,2) NOT NULL DEFAULT 0,
    
    -- Kicking stats
    field_goals_made DECIMAL(10,2) NOT NULL DEFAULT 0,
    field_goal_attempts DECIMAL(10,2) NOT NULL DEFAULT 0,
    extra_points_made DECIMAL(10,2) NOT NULL DEFAULT 0,
    extra_point_attempts DECIMAL(10,2) NOT NULL DEFAULT 0,
    
    -- Defensive stats
    total_tackles DECIMAL(10,2) NOT NULL DEFAULT 0,
    sacks_def DECIMAL(10,2) NOT NULL DEFAULT 0,
    interceptions_def DECIMAL(10,2) NOT NULL DEFAULT 0,
    fumbles_recovered DECIMAL(10,2) NOT NULL DEFAULT 0,
    defensive_touchdowns DECIMAL(10,2) NOT NULL DEFAULT 0,
    
    -- Calculated fantasy points
    fantasy_points_ppr DECIMAL(18,4) NOT NULL DEFAULT 0,
    fantasy_points_standard DECIMAL(18,4) NOT NULL DEFAULT 0,
    
    -- Audit columns
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Primary key ensures one row per athlete per game
    PRIMARY KEY (athlete_id, season, week, game_id)
);

