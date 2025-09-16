# this function will setup the raw schema in Motherduck

import functions_framework
from google.cloud import secretmanager
import duckdb

# settings
project_id = 'btibert-ba882-fall25'
secret_id = 'MotherDuck'   #<---------- this is the name of the secret you created
version_id = 'latest'

# db setup
db = 'nfl'
schema = "raw"
db_schema = f"{db}.{schema}"

@functions_framework.http
def task(request):

    # instantiate the services 
    sm = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version in order to get the MotherDuck token
    response = sm.access_secret_version(request={"name": name})
    md_token = response.payload.data.decode("UTF-8")

    # initiate the MotherDuck connection through an access token through
    # this syntax lets us connect to our motherduck cloud warehouse and execute commands via the duckdb library

    md = duckdb.connect(f'md:?motherduck_token={md_token}') 

    ##################################################### create the database and schema

    # define the DDL statement with an f string
    create_db_sql = f"CREATE DATABASE IF NOT EXISTS {db};"   

    # execute the command to create the database
    md.sql(create_db_sql)

    # confirm the db exists and add to our logging on GCP!
    print(md.sql("SHOW DATABASES").show())

    # create the schema
    md.sql(f"CREATE SCHEMA IF NOT EXISTS {db_schema};") 

    ##################################################### create the core tables in the raw schema

    # venue
    raw_tbl_name = f"{db_schema}.venues"
    raw_tbl_sql = f"""
    CREATE TABLE IF NOT EXISTS {raw_tbl_name} (
        id INT
        ,fullname VARCHAR
        ,city VARCHAR
        ,state VARCHAR
        ,country VARCHAR
        ,indoor BOOLEAN
        ,ingest_timestamp TIMESTAMP
        ,source_path VARCHAR
        ,run_id VARCHAR
    );
    """
    print(f"{raw_tbl_sql}")
    md.sql(raw_tbl_sql)

    # game
    raw_tbl_name = f"{db_schema}.games"
    raw_tbl_sql = f"""
    CREATE TABLE IF NOT EXISTS {raw_tbl_name} (
        id INT 
        ,start_date TIMESTAMP
        ,season INT
        ,week INT
        ,venue_id INT
        ,attendance INT
        ,ingest_timestamp TIMESTAMP
        ,source_path VARCHAR
        ,run_id VARCHAR
    );
    """
    print(f"{raw_tbl_sql}")
    md.sql(raw_tbl_sql)

    # teams
    raw_tbl_name = f"{db_schema}.teams"
    raw_tbl_sql = f"""
    CREATE TABLE IF NOT EXISTS {raw_tbl_name} (
        id INT 
        ,name VARCHAR
        ,abbrev VARCHAR
        ,display_name VARCHAR
        ,short_name VARCHAR
        ,color VARCHAR
        ,alternate_color VARCHAR
        ,venue_id INT
        ,logo VARCHAR
        ,ingest_timestamp TIMESTAMP
        ,source_path VARCHAR
        ,run_id VARCHAR
    );
    """
    print(f"{raw_tbl_sql}")
    md.sql(raw_tbl_sql)  

    # game_team
    raw_tbl_name = f"{db_schema}.game_team"
    raw_tbl_sql = f"""
    CREATE TABLE IF NOT EXISTS {raw_tbl_name} (
        game_id INT 
        ,team_id INT
        ,home_away VARCHAR
        ,score INT
        ,ingest_timestamp TIMESTAMP
        ,source_path VARCHAR
        ,run_id VARCHAR
    );
    """
    print(f"{raw_tbl_sql}")
    md.sql(raw_tbl_sql)  


    # return a dictionary/json entry, its blank because are not returning data, 200 for success
    return {}, 200