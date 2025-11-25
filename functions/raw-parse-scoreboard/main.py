import functions_framework
from google.cloud import secretmanager
from google.cloud import storage
import duckdb
import pandas as pd
import json

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

    # Parse the request data
    # request_json = request.get_json(silent=True)
    # print(f"request: {json.dumps(request_json)}")
    # # we should be passing in a payload, but for rapid testing, ensure there is a dictionary
    # if request_json is None:
    #     request_json = {'num_entries':1} # force eval on hardcoded test data

    # instantiate the services 
    sm = secretmanager.SecretManagerServiceClient()
    storage_client = storage.Client()

    # Build the resource name of the secret version
    secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version in order to get the MotherDuck token
    response = sm.access_secret_version(request={"name": secret_name})
    md_token = response.payload.data.decode("UTF-8")

    md = duckdb.connect(f'md:?motherduck_token={md_token}') 

    ##################################################### core logic: read json from gcs, parse, and insert/store

    # if there aren't any records to process, exit
    num_entries = request.args.get("num_entries")
    print(f"num_entries = {num_entries}")
    if int(num_entries) == 0:
        print("no entries found for the date evaluated downsream - EXITING")
        return {}, 200

    
    # read in the file from gcs that was created as part of this "run"
    bucket_name = request.args.get("bucket_name", "btibert-ba882-fall25-nfl")
    bucket = storage_client.bucket(bucket_name)
    blob_name = request.args.get("blob_name", "raw/scoreboard/season=2025/week=2/9b8ef48f3092/data.json")
    blob = bucket.blob(blob_name)
    data_str = blob.download_as_text()
    j = json.loads(data_str)
    print(f"number of entries parsed: {len(j.get('events'))} ==========")

    # parse out the events -> these are "games", which are a list
    events = j.get('events')

    # isolate data elements
    venues = []
    games = []
    _team = []
    gameteam = []
    ingest_ts_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # walk the data - this 100% could be improved?  how? what would you do differently?
    event_ids = [] 
    for e in events:
        game_id = e.get('id')
        event_ids.append(game_id)
        start_date = pd.to_datetime(e.get("date"), utc=True).tz_localize(None)
        season = e['season']['year']
        week = e['week']['number']
        venue = e['competitions'][0]['venue']
        teams = e['competitions'][0]['competitors']
        attendance = e['competitions'][0]['attendance']
        source_path = bucket_name + blob_name
        run_id = request.args.get('run_id')
        
        # append the info for games and venues
        games.append(
            {
                'game_id': game_id,
                'start_date': start_date,
                'season': season,
                'week': week,
                'venue_id': venue['id'],
                'attendance': attendance,
                'ingest_timestamp': ingest_ts_str,
                'source_path': source_path,
                'run_id': run_id
            }
        )

        venues.append(
            {
                'id': venue['id'],
                'fullname': venue['fullName'],
                'city': venue['address']['city'],
                'state': venue.get('address', {}).get('state', None), #errors on intl -> 'state': venue['address']['state'],
                'country': venue['address']['country'],
                'indoor': venue['indoor'],
                'ingest_timestamp': ingest_ts_str,
                'source_path': source_path,
                'run_id': run_id
            }
        )

        # parse out teams by using json to flatten out the data
        teams_parsed  = pd.json_normalize(teams)

        teams_df = teams_parsed.copy()
        team_cols = ['id', 
                        'team.name', 
                        'team.abbreviation',
                        'team.displayName',
                        'team.shortDisplayName',
                        'team.color',
                        'team.alternateColor',
                        'team.venue.id',
                        'team.logo']
        teams_df = teams_df[team_cols]
        rename_mapper = {
            'team.name':'name',
            'team.displayName': 'display_name',
            'team.shortDisplayName': 'short_name',
            'team.color': 'color',
            'team.alternateColor': 'alternate_color',
            'team.venue.id': 'venue_id',
            'team.logo':'logo'
        }
        teams_df = teams_df.rename(columns=rename_mapper)
        teams_df['ingest_timestamp'] = ingest_ts_str
        teams_df['source_path'] = source_path
        teams_df['run_id'] = run_id
        _team.append(teams_df)

        # the data are mostly there for game team
        game_team = teams_parsed.copy()
        game_team = game_team[['id', 'homeAway', 'score']]
        game_team.insert(0, 'game_id', game_id) # insert the game id as the first column
        game_team = game_team.rename(columns = {
            'homeAway': 'home_away',
            'id': 'team_id'
        })
        game_team['ingest_timestamp'] = ingest_ts_str
        game_team['source_path'] = source_path
        game_team['run_id'] = run_id
        gameteam.append(game_team)
    
    # with the data parsed, put into dataframes across the full dataset
    games_df = pd.DataFrame(games)
    games_df['ingest_timestamp'] = pd.to_datetime(games_df['ingest_timestamp'], errors="coerce")
    venues_df = pd.DataFrame(venues)
    venues_df['ingest_timestamp'] = pd.to_datetime(venues_df['ingest_timestamp'], errors="coerce")
    teams_df = pd.concat(_team)
    teams_df = teams_df.dropna(subset="id") # a bug/parse issue crept in
    teams_df['ingest_timestamp'] = pd.to_datetime(teams_df['ingest_timestamp'], errors="coerce")
    gt_df = pd.concat(gameteam)
    gt_df = gt_df.dropna(subset="team_id")
    gt_df['ingest_timestamp'] = pd.to_datetime(gt_df['ingest_timestamp'], errors="coerce")


    # log
    print(f"length of games: {len(games_df)} =======")
    print(f"length of venues: {len(venues_df)} =======")
    print(f"length of teams: {len(teams_df)} =======")
    print(f"length of game_teams: {len(gt_df)} =======")

    # write to games gcs - both at the core and within the partition of the run id
    gcs_path = "gs://btibert-ba882-fall25-nfl/raw"
    full_path = gcs_path + f"/games/season={season}/week={week}/data.parquet"
    games_df.to_parquet(full_path, index=False)
    full_path_run = gcs_path + f"/games/season={season}/week={week}/run_id={run_id}/data.parquet"
    games_df.to_parquet(full_path_run, index=False)

    # venues 
    full_path = gcs_path + f"/venues/season={season}/week={week}/data.parquet"
    venues_df.to_parquet(full_path, index=False)
    full_path_run = gcs_path + f"/venues/season={season}/week={week}/run_id={run_id}/data.parquet"
    venues_df.to_parquet(full_path_run, index=False)

    # teams
    full_path = gcs_path + f"/teams/season={season}/week={week}/data.parquet"
    teams_df.to_parquet(full_path, index=False)
    full_path_run = gcs_path + f"/teams/season={season}/week={week}/run_id={run_id}/data.parquet"
    teams_df.to_parquet(full_path_run, index=False)

    # game_team
    full_path = gcs_path + f"/game_team/season={season}/week={week}/data.parquet"
    gt_df.to_parquet(full_path, index=False)
    full_path_run = gcs_path + f"/game_team/season={season}/week={week}/run_id={run_id}/data.parquet"
    gt_df.to_parquet(full_path_run, index=False)

    # insert/append into the raw tables in the warehouse (MotherDuck)
    print(f"appending rows to raw/games")
    tbl = db_schema + ".games"
    md.execute(f"INSERT INTO {tbl} SELECT * FROM games_df")

    print(f"appending rows to raw/venues")
    tbl = db_schema + ".venues"
    md.execute(f"INSERT INTO {tbl} SELECT * FROM venues_df")

    print(f"appending rows to raw/teams")
    tbl = db_schema + ".teams"
    md.execute(f"INSERT INTO {tbl} SELECT * FROM teams_df")

    print(f"appending rows to raw/game teams")
    tbl = db_schema + ".game_team"
    md.execute(f"INSERT INTO {tbl} SELECT * FROM gt_df")
    

    return {"game_ids": event_ids}, 200

