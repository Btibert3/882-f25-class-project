import functions_framework
from google.cloud import secretmanager
from google.cloud import storage
import duckdb
import pandas as pd
import json
import requests
import uuid
import json

# settings
project_id = 'btibert-ba882-fall25'
secret_id = 'MotherDuck'   #<---------- this is the name of the secret you created
version_id = 'latest'
bucket_name = 'btibert-ba882-fall25-nfl'  #<------ we created this bucket at the end of the Cloud Storage section

# db setup
db = 'nfl'
schema = "raw"
db_schema = f"{db}.{schema}"

@functions_framework.http
def task(request):

    # instantiate the services 
    sm = secretmanager.SecretManagerServiceClient()
    storage_client = storage.Client()

    # Build the resource name of the secret version
    secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version in order to get the MotherDuck token
    response = sm.access_secret_version(request={"name": secret_name})
    md_token = response.payload.data.decode("UTF-8")

    md = duckdb.connect(f'md:?motherduck_token={md_token}') 

    ##################################################### core logic: grab the game id and process a SINGLE game

    game_id = request.args.get("game_id")
    print(f"Game id: {game_id}")   #401772936

    # grab the run id
    run_id = request.args.get("run_id")
    if not run_id:
        run_id = uuid.uuid4().hex[:12]
    print(f"run_id: {run_id}")

    ingest_ts_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # get the data
    url = f"https://btibert-bu--ba882-class-project-202509-api.modal.run/event?id={game_id}"
    print(f"url: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    print(f"requested url - status {resp.status_code}")
    j = resp.json()

    ############################## write the game file to GCS

    def upload_to_gcs(bucket_name, path, run_id, data, include_run=True):
        """Uploads data to a Google Cloud Storage bucket."""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        if include_run:
            blob_name = f"{path}/run_id={run_id}/data.json"
        else:
            blob_name = f"{path}/data.json"
        blob = bucket.blob(blob_name)

        # Upload the data (here it's a serialized string)
        blob.upload_from_string(data)
        print(f"File {blob_name} uploaded to {bucket_name}.")

        return {'bucket_name':bucket_name, 'blob_name': blob_name}
    
    season = j.get('header').get('season').get('year', 2025)
    week = j.get('header').get('week')
    j_string = json.dumps(j)
    _path = f"raw/game_detail/season={season}/week={week}/game_id={game_id}"
    gcs_path = upload_to_gcs(bucket_name, path=_path, run_id=run_id, data=j_string)
    print(type(gcs_path))
    print(gcs_path)
    print(f"wrote API response to: {_path}")


    ############################## for this game, parse out information (assumes game is over, but we can control that in our orchestration!)

    ## article
    article = j.get("article", None)
    article_df = None 
    article_imgdf = None
    adf = None
    if article:
        adf = pd.json_normalize(article)
        article_cols = ['id', 'headline', 'published', 'source', 'story']
        article_df = adf[article_cols].copy()
        article_df['game_id'] = game_id
        article_df['ingest_timestamp'] = ingest_ts_str
        article_df["published"] = pd.to_datetime(article_df["published"], utc=True)
        article_df['source_path'] = gcs_path.get('blob_name')
        article_df['run_id'] = run_id
        print("finished parsing the article")

    ## article images
    if adf is not None:
        tmp = adf[["images"]].explode("images", ignore_index=True)
        article_imgdf = pd.json_normalize(tmp["images"])
        article_imgdf['game_id'] = game_id
        article_imgdf['article_id'] = article['id']
        article_imgdf['ingest_timestamp'] = ingest_ts_str
        article_imgdf['source_path'] = gcs_path.get('blob_name')
        article_imgdf['run_id'] = run_id
        print("finished parsing the images")
   
    # team statistics
    team_stats = pd.json_normalize(
        j,
        record_path=["boxscore", "teams", "statistics"],
        meta=[
            ["boxscore", "teams", "team", "id"],
            ["boxscore", "teams", "team", "abbreviation"],
            ["boxscore", "teams", "homeAway"],
        ],
        errors="ignore"
    ).rename(columns={
        "boxscore.teams.team.id": "team_id",
        "boxscore.teams.team.abbreviation": "team_abbr",
        "boxscore.teams.homeAway": "home_away"
    })
    team_stats['game_id'] = game_id
    team_stats['ingest_timestamp'] = ingest_ts_str
    team_stats['source_path'] = gcs_path.get('blob_name')
    team_stats['run_id'] = run_id
    print("finished parsing the team stats")

    # player stats
    rows = []
    for team_block in j["boxscore"]["players"]:
        team_id = team_block["team"]["id"]
        team_abbr = team_block["team"]["abbreviation"]

        for cat in team_block["statistics"]:
            cat_name = cat["name"]              # e.g., "passing"
            keys = cat.get("keys", [])          # e.g., ["completions/passingAttempts", "passingYards", ...]
            labels = cat.get("labels", [])      # human-friendly labels
            for ath in cat.get("athletes", []):
                athlete = ath["athlete"]
                athlete_id = athlete["id"]
                athlete_name = athlete.get("displayName")
                values = ath.get("stats", [])

                for k, v, lbl in zip(keys, values, labels[:len(keys)]):
                    rows.append({
                        "game_id": game_id,
                        "team_id": team_id,
                        "team_abbr": team_abbr,
                        "athlete_id": athlete_id,
                        "athlete_name": athlete_name,
                        "category": cat_name,       # "passing", "rushing", etc.
                        "stat_key": k,              # granular key, e.g., "passingYards" or "completions/passingAttempts"
                        "stat_label": lbl,          # e.g., "YDS", "C/ATT"
                        "value_str": v,
                    })
    player_stats_df = pd.DataFrame(rows)
    player_stats_df['ingest_timestamp'] = ingest_ts_str
    player_stats_df['source_path'] = gcs_path.get('blob_name')
    player_stats_df['run_id'] = run_id
    print("finished parsing the player stats")


    # drives and plays
    # def spot_text(node: dict | None) -> str | None:
    #     """Return a readable spot string from a start/end node.
    #     Prefers: possessionText > downDistanceText > text."""
    #     if not node:
    #         return None
    #     for k in ("possessionText", "downDistanceText", "text"):
    #         v = node.get(k)
    #         if v:
    #             return v
    #     return None

    # dr_rows = []
    # pl_rows = []

    # for d in j.get("drives", {}).get("previous", []):
    #     d_start = d.get("start") or {}
    #     d_end = d.get("end") or {}

    #     dr_rows.append({
    #         "game_id": game_id,
    #         "drive_id": d.get("id"),
    #         "team_id": (d.get("team") or {}).get("id"),
    #         "team_abbr": (d.get("team") or {}).get("abbreviation"),
    #         "description": d.get("description"),
    #         "result": d.get("result"),
    #         "display_result": d.get("displayResult"),
    #         "short_result": d.get("shortDisplayResult"),
    #         "yards": d.get("yards"),
    #         "offensive_plays": d.get("offensivePlays"),
    #         "is_score": d.get("isScore"),
    #         "start_period": (d_start.get("period") or {}).get("number"),
    #         "start_clock": (d_start.get("clock") or {}).get("displayValue"),
    #         "start_spot": spot_text(d_start),   # <-- FIX
    #         "end_period": (d_end.get("period") or {}).get("number"),
    #         "end_clock": (d_end.get("clock") or {}).get("displayValue"),
    #         "end_spot": spot_text(d_end),       # <-- FIX
    #         "time_elapsed": d.get("timeElapsed"),
    #     })

    # # 4B) Plays (one row per play)
    # for p in d.get("plays", []):
    #     p_start = p.get("start") or {}
    #     p_end = p.get("end") or {}

    #     pl_rows.append({
    #         "game_id": game_id,
    #         "drive_id": d.get("id"),
    #         "sequence": int(p.get("sequenceNumber")),
    #         "play_id": p.get("id"),
    #         "period": (p.get("period") or {}).get("number"),
    #         "clock": (p.get("clock") or {}).get("displayValue"),
    #         "text": p.get("text"),
    #         "stat_yardage": p.get("statYardage"),
    #         "scoring_play": p.get("scoringPlay"),
    #         "start_spot": spot_text(p_start),  # <-- FIX
    #         "end_spot": spot_text(p_end),      # <-- FIX
    #         "home_score": p.get("homeScore"),
    #         "away_score": p.get("awayScore"),
    #         "wallclock": p.get("wallclock"),
    #         "type_id": (p.get("type") or {}).get("id"),
    #         "type_text": (p.get("type") or {}).get("text"),
    #     })

    # drives_df = pd.DataFrame(dr_rows)
    # plays_df = pd.DataFrame(pl_rows)


    ############################################## write files to GCS

    # holy moly variable overload
    gcs_path = "gs://" + gcs_path.get('bucket_name') + "/raw"

    # articles
    if article_df is not None and not article_df.empty:
        # gcs_path = "gs://btibert-ba882-fall25-nfl/raw"
        full_path = gcs_path + f"/game_detail/season={season}/week={week}/game_id={game_id}/articles/data.parquet"
        print(full_path)
        article_df.to_parquet(full_path, index=False)
        full_path_run = gcs_path + f"/game_detail/season={season}/week={week}/game_id={game_id}/run_id={run_id}/articles/data.parquet"
        print(full_path_run)
        article_df.to_parquet(full_path_run, index=False)

    # article iamges
    if article_imgdf is not None and not article_imgdf.empty:
        full_path = gcs_path + f"/game_detail/season={season}/week={week}/game_id={game_id}/article_images/data.parquet"
        print(full_path)
        article_imgdf.to_parquet(full_path, index=False)
        full_path_run = gcs_path + f"/game_detail/season={season}/week={week}/game_id={game_id}/run_id={run_id}/article_images/data.parquet"
        print(full_path_run)
        article_imgdf.to_parquet(full_path_run, index=False)

    # team stats
    full_path = gcs_path + f"/game_detail/season={season}/week={week}/game_id={game_id}/team_stats/data.parquet"
    print(full_path)
    team_stats['value'] = team_stats["value"].astype("string")  # avoid implicit conversion
    team_stats.to_parquet(full_path, index=False)
    full_path_run = gcs_path + f"/game_detail/season={season}/week={week}/game_id={game_id}/run_id={run_id}/team_stats/data.parquet"
    print(full_path_run)
    team_stats.to_parquet(full_path_run, index=False)
    print("Completed team stats")

    # player stats
    full_path = gcs_path + f"/game_detail/season={season}/week={week}/game_id={game_id}/player_stats/data.parquet"
    print(full_path)
    player_stats_df.to_parquet(full_path, index=False)
    full_path_run = gcs_path + f"/game_detail/season={season}/week={week}/game_id={game_id}/run_id={run_id}/player_stats/data.parquet"
    print(full_path_run)
    player_stats_df.to_parquet(full_path_run, index=False)
    print("Completed player stats")

    # insert/append into the raw tables in the warehouse (MotherDuck)
    if article_df is not None and not article_df.empty:
        print(f"appending rows to raw, articles")
        tbl = db_schema + ".articles"
        md.execute(f"INSERT INTO {tbl} SELECT * FROM article_df")

    if article_imgdf is not None and not article_imgdf.empty:
        # more defensive, a clear example where modular is better (i.e. each table is a task, not bundled together)
        # below is to guard against some instances where there is an entry but no valid data, check for a url
        if 'url' in article_imgdf.columns:   
            print(f"appending rows to raw, article images")
            tbl = db_schema + ".article_images"
            md.execute(f"INSERT INTO {tbl} SELECT * FROM article_imgdf")

    print(f"appending rows to raw, team stats")
    tbl = db_schema + ".team_stats"
    md.execute(f"INSERT INTO {tbl} SELECT * FROM team_stats")

    print(f"appending rows to raw, player stats")
    tbl = db_schema + ".player_stats"
    md.execute(f"INSERT INTO {tbl} SELECT * FROM player_stats_df")


    return {}, 200

    

