# this function will parse a game id to get the play by play and append into raw

import functions_framework
from google.cloud import secretmanager
import duckdb
import pandas as pd

# settings
project_id = 'btibert-ba882-fall25'
secret_id = 'MotherDuck'   #<---------- this is the name of the secret you created
version_id = 'latest'

# db setup
db = 'nfl'
schema = "raw"
db_schema = f"{db}.{schema}"
tbl_name = "pbp"



@functions_framework.http
def task(request):

    # instantiate the services 
    sm = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version in order to get the MotherDuck token
    response = sm.access_secret_version(request={"name": name})
    md_token = response.payload.data.decode("UTF-8")



    ##################################################### take the input
    ## this expects to be posted a dictionary of k/v pairs, where the dictionary is a single record
    _game = request.get_json(silent=True) or {}

    print("=== Received article payload ===")
    print(article)
    print("================================")
    print(article.keys())

    gid = _game.get('gid', '401772936')



    # metadata could/should be added here
    return {}, 200
