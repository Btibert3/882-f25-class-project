from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
import requests
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.exceptions import AirflowSkipException

# helper
def invoke_function(url, params={}) ->dict:
    """
    Invoke our cloud function url and optionally pass data for the function to use
    """
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


@dag(
    schedule="0 10 * * *",  # this is 10am, daily         
    start_date=datetime(2025, 9, 4),
    catchup=False,  # if True, when the DAG is activated, Airflow will coordinate backfills
    max_active_runs = 1,  # if backfilling, this will seralize the work 1x per backfill date/job
    tags=["raw", "ingest"]
)
def nfl_raw_pipeline():

    @task
    def schema():
        url = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/raw-schema-setup"
        resp = invoke_function(url)
        print(resp)
        return resp

    @task
    def extract(payload:dict) -> dict:
        url = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/raw-extract-scoreboard"
        ctx = get_current_context()
        # "Yesterday" as a calendar day relative to this run
        process_date = (ctx["data_interval_end"] - timedelta(days=1)).strftime("%Y%m%d")
        print(process_date)
        # add the pieces of info to the payload passed through
        payload['run_id'] = ctx["dag_run"].run_id
        payload['date'] = process_date
        print(f"[extract] process_date={process_date}, run_id={payload['run_id']}")
        resp = invoke_function(url, params=payload)
        print("response=============================")
        print(resp)
        return resp

    @task
    def load(payload:dict) -> dict:
        url = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/raw-load-scoreboard"
        ctx = get_current_context()
        print("incoming payload =======================")
        print(payload)
        payload['date'] = ctx["ds_nodash"]
        print("final payload =======================")
        print(payload)
        resp = invoke_function(url, params=payload)
        return resp
    
    @task
    def extract_game_ids(load_payload: dict) -> list[str]:
        # returns [] if key missing or empty → mapped task will be skipped
        ids = load_payload.get("game_ids") or []
        # dedupe, stringify to be safe for query params
        ids = [str(x) for x in dict.fromkeys(ids)]

        # if there are no ids, bail out.
        if not ids:
            print("No game IDs found — skipping remainder of pipeline.")
            raise AirflowSkipException("No games to process for this date")

        print(f"Found {len(ids)} game_ids: {ids}")
        return ids

    @task(retries=1, retry_delay=timedelta(seconds=30))
    def parse_load_game_detail(game_id: str) -> dict:
        url = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/raw-parse-game"
        ctx = get_current_context()
        params = {
            "game_id": game_id,
            "run_id": ctx["dag_run"].run_id,
        }
        print(params)
        resp = invoke_function(url, params=params)
        print(f"Finished game_id={game_id}")
        return resp

    schema_result = schema()
    extract_result = extract(schema_result)
    load_result = load(extract_result)
    game_ids = extract_game_ids(load_result)
    mapped_parse = parse_load_game_detail.expand(game_id=game_ids)

    # ---- Fire the next DAG and move on (no waiting) ----
    # this is also a task!
    trigger_stage = TriggerDagRunOperator(
        task_id="trigger_seed_stage",
        trigger_dag_id="nfl_seed_stage",   # <-- use the TARGET DAG'S *dag_id*, not filename
        conf={
            "source_dag_run_id": "{{ dag_run.run_id }}",
            "source_logical_date": "{{ ds_nodash }}",
        },
        wait_for_completion=False,         # fire-and-forget
        reset_dag_run=False,               # don't clear existing runs if one already exists
        trigger_rule="none_failed_min_one_success",
        # ^ ensures this only fires if at least one upstream succeeded;
        #   if we skipped due to no IDs, this won't trigger.
    )

    mapped_parse >> trigger_stage


nfl_raw_pipeline()