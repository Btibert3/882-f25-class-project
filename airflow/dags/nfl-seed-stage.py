# imports
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import requests

# helper
def invoke_function(url, params={}) ->dict:
    """
    Invoke our cloud function url and optionally pass data for the function to use
    """
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

@dag(
    schedule=None,                 
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["stage"]
)
def nfl_seed_stage():

    @task
    def schema():
        url = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/stage-schema"
        resp = invoke_function(url)
        print(resp)
        return resp
    
    @task
    def load():
        url = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/stage-load-tables"
        resp = invoke_function(url)
        print(resp)
        return resp   

    # once this completes, fire the next dag to update the gold layer
    trigger_stage = TriggerDagRunOperator(
        task_id="trigger_gold",
        trigger_dag_id="gold",   # <-- defined in nfl-gold.py
        conf={
            "source_dag_run_id": "{{ dag_run.run_id }}",
            "source_logical_date": "{{ ds_nodash }}",
        },
        wait_for_completion=False,         # fire-and-forget
        reset_dag_run=False,               # don't clear existing runs if one already exists
        trigger_rule="none_failed_min_one_success",
    )

    # wire in - week 12
    trigger_articles = TriggerDagRunOperator(
        task_id="trigger_nfl_article_processing",
        trigger_dag_id="nfl_article_processing",  
        conf={
            "source_dag_run_id": "{{ dag_run.run_id }}",
            "source_logical_date": "{{ ds_nodash }}",
        },
        wait_for_completion=False,
        reset_dag_run=False,
        trigger_rule="none_failed_min_one_success",
    )
    
    schema() >> load() >> [trigger_stage, trigger_articles]




nfl_seed_stage()