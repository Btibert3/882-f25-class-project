"""
RAG Pipeline Evaluation DAG using LangSmith.

Evaluates retrieval performance using Precision@25, Recall@25, MRR, and NDCG@25.
Creates LangSmith dataset, maps over examples, and registers experiment results.
"""

from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from pathlib import Path
import pandas as pd
import json
import math
import os
import requests
from langsmith import Client

# Configuration
BASE_DIR = Path(os.environ.get("AIRFLOW_HOME", "/usr/local/airflow"))
CSV_FILE = BASE_DIR / "include" / "eval-dataset.csv"  # Place CSV here
CLOUD_FUNCTION_URL = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/langgraph-rag"
DATASET_NAME = "ba882-nfl-article-rag"
EXPERIMENT_NAME = "baseline-25"
K = 25  # top-k retrieval

# Helper to invoke Cloud Function
def invoke_function(url: str, json_data: dict) -> dict:
    """Invoke Cloud Function with POST request."""
    resp = requests.post(url, json=json_data, headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    return resp.json()


def compute_precision_at_k(retrieved_article_ids: list, expected_article_id: int, k: int) -> float:
    """Compute Precision@k: relevant retrieved / k."""
    relevant = sum(1 for aid in retrieved_article_ids[:k] if aid == expected_article_id)
    return relevant / k


def compute_recall_at_k(retrieved_article_ids: list, expected_article_id: int, total_relevant_chunks: int, k: int) -> float:
    """Compute Recall@k: relevant retrieved / total relevant available."""
    relevant_retrieved = sum(1 for aid in retrieved_article_ids[:k] if aid == expected_article_id)
    return relevant_retrieved / total_relevant_chunks if total_relevant_chunks > 0 else 0.0


def compute_mrr(retrieved_article_ids: list, expected_article_id: int) -> float:
    """Compute Mean Reciprocal Rank: 1 / (position + 1) of first relevant item."""
    for i, aid in enumerate(retrieved_article_ids):
        if aid == expected_article_id:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(contexts: list, expected_article_id: int, total_relevant_chunks: int, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain@k.
    Uses binary relevance (1 if article_id matches, 0 otherwise).
    """
    # Compute actual DCG
    dcg = 0.0
    for i, ctx in enumerate(contexts[:k]):
        relevance = 1.0 if ctx.get("article_id") == expected_article_id else 0.0
        dcg += relevance / math.log2(i + 2)
    
    # Compute ideal DCG (all relevant chunks at top)
    ideal_dcg = 0.0
    num_relevant_to_consider = min(total_relevant_chunks, k)
    
    for i in range(num_relevant_to_consider):
        ideal_dcg += 1.0 / math.log2(i + 2)
    
    if ideal_dcg == 0.0:
        return 0.0
    
    return dcg / ideal_dcg


@dag(
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["genai", "rag", "evaluation", "langsmith"],
)
def nfl_article_rag_evals():
    
    @task
    def setup_dataset():
        """Create/verify LangSmith dataset from CSV file."""
        ls_api_key = os.environ.get("LANGSMITH_API_KEY")
        if not ls_api_key:
            raise ValueError("LANGSMITH_API_KEY environment variable not set")
        
        client = Client(api_key=ls_api_key)
        
        # Check if dataset exists
        try:
            dataset = client.read_dataset(dataset_name=DATASET_NAME)
            print(f"Dataset '{DATASET_NAME}' already exists (ID: {dataset.id})")
            return {"dataset_id": str(dataset.id), "exists": True}  # Convert UUID to string
        except Exception:
            print(f"Dataset '{DATASET_NAME}' does not exist, creating it...")
        
        # Read CSV and create dataset
        if not CSV_FILE.exists():
            raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
        
        # Read CSV with pandas
        df = pd.read_csv(CSV_FILE)
        print(f"Loaded {len(df)} examples from CSV")
        
        # Convert to list of dicts
        examples = []
        for _, row in df.iterrows():
            examples.append({
                "question": row["question"],
                "expected_article_id": int(row["article_id"]),
                "total_chunks": int(row["n_chunks"])
            })
        
        # Create dataset
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="NFL article RAG evaluation dataset with test questions"
        )
        
        # Add examples to dataset
        for example in examples:
            client.create_example(
                dataset_id=dataset.id,
                inputs={"question": example["question"]},
                outputs={
                    "expected_article_id": example["expected_article_id"],
                    "total_chunks": example["total_chunks"]
                },
                metadata={
                    "question": example["question"],
                    "expected_article_id": example["expected_article_id"],
                    "total_chunks": example["total_chunks"]
                }
            )
        
        print(f"Created dataset '{DATASET_NAME}' (ID: {dataset.id}) with {len(examples)} examples")
        return {"dataset_id": str(dataset.id), "exists": False}  # Convert UUID to string
    
    @task
    def load_evaluation_examples(dataset_info: dict):
        """Load examples from LangSmith dataset."""
        ls_api_key = os.environ.get("LANGSMITH_API_KEY")
        if not ls_api_key:
            raise ValueError("LANGSMITH_API_KEY environment variable not set")
        
        client = Client(api_key=ls_api_key)
        dataset_id = dataset_info["dataset_id"]
        
        # Fetch examples from dataset
        examples = list(client.list_examples(dataset_id=dataset_id))
        
        # Format for downstream tasks
        formatted_examples = []
        for ex in examples:
            formatted_examples.append({
                "dataset_id": dataset_id,
                "example_id": str(ex.id),
                "question": ex.inputs["question"],
                "expected_article_id": ex.outputs["expected_article_id"],
                "total_chunks": ex.outputs["total_chunks"]
            })
        
        print(f"Loaded {len(formatted_examples)} examples from LangSmith dataset")
        return formatted_examples
    
    @task(retries=1, retry_delay=timedelta(seconds=30))
    def evaluate_example(example: dict) -> dict:
        """Evaluate a single example: call RAG function and compute metrics."""
        question = example["question"]
        expected_article_id = example["expected_article_id"]
        total_chunks = example["total_chunks"]
        
        print(f"Evaluating: {question}")
        print(f"  Expected article_id: {expected_article_id}, Total chunks: {total_chunks}")
        
        # Call RAG Cloud Function
        response = invoke_function(CLOUD_FUNCTION_URL, {"question": question})
        
        # Extract retrieved contexts and article_ids
        contexts = response.get("contexts", [])
        retrieved_article_ids = [ctx.get("article_id") for ctx in contexts if ctx.get("article_id") is not None]
        
        print(f"  Retrieved {len(retrieved_article_ids)} chunks (top {K} considered)")
        
        # Compute metrics
        precision = compute_precision_at_k(retrieved_article_ids, expected_article_id, K)
        recall = compute_recall_at_k(retrieved_article_ids, expected_article_id, total_chunks, K)
        mrr = compute_mrr(retrieved_article_ids, expected_article_id)
        ndcg = compute_ndcg_at_k(contexts, expected_article_id, total_chunks, K)
        
        # Count relevant chunks retrieved
        relevant_retrieved = sum(1 for aid in retrieved_article_ids[:K] if aid == expected_article_id)
        
        print(f"  Precision@{K}: {precision:.4f} ({relevant_retrieved}/{K} relevant)")
        print(f"  Recall@{K}: {recall:.4f} ({relevant_retrieved}/{total_chunks} relevant)")
        print(f"  MRR: {mrr:.4f}")
        print(f"  NDCG@{K}: {ndcg:.4f}")
        
        return {
            "dataset_id": example["dataset_id"],
            "example_id": example["example_id"],
            "question": question,
            "expected_article_id": expected_article_id,
            "total_chunks": total_chunks,
            "metrics": {
                f"precision@{K}": precision,
                f"recall@{K}": recall,
                "mrr": mrr,
                f"ndcg@{K}": ndcg,
            },
            "retrieved_count": len(retrieved_article_ids),
            "relevant_retrieved": relevant_retrieved,
            "answer": response.get("answer", ""),
        }
    
    @task
    def aggregate_results(results: list) -> dict:
        """Aggregate metrics and register experiment in LangSmith."""
        ls_api_key = os.environ.get("LANGSMITH_API_KEY")
        if not ls_api_key:
            raise ValueError("LANGSMITH_API_KEY environment variable not set")
        
        client = Client(api_key=ls_api_key)
        ctx = get_current_context()
        
        # Compute aggregate metrics
        valid_results = [r for r in results if "metrics" in r]
        if not valid_results:
            raise ValueError("No valid evaluation results to aggregate")
        
        avg_precision = sum(r["metrics"][f"precision@{K}"] for r in valid_results) / len(valid_results)
        avg_recall = sum(r["metrics"][f"recall@{K}"] for r in valid_results) / len(valid_results)
        avg_mrr = sum(r["metrics"]["mrr"] for r in valid_results) / len(valid_results)
        avg_ndcg = sum(r["metrics"][f"ndcg@{K}"] for r in valid_results) / len(valid_results)
        
        aggregate_metrics = {
            f"avg_precision@{K}": avg_precision,
            f"avg_recall@{K}": avg_recall,
            "avg_mrr": avg_mrr,
            f"avg_ndcg@{K}": avg_ndcg,
        }
        
        print("=" * 80)
        print("AGGREGATE METRICS")
        print("=" * 80)
        for metric, value in aggregate_metrics.items():
            print(f"{metric}: {value:.4f}")
        print("=" * 80)
        
        # Register experiment runs in LangSmith
        dag_run_id = ctx["dag_run"].run_id
        timestamp = datetime.now().isoformat()
        dataset_id = valid_results[0]["dataset_id"]  # Get dataset_id from first result
        
        print(f"\nRegistering {len(valid_results)} runs to LangSmith experiment '{EXPERIMENT_NAME}'...")
        
        registered_count = 0
        for result in valid_results:
            try:
                # Create run for each example evaluation
                # Note: create_run may return None on success, so we check for exceptions only
                client.create_run(
                    name=f"{EXPERIMENT_NAME}-{dag_run_id}",
                    run_type="chain",
                    inputs={"question": result["question"]},
                    outputs={
                        "answer": result.get("answer", ""),
                        "expected_article_id": result["expected_article_id"],
                    },
                    extra={
                        "dag_run_id": dag_run_id,
                        "timestamp": timestamp,
                        "example_id": result["example_id"],
                        **result["metrics"],
                        "relevant_retrieved": result["relevant_retrieved"],
                        "total_chunks": result["total_chunks"],
                    },
                    dataset_id=dataset_id,
                    project_name=EXPERIMENT_NAME,
                )
                registered_count += 1
                print(f"  Registered run for example {result['example_id']}")
            except Exception as e:
                print(f"ERROR: Could not register run for example {result['example_id']}: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        print(f"\nSuccessfully registered {registered_count}/{len(valid_results)} experiment runs")
        print(f"\nView results in LangSmith:")
        print(f"  Dataset: {DATASET_NAME}")
        print(f"  Experiment/Project: {EXPERIMENT_NAME}")
        
        return {
            "aggregate_metrics": aggregate_metrics,
            "num_examples": len(valid_results),
            "dag_run_id": dag_run_id,
        }
    
    # Pipeline flow
    dataset_info = setup_dataset()
    examples = load_evaluation_examples(dataset_info)
    evaluation_results = evaluate_example.expand(example=examples)
    final_results = aggregate_results(evaluation_results)


nfl_article_rag_evals()
