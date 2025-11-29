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
    def run_evaluation(dataset_info: dict) -> dict:
        """Run evaluation using LangSmith evaluate() - handles everything."""
        ls_api_key = os.environ.get("LANGSMITH_API_KEY")
        if not ls_api_key:
            raise ValueError("LANGSMITH_API_KEY environment variable not set")
        
        client = Client(api_key=ls_api_key)
        
        # Define the function that evaluate() will call
        def rag_target_function(inputs: dict) -> dict:
            """Function that evaluate() calls - invokes Cloud Function."""
            question = inputs["question"]
            try:
                response = invoke_function(CLOUD_FUNCTION_URL, {"question": question})
                return {
                    "answer": response.get("answer", ""),
                    "articles": response.get("articles", []),
                    "contexts": response.get("contexts", []),
                }
            except Exception as e:
                print(f"ERROR in rag_target_function for '{question}': {str(e)}")
                return {
                    "answer": f"ERROR: {str(e)}",
                    "articles": [],
                    "contexts": [],
                }
        
        # Define evaluators
        def precision_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
            expected_article_id = reference_outputs.get("expected_article_id")
            retrieved_articles = outputs.get("articles", [])
            precision = compute_precision_at_k(retrieved_articles, expected_article_id, K)
            return {"key": f"precision@{K}", "score": precision}
        
        def recall_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
            expected_article_id = reference_outputs.get("expected_article_id")
            total_chunks = reference_outputs.get("total_chunks", 1)
            retrieved_articles = outputs.get("articles", [])
            recall = compute_recall_at_k(retrieved_articles, expected_article_id, total_chunks, K)
            return {"key": f"recall@{K}", "score": recall}
        
        def mrr_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
            expected_article_id = reference_outputs.get("expected_article_id")
            retrieved_articles = outputs.get("articles", [])
            mrr = compute_mrr(retrieved_articles, expected_article_id)
            return {"key": "mrr", "score": mrr}
        
        def ndcg_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
            expected_article_id = reference_outputs.get("expected_article_id")
            total_chunks = reference_outputs.get("total_chunks", 1)
            contexts = outputs.get("contexts", [])
            ndcg = compute_ndcg_at_k(contexts, expected_article_id, total_chunks, K)
            return {"key": f"ndcg@{K}", "score": ndcg}
        
        # Run evaluation - this handles everything
        print(f"\nRunning evaluation on dataset '{DATASET_NAME}'...")
        
        try:
            evaluate_result = client.evaluate(
                rag_target_function,
                data=DATASET_NAME,
                evaluators=[precision_evaluator, recall_evaluator, mrr_evaluator, ndcg_evaluator],
                experiment_prefix=EXPERIMENT_NAME,
                description="RAG pipeline evaluation with Precision, Recall, MRR, and NDCG metrics",
                max_concurrency=2,
            )
            
            print(f"\nSuccessfully created experiment '{EXPERIMENT_NAME}'")
            print(f"View results in LangSmith dataset: {DATASET_NAME}")
            
            return {
                "experiment_name": EXPERIMENT_NAME,
                "status": "success",
            }
            
        except Exception as e:
            print(f"ERROR: Could not create experiment: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
    
    # Pipeline flow
    dataset_info = setup_dataset()
    evaluation_result = run_evaluation(dataset_info)


nfl_article_rag_evals()
