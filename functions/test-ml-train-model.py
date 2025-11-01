import requests
import json

def invoke_function(url, params={}) -> dict:
    """
    Invoke our cloud function url and optionally pass data for the function to use
    """
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# Test the ml-train-model function
url = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/ml-train-model"

# Test parameters
params = {
    "algorithm": "linear_regression",
    "hyperparameters": json.dumps({}),  # Empty dict for linear regression
    "run_id": "test_run_20250101_120000",
    "dataset_id": "ds-player-fantasy-2025_w08",  # Update with actual dataset_id if different
    "model_id": "model-fantasy-predictor"
}

print("Testing ml-train-model function...")
print(f"Parameters: {params}")
print("\nInvoking function...")

try:
    result = invoke_function(url, params=params)
    print("\nSuccess!")
    print(json.dumps(result, indent=2))
except requests.exceptions.HTTPError as e:
    print(f"\nError: {e}")
    if e.response is not None:
        print(f"Response: {e.response.text}")

