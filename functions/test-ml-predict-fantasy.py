import requests
import json

def invoke_function(url, params={}) -> dict:
    """
    Invoke our cloud function url and optionally pass data for the function to use
    """
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# Test the ml-predict-fantasy function
url = "https://us-central1-btibert-ba882-fall25.cloudfunctions.net/ml-predict-fantasy"

# Test parameters - get all predictions (no filters)
print("Testing ml-predict-fantasy function (no filters)...")
print("\nInvoking function...")

try:
    result = invoke_function(url, params={})
    print("\nSuccess!")
    print(f"Model version: {result.get('model_version_id')}")
    print(f"Model type: {result.get('model_type')}")
    print(f"Count: {result.get('count')}")
    print(f"\nFirst 3 predictions:")
    for pred in result.get('predictions', [])[:3]:
        print(json.dumps(pred, indent=2, default=str))
except requests.exceptions.HTTPError as e:
    print(f"\nError: {e}")
    if e.response is not None:
        print(f"Response: {e.response.text}")

# Test with filters (using Drake Maye)
print("\n" + "="*60)
print("Testing with filters (athlete_id=4431452, Drake Maye)...")
print("="*60)

try:
    result = invoke_function(url, params={
        "athlete_id": "4431452",  # Drake Maye
    })
    print("\nSuccess!")
    print(f"Model version: {result.get('model_version_id')}")
    print(f"Model type: {result.get('model_type')}")
    print(f"Count: {result.get('count')}")
    print(f"\nPredictions:")
    for pred in result.get('predictions', []):
        print(json.dumps(pred, indent=2, default=str))
except requests.exceptions.HTTPError as e:
    print(f"\nError: {e}")
    if e.response is not None:
        print(f"Response: {e.response.text}")

