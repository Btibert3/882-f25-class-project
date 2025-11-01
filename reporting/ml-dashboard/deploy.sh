#!/bin/bash

# Deploy ML Dashboard to Google Cloud Run
# This script builds and deploys the Streamlit app as a containerized service

gcloud config set project btibert-ba882-fall25

echo "======================================================"
echo "Building Docker image (no cache)"
echo "======================================================"

docker build --no-cache -t gcr.io/btibert-ba882-fall25/ml-dashboard .

echo "======================================================"
echo "Pushing to Google Container Registry"
echo "======================================================"

docker push gcr.io/btibert-ba882-fall25/ml-dashboard

echo "======================================================"
echo "Deploying to Cloud Run"
echo "======================================================"

gcloud run deploy ml-dashboard \
    --image gcr.io/btibert-ba882-fall25/ml-dashboard \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --memory 2Gi \
    --timeout 300

echo "======================================================"
echo "Deployment complete!"
echo "======================================================"

