#!/bin/bash

# Deploy NL Query App to Google Cloud Run

gcloud config set project btibert-ba882-fall25

echo "======================================================"
echo "Building Docker image (no cache)"
echo "======================================================"

docker build --no-cache -t gcr.io/btibert-ba882-fall25/nl-query .

echo "======================================================"
echo "Pushing to Google Container Registry"
echo "======================================================"

docker push gcr.io/btibert-ba882-fall25/nl-query

echo "======================================================"
echo "Deploying to Cloud Run"
echo "======================================================"

gcloud run deploy nl-query \
    --image gcr.io/btibert-ba882-fall25/nl-query \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --memory 2Gi \
    --timeout 300

echo "======================================================"
echo "Deployment complete!"
echo "======================================================"

