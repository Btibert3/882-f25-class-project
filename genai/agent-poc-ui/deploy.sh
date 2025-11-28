gcloud config set project btibert-ba882-fall25

echo "======================================================"
echo "build (no cache)"
echo "======================================================"

docker build --no-cache -t gcr.io/btibert-ba882-fall25/agent-poc-ui .

echo "======================================================"
echo "push"
echo "======================================================"

docker push gcr.io/btibert-ba882-fall25/agent-poc-ui

echo "======================================================"
echo "deploy run"
echo "======================================================"


gcloud run deploy agent-poc-ui \
    --image gcr.io/btibert-ba882-fall25/agent-poc-ui \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --memory 1Gi