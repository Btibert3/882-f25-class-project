gcloud config set project btibert-ba882-fall25

echo "======================================================"
echo "build (no cache)"
echo "======================================================"

docker build --no-cache -t gcr.io/btibert-ba882-fall25/genai-chat-poc .

echo "======================================================"
echo "push"
echo "======================================================"

docker push gcr.io/btibert-ba882-fall25/streamlit-poc

echo "======================================================"
echo "deploy run"
echo "======================================================"


gcloud run deploy streamlit-poc \
    --image gcr.io/btibert-ba882-fall25/genai-chat-poc \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --memory 1Gi