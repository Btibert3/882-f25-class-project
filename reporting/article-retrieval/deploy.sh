gcloud config set project btibert-ba882-fall25

echo "======================================================"
echo "build (no cache)"
echo "======================================================"

docker build --no-cache -t gcr.io/btibert-ba882-fall25/genai-article-retrieval-poc .

echo "======================================================"
echo "push"
echo "======================================================"

docker push gcr.io/btibert-ba882-fall25/genai-article-retrieval-poc

echo "======================================================"
echo "deploy run"
echo "======================================================"


gcloud run deploy genai-article-retrieval-poc \
    --image gcr.io/btibert-ba882-fall25/genai-article-retrieval-poc \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --memory 1Gi