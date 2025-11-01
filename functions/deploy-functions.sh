# deploy the functions

gcloud config set project btibert-ba882-fall25

echo "======================================================"
echo "deploying the schema setup"
echo "======================================================"

gcloud functions deploy raw-schema-setup \
    --gen2 \
    --runtime python312 \
    --trigger-http \
    --entry-point task \
    --source ./raw-schema \
    --stage-bucket btibert-ba882-fall25-functions \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512MB 

echo "======================================================"
echo "deploying the scoreboard extractor"
echo "======================================================"

gcloud functions deploy raw-extract-scoreboard \
    --gen2 \
    --runtime python312 \
    --trigger-http \
    --entry-point task \
    --source ./raw-extract-scoreboard \
    --stage-bucket btibert-ba882-fall25-functions \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512MB 

echo "======================================================"
echo "deploying the scoreboard file parser/ingestor"
echo "======================================================"

gcloud functions deploy raw-load-scoreboard \
    --gen2 \
    --runtime python312 \
    --trigger-http \
    --entry-point task \
    --source ./raw-parse-scoreboard \
    --stage-bucket btibert-ba882-fall25-functions \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512MB 

echo "======================================================"
echo "function to parse a single game"
echo "======================================================"

gcloud functions deploy raw-parse-game \
    --gen2 \
    --runtime python312 \
    --trigger-http \
    --entry-point task \
    --source ./raw-parse-game \
    --stage-bucket btibert-ba882-fall25-functions \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1GB 

echo "======================================================"
echo "staging: schema setup"
echo "======================================================"

gcloud functions deploy stage-schema \
    --gen2 \
    --runtime python312 \
    --trigger-http \
    --entry-point task \
    --source ./stage-schema \
    --stage-bucket btibert-ba882-fall25-functions \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512MB 

echo "======================================================"
echo "staging: load tables"
echo "======================================================"

gcloud functions deploy stage-load-tables \
    --gen2 \
    --runtime python312 \
    --trigger-http \
    --entry-point task \
    --source ./stage-load-tables \
    --stage-bucket btibert-ba882-fall25-functions \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512MB 

echo "======================================================"
echo "ML: train model"
echo "======================================================"

gcloud functions deploy ml-train-model \
    --gen2 \
    --runtime python312 \
    --trigger-http \
    --entry-point task \
    --source ./ml-train-model \
    --stage-bucket btibert-ba882-fall25-functions \
    --service-account ba882-fall25@btibert-ba882-fall25.iam.gserviceaccount.com \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2GB \
    --timeout 540s