# Deploying to Google Cloud Platform

This guide explains how to deploy the Smart Portfolio backend to Google Cloud Run.

## Prerequisites

1. [Create a Google Cloud account](https://cloud.google.com/free) if you don't have one
2. [Install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
3. [Install Docker](https://docs.docker.com/get-docker/) for local testing (optional)

## Setup Google Cloud Project

1. Create a new project in the [Google Cloud Console](https://console.cloud.google.com/)
   ```
   gcloud projects create smartportfolio-[unique-id] --name="Smart Portfolio"
   ```

2. Set the project as your default
   ```
   gcloud config set project smartportfolio-[unique-id]
   ```

3. Enable required APIs
   ```
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable run.googleapis.com
   gcloud services enable artifactregistry.googleapis.com
   ```

## Manual Deployment

1. Navigate to the backend directory
   ```
   cd backend
   ```

2. Build the Docker image
   ```
   gcloud builds submit --tag gcr.io/[PROJECT_ID]/smartportfolio-backend
   ```

3. Deploy to Cloud Run
   ```
   gcloud run deploy smartportfolio-backend \
     --image gcr.io/[PROJECT_ID]/smartportfolio-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars OPENAI_API_KEY=[your-api-key],ENCRYPTION_KEY=[your-encryption-key]
   ```

4. Get the service URL
   ```
   gcloud run services describe smartportfolio-backend --platform managed --region us-central1 --format 'value(status.url)'
   ```

## Automated Deployment with Cloud Build

1. Set up secret environment variables in Secret Manager
   ```
   gcloud secrets create OPENAI_API_KEY --replication-policy automatic
   gcloud secrets create ENCRYPTION_KEY --replication-policy automatic
   
   echo -n "your-openai-api-key" | gcloud secrets versions add OPENAI_API_KEY --data-file=-
   echo -n "your-encryption-key" | gcloud secrets versions add ENCRYPTION_KEY --data-file=-
   ```

2. Grant Secret Manager access to Cloud Build
   ```
   PROJECT_NUMBER=$(gcloud projects describe [PROJECT_ID] --format='value(projectNumber)')
   gcloud projects add-iam-policy-binding [PROJECT_ID] \
     --member=serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com \
     --role=roles/secretmanager.secretAccessor
   ```

3. Trigger a build from the repository
   ```
   gcloud builds submit --config cloudbuild.yaml \
     --substitutions _OPENAI_API_KEY=$(gcloud secrets versions access latest --secret=OPENAI_API_KEY),_ENCRYPTION_KEY=$(gcloud secrets versions access latest --secret=ENCRYPTION_KEY)
   ```

## Update Frontend Configuration

After deployment, update the frontend environment variable in `frontend/.env.production`:

```
VITE_API_URL=https://smartportfolio-backend-[hash].run.app
```

Replace the URL with the actual service URL from step 4 of the Manual Deployment section.

## Monitoring and Logs

- View logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=smartportfolio-backend" --limit 10`
- View service metrics: Visit the [Cloud Run console](https://console.cloud.google.com/run) 