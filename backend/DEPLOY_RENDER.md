# Deploying to Render.com

This guide explains how to deploy the Smart Portfolio backend to Render.com.

## Prerequisites

1. [Create a Render.com account](https://dashboard.render.com/register) if you don't have one already
2. Your code should be pushed to a GitHub repository

## Deployment Steps

### 1. Connect Your GitHub Repository

1. Log in to your Render.com account
2. Go to the Dashboard
3. Click on "New" and select "Blueprint" from the dropdown menu
4. Connect your GitHub account if you haven't already
5. Select the repository containing your Smart Portfolio application
6. Render will detect the `render.yaml` file and configure the service accordingly

### 2. Configure Environment Variables

After connecting your repository, you'll need to set up the required environment variables:

1. Click on the newly created service
2. Go to the "Environment" tab
3. Add the following environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ENCRYPTION_KEY`: A secure encryption key for data anonymization
   - `DATABASE_URL`: Your database URL (if applicable)

### 3. Deploy the Service

1. After setting up the environment variables, click on "Manual Deploy" and select "Deploy latest commit"
2. Render will build and deploy your application
3. Once the deployment is complete, you'll see a URL for your service (e.g., `https://smartportfolio-backend.onrender.com`)

### 4. Update Frontend Configuration

After deployment, update the frontend environment variable in `frontend/.env.production`:

```
VITE_API_URL=https://smartportfolio-backend.onrender.com
```

Replace the URL with the actual service URL from Render.com.

### 5. Redeploy the Frontend

After updating the frontend configuration, redeploy the frontend to GitHub Pages:

```bash
cd frontend
npm run deploy
```

## Monitoring and Logs

- To view logs, go to the "Logs" tab in your Render.com dashboard
- To monitor the service, go to the "Metrics" tab

## Troubleshooting

### CORS Issues

If you encounter CORS issues, make sure the CORS configuration in `backend/app/main.py` includes your frontend domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://organica-ai-solutions.github.io",  # GitHub Pages domain
        "http://localhost:5173",  # Local development
        "http://127.0.0.1:5173",  # Local development alternative
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Environment Variables

If your application is not working correctly, check that all required environment variables are set correctly in the Render.com dashboard. 