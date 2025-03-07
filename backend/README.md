# Smart Portfolio Backend

This is the backend API for the Smart Portfolio application, built with FastAPI.

## Local Development

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   uvicorn app.main:app --reload --port 8001
   ```

## Deployment

The backend can be deployed to various cloud platforms:

### Environment Variables

The following environment variables need to be set in the deployment environment:

- `OPENAI_API_KEY`: API key for OpenAI services
- `ENCRYPTION_KEY`: Key for encrypting sensitive data
- `DATABASE_URL`: Connection string for the database (if applicable)

### Render.com Deployment (Recommended)

The backend is configured for deployment on Render.com using Docker.

For detailed deployment instructions, see [DEPLOY_RENDER.md](./DEPLOY_RENDER.md).

Quick steps:
1. Connect your GitHub repository to Render.com
2. Create a new Web Service using the Blueprint option
3. Set the required environment variables
4. Deploy the service
5. Update the frontend environment variables with the Render.com service URL

### Google Cloud Run Deployment (Alternative)

Alternatively, the backend can be deployed on Google Cloud Run:

For detailed deployment instructions, see [DEPLOY_GCP.md](./DEPLOY_GCP.md).

Quick steps:
1. Set up a Google Cloud project with billing enabled
2. Build and deploy the Docker image to Cloud Run
3. Update the frontend environment variables with the Cloud Run service URL

## API Documentation

Once deployed, the API documentation is available at:
- Swagger UI: `https://[your-service-url]/docs`
- ReDoc: `https://[your-service-url]/redoc` 