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

The backend is configured for deployment on Render.com using Docker.

### Environment Variables

The following environment variables need to be set in the deployment environment:

- `OPENAI_API_KEY`: API key for OpenAI services
- `ENCRYPTION_KEY`: Key for encrypting sensitive data
- `DATABASE_URL`: Connection string for the database (if applicable)

### Deployment Steps

1. Push the code to a GitHub repository
2. Connect the repository to Render.com
3. Create a new Web Service, selecting "Deploy from Blueprint" and using the `render.yaml` file
4. Set the required environment variables
5. Deploy the service

## API Documentation

Once deployed, the API documentation is available at:
- Swagger UI: `https://[your-render-service-url]/docs`
- ReDoc: `https://[your-render-service-url]/redoc` 