FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} 