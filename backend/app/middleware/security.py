import os
import time
from typing import Dict, Callable
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get current time
        current_time = time.time()
        
        # Initialize request list for client if not exists
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Remove requests older than 1 minute
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check if rate limit exceeded
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response

class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.api_key_header = os.getenv("API_KEY_HEADER", "X-API-Key")
        self.allowed_api_keys = set(os.getenv("ALLOWED_API_KEYS", "").split(","))
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip API key check for health check endpoints
        if request.url.path in ["/health", "/readiness"]:
            return await call_next(request)
        
        # Get API key from header
        api_key = request.headers.get(self.api_key_header)
        
        # Validate API key if configured
        if self.allowed_api_keys and self.allowed_api_keys != {""}:
            if not api_key:
                logger.warning("Missing API key in request")
                raise HTTPException(
                    status_code=401,
                    detail="Missing API key"
                )
            
            if api_key not in self.allowed_api_keys:
                logger.warning(f"Invalid API key: {api_key}")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key"
                )
        
        # Process request
        response = await call_next(request)
        return response 