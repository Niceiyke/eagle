# middleware/rate_limit.py
"""Rate limiting middleware for Eagle framework."""
from .base import EagleMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from typing import Dict
import time
import asyncio
import logging as log


log.basicConfig(level=log.INFO)

logger = log.getLogger("eagle.ratelimit-middleware")


class RateLimitMiddleware(EagleMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def setup(self):
        self.calls = self.config.get('calls', 100)
        self.period = self.config.get('period', 60)
        self.storage: Dict[str, Dict] = {}
        self.cleanup_task = None
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        return request.client.host if request.client else "unknown"
    
    async def before_request(self, request: Request):
        """Check rate limit before processing request."""
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        if client_id not in self.storage:
            self.storage[client_id] = {
                'calls': 1,
                'window_start': current_time
            }
            return
        
        client_data = self.storage[client_id]
        
        # Reset window if period has passed
        if current_time - client_data['window_start'] >= self.period:
            client_data['calls'] = 1
            client_data['window_start'] = current_time
        else:
            client_data['calls'] += 1
        
        # Check if rate limit exceeded
        if client_data['calls'] > self.calls:
            remaining_time = self.period - (current_time - client_data['window_start'])
            raise RateLimitExceeded(remaining_time)
    
    async def handle_exception(self, request: Request, exc: Exception) -> Response:
        """Handle rate limit exceptions."""
        if isinstance(exc, RateLimitExceeded):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": int(exc.retry_after)
                },
                headers={"Retry-After": str(int(exc.retry_after))}
            )
        return await super().handle_exception(request, exc)
    
    async def _cleanup_expired(self):
        """Periodic cleanup of expired rate limit data."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                current_time = time.time()
                
                expired_clients = [
                    client_id for client_id, data in self.storage.items()
                    if current_time - data['window_start'] >= self.period * 2
                ]
                
                for client_id in expired_clients:
                    del self.storage[client_id]
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limit cleanup error: {e}")

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.1f} seconds")
