# auth/rate_limiting.py
"""
Rate limiting for authentication endpoints.
"""
import asyncio
import time
from typing import Dict, Tuple
from fastapi import HTTPException, status, Request
from functools import wraps


class InMemoryRateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
        self.lock = asyncio.Lock()
    
    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, int]:
        """Check if request is allowed and return remaining requests."""
        async with self.lock:
            now = time.time()
            window_start = now - window_seconds
            
            # Clean old requests
            if key in self.requests:
                self.requests[key] = [
                    req_time for req_time in self.requests[key]
                    if req_time > window_start
                ]
            else:
                self.requests[key] = []
            
            # Check rate limit
            current_requests = len(self.requests[key])
            if current_requests >= max_requests:
                return False, 0
            
            # Add current request
            self.requests[key].append(now)
            return True, max_requests - current_requests - 1

# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()

def rate_limit(max_requests: int, window_seconds: int, key_func=None):
    """Rate limiting decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # Look in kwargs
                request = kwargs.get('request')
            
            if not request:
                # If no request found, proceed without rate limiting
                return await func(*args, **kwargs)
            
            # Generate rate limit key
            if key_func:
                key = key_func(request)
            else:
                key = request.client.host if request.client else "unknown"
            
            # Check rate limit
            allowed, remaining = await rate_limiter.is_allowed(
                key, max_requests, window_seconds
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(window_seconds),
                        "X-RateLimit-Remaining": "0"
                    }
                )
            
            # Add rate limit headers to response
            response = await func(*args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Remaining"] = str(remaining)
            
            return response
        
        return wrapper
    return decorator
