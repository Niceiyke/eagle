# middleware/base.py
"""Base middleware classes for Eagle framework."""
from abc import ABC, abstractmethod
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Callable, Optional, Any
import time
import uuid

class EagleMiddleware(BaseHTTPMiddleware, ABC):
    """Base class for Eagle middlewares with common functionality."""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.config = kwargs
        self.setup()
    
    def setup(self) -> None:
        """Override this method for middleware-specific setup."""
        pass
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method."""
        # Set request ID and timing
        request.state.request_id = str(uuid.uuid4())
        request.state.start_time = time.time()
        
        # Pre-processing
        await self.before_request(request)
        
        try:
            response = await call_next(request)
            # Post-processing
            response = await self.after_response(request, response)
            return response
        except Exception as e:
            return await self.handle_exception(request, e)
    
    async def before_request(self, request: Request) -> None:
        """Called before the request is processed."""
        pass
    
    async def after_response(self, request: Request, response: Response) -> Response:
        """Called after the response is generated."""
        return response
    
    async def handle_exception(self, request: Request, exc: Exception) -> Response:
        """Handle exceptions that occur during request processing."""
        raise exc