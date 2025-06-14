from typing import Dict, Any
from .base import EagleMiddleware
from fastapi import Request, Response
import time
import logging

logger = logging.getLogger(__name__)

class TimmingMiddleware(EagleMiddleware):
    """
    Middleware for measuring request processing time.
    
    This middleware tracks how long each request takes to process and adds the timing information to the response headers.
    """
    
    def __init__(self, app, time_header: str = "X-Process-Time", **kwargs):
        """
        Initialize the timing middleware.
        
        Args:
            app: The FastAPI application
            time_header: Header name to use for the process time (default: "X-Process-Time")
        """
        super().__init__(app)
        self.time_header = time_header
        
    async def before_request(self, request: Request) -> None:
        """
        Store the start time of the request.
        
        Args:
            request: The incoming request
        """
        request.state.start_time = time.time()
        
    async def after_response(self, request: Request, response: Response) -> Response:
        """
        Calculate the request processing time and add it to the response headers.
        
        Args:
            request: The request object
            response: The response object
            
        Returns:
            The response with added timing headers
        """
        # Calculate processing time
        process_time = time.time() - request.state.start_time
        
        # Add timing header to response
        response.headers[self.time_header] = f"{process_time:.4f} sec"
        
        return response