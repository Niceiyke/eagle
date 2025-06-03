# middleware/cors.py
"""CORS middleware for Eagle framework."""
from .base import EagleMiddleware
from fastapi.requests import Request
from fastapi.responses import Response, PlainTextResponse
from typing import List

class CORSMiddleware(EagleMiddleware):
    """CORS middleware with configurable options."""
    
    def setup(self):
        self.allow_origins = self.config.get('allow_origins', ["*"])
        self.allow_methods = self.config.get('allow_methods', ["*"])
        self.allow_headers = self.config.get('allow_headers', ["*"])
        self.allow_credentials = self.config.get('allow_credentials', True)
        self.max_age = self.config.get('max_age', 86400)
    
    async def before_request(self, request: Request):
        """Handle CORS preflight requests."""
        if request.method == "OPTIONS":
            request.state.is_preflight = True
    
    async def after_response(self, request: Request, response: Response) -> Response:
        """Add CORS headers to response."""
        if getattr(request.state, 'is_preflight', False):
            return PlainTextResponse(
                content="",
                status_code=200,
                headers=self._get_cors_headers(request)
            )
        
        # Add CORS headers to regular responses
        for key, value in self._get_cors_headers(request).items():
            response.headers[key] = value
        
        return response
    
    def _get_cors_headers(self, request: Request) -> dict:
        """Generate CORS headers."""
        origin = request.headers.get("origin")
        headers = {}
        
        if self.allow_origins == ["*"] or (origin and origin in self.allow_origins):
            headers["Access-Control-Allow-Origin"] = origin or "*"
        
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        
        if request.method == "OPTIONS":
            if self.allow_methods == ["*"]:
                headers["Access-Control-Allow-Methods"] = "*"
            else:
                headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            
            if self.allow_headers == ["*"]:
                headers["Access-Control-Allow-Headers"] = "*"
            else:
                headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            
            headers["Access-Control-Max-Age"] = str(self.max_age)
        
        return headers