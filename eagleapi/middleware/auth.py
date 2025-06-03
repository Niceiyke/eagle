# middleware/auth.py
"""Authentication middleware for Eagle framework."""
from .base import EagleMiddleware
from fastapi.requests import Request
from fastapi.responses import Response, JSONResponse
from typing import Optional, List
import logging

logger = logging.getLogger("eagle.middleware.auth")

class AuthMiddleware(EagleMiddleware):
    """Authentication middleware for protected routes."""
    
    def setup(self):
        self.protected_paths = self.config.get('protected_paths', [])
        self.excluded_paths = self.config.get('excluded_paths', ['/auth', '/health'])
        self.require_auth = self.config.get('require_auth', False)
    
    async def before_request(self, request: Request):
        """Check authentication before processing request."""
        if not self._requires_auth(request.url.path):
            return
        
        # Check for authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise AuthenticationRequired()
        
        # Extract and validate token
        token = auth_header.split(" ")[1]
        user = await self._validate_token(token)
        
        if not user:
            raise AuthenticationRequired()
        
        # Store user in request state
        request.state.user = user
    
    def _requires_auth(self, path: str) -> bool:
        """Check if path requires authentication."""
        if path in self.excluded_paths:
            return False
        
        if self.require_auth:
            return True
        
        return any(path.startswith(protected) for protected in self.protected_paths)
    
    async def _validate_token(self, token: str) -> Optional[dict]:
        """Validate JWT token and return user data."""
        # This would integrate with your auth system
        # For now, return a mock user
        try:
            # Add your JWT validation logic here
            return {"id": 1, "username": "user"}
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return None
    
    async def handle_exception(self, request: Request, exc: Exception) -> Response:
        """Handle authentication exceptions."""
        if isinstance(exc, AuthenticationRequired):
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication required"}
            )
        return await super().handle_exception(request, exc)

class AuthenticationRequired(Exception):
    """Exception raised when authentication is required."""
    pass