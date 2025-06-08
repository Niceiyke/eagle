# auth/middleware.py
"""
Authentication middleware for Eagle Framework.
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_db
from . import decode_token, TokenType, get_user_by_id
from .session_management import SessionManager
from .audit import AuditService, AuditAction
from typing import List


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware."""
    
    def __init__(self, app, excluded_paths: List[str] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/docs", "/redoc", "/openapi.json",
            "/auth/login", "/auth/register", "/auth/token"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Skip middleware for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Check for authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid authorization header"}
            )
        
        token = auth_header.split(" ")[1]
        
        try:
            # Decode token
            payload = decode_token(token, TokenType.ACCESS)
            user_id = payload.get("user_id")
            
            # Get database session
            db_gen = get_db()
            db = await db_gen.__anext__()
            
            try:
                # Verify user exists and is active
                user = await get_user_by_id(db, user_id)
                if not user or not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User not found or inactive"
                    )
                
                # Update session activity if session management is enabled
                session_token = request.headers.get("x-session-token")
                if session_token:
                    await SessionManager.update_session_activity(db, session_token)
                
                # Add user to request state
                request.state.current_user = user
                
                response = await call_next(request)
                return response
                
            finally:
                await db_gen.aclose()
                
        except HTTPException:
            raise
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Authentication error"}
            )
