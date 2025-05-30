"""
Eagle - A modern Python web framework built on FastAPI.

Eagle extends FastAPI with enterprise-grade features including database integration,
authentication, admin interface, and more, while maintaining performance and developer experience.
"""

__version__ = "0.1.0"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional, Type, AsyncGenerator
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .core.config import settings

class EagleAPI(FastAPI):
    """Main Eagle application class that extends FastAPI."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._extensions: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
    
    def add_extension(self, name: str, extension: Any) -> None:
        """Register an extension."""
        self._extensions[name] = extension
    
    def get_extension(self, name: str) -> Any:
        """Get a registered extension by name."""
        return self._extensions.get(name)
    
    def configure(self, **config: Any) -> None:
        """Update application configuration."""
        self._config.update(config)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config

def create_app() -> EagleAPI:
    """Create and configure the Eagle application."""
    # Initialize the app
    app = EagleAPI(
        title="Eagle Framework",
        description="A modern Python web framework built on FastAPI",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Set up database session getter for security module
    from .core import security
    from .db import get_db
    
    async def get_db_session() -> AsyncSession:
        """Get a database session for the security module."""
        async for session in get_db():
            return session
    
    # Set the database session getter in the security module
    security.set_db_session_getter(get_db_session)
    
    # Include API routers
    from .api import router as api_router
    app.include_router(api_router)
    
    # Add health check endpoint
    @app.get(f"{settings.API_V1_STR}/health", tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok"}
    
    # Add root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Welcome to Eagle Framework",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    
    return app

# Create a default application instance
app = create_app()

# Export common FastAPI components for easier access
from fastapi import (
    Request, Response, Depends, HTTPException, status,
    APIRouter, BackgroundTasks, UploadFile, File, Form, Query, Path, Body, Header, Cookie
)

__all__ = [
    'EagleAPI', 'app', 'create_app', 'Request', 'Response', 'Depends', 'HTTPException',
    'status', 'APIRouter', 'BackgroundTasks', 'UploadFile', 'File',
    'Form', 'Query', 'Path', 'Body', 'Header', 'Cookie', 'BaseModel'
]
