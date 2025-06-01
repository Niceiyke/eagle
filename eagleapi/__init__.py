"""
Eagle - A modern Python web framework built on FastAPI.

Eagle extends FastAPI with enterprise-grade features including database integration,
authentication, admin interface, and more, while maintaining performance and developer experience.
"""

__version__ = "0.1.0"

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from typing import Optional, List, Dict, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import os
import logging
from pathlib import Path
from eagleapi.db import db
from .core.config import settings
from .middleware.logging import LoggingMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EagleAPI(FastAPI):
    """Main application class for the Eagle framework."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self._setup()
        self.add_event_handler("startup", self.on_startup)
        self.add_event_handler("shutdown", self.on_shutdown)
    
    def _setup(self):
        """Set up the application with middleware and routes."""
        # Add CORS middleware
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Set up admin dashboard if enabled
        self._admin = None
        if os.getenv("EAGLE_ADMIN_ENABLED", "true").lower() == "true":
            self.enable_admin()
    
    @property
    def admin(self):
        """Get the admin interface instance."""
        if self._admin is None:
            raise RuntimeError("Admin interface is not enabled. Call enable_admin() first.")
        return self._admin
    
    def enable_admin(self, path: str = None) -> 'Admin':
        """
        Enable the admin interface.
        
        Args:
            path: The URL path where the admin dashboard will be mounted
            
        Returns:
            The Admin instance
        """
        if self._admin is not None:
            return self._admin
            
        from .admin import Admin
        admin_path = path or os.getenv("EAGLE_ADMIN_PATH", "/admin")
        self._admin = Admin(self, path=admin_path)
        return self._admin
        
    def disable_admin(self):
        """Disable the admin interface."""
        # Note: This won't remove already mounted routes, but will prevent new ones
        self._admin = None
        
    async def on_startup(self):
        """Handle application startup."""
        self.logger.info("Starting up Eagle application...")
        # Initialize database connection and create tables
        try:
            self.logger.info("Initializing database connection...")
            await db.create_tables()
            self.logger.info("Database tables created/verified")
            
        except Exception as e:
            self.logger.error(f"Error during database initialization: {e}")
            raise
    
    async def on_shutdown(self):
        """
        Handle application shutdown events.
        
        This method is called when the application is shutting down and is responsible
        for cleaning up resources like database connections.
        """
        self.logger.info("Shutting down Eagle application...")
        
        try:
            # Close database connections if they exist
            self.logger.info("Shutting down Eagle Application")
                
        except Exception as e:
            self.logger.error(f"Error during application shutdown: {e}", exc_info=True)
            # Don't raise during shutdown to allow other cleanup to proceed

def create_app(
    title: str = "Eagle Framework",
    description: str = "A modern Python web framework built on FastAPI",
    version: str = __version__,
    docs_url: str = "/docs",
    redoc_url: str = "/redoc",
    openapi_url: str = "/openapi.json",
    debug: bool = False,
    **kwargs
) -> EagleAPI:
    """
    Create and configure the Eagle application.
    
    Args:
        title: The title of the API.
        description: The description of the API.
        version: The version of the API.
        docs_url: The URL where the API documentation will be served.
        redoc_url: The URL where the ReDoc documentation will be served.
        openapi_url: The URL where the OpenAPI schema will be served.
        debug: Whether to run the application in debug mode.
        **kwargs: Additional keyword arguments to pass to the FastAPI constructor.
        
    Returns:
        EagleAPI: The configured Eagle application instance.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Creating {title} application (version: {version})")
        
        # Initialize the application
        app = EagleAPI(
            title=title,
            description=description,
            version=version,
            docs_url=docs_url,
            redoc_url=redoc_url,
            openapi_url=openapi_url,
            debug=debug,
            **kwargs
        )
        
        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.add_middleware(LoggingMiddleware)
        
        # Import and include API routers
        try:
            from .auth import router as auth_router
            app.include_router(auth_router, prefix="/auth")
            logger.info("Auth router included")
            # Add root endpoint that redirects to the API documentation
            @app.get("/root", include_in_schema=False)
            async def root():
                """Root endpoint that redirects to the API documentation."""
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url=docs_url)
            
            # Add health check endpoint
            @app.get("/health", include_in_schema=False)
            async def health_check():
                """Health check endpoint."""
                try:
                    # Check if database is initialized
                    if not hasattr(db, 'engine') or db.engine is None:
                        return {
                            "status": "ok",
                            "database": "disconnected",
                            "message": "Database not initialized"
                        }
                    
                    # Test database connection
                    async with db.engine.connect() as conn:
                        await conn.execute(text("SELECT 1"))
                    
                    return {
                        "status": "ok",
                        "database": "connected"
                    }
                except Exception as e:
                    app.logger.error(f"Database connection error: {e}", exc_info=True)
                    return {
                        "status": "ok",  # Still return 200 but indicate database is down
                        "database": "disconnected",
                        "error": str(e)
                    }
        except ImportError as e:
            logger.warning(f"Could not import API v1 router: {e}")
        
        logger.info("Application initialization complete")
        return app
        
    except Exception as e:
        logger.critical(f"Failed to create application: {e}", exc_info=True)
        raise

# Create a default application instance
app = create_app()

# Export common FastAPI components for easier access
from fastapi import (  # noqa
    Depends, FastAPI, HTTPException, status, Request, Response, 
    APIRouter, BackgroundTasks, 
    UploadFile, File, Form, Query, Path, Body, Header, Cookie
)
from fastapi.security import OAuth2PasswordBearer  # noqa
from fastapi.middleware import Middleware  # noqa
from fastapi.middleware.cors import CORSMiddleware  # noqa
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # noqa
from fastapi.middleware.gzip import GZipMiddleware  # noqa
from pydantic import BaseModel  # noqa

__all__ = [
    'EagleAPI', 'create_app', 'Request', 'Response', 'Depends', 
    'HTTPException', 'status', 'APIRouter', 'BackgroundTasks', 'UploadFile', 
    'File', 'Form', 'Query', 'Path', 'Body', 'Header', 'Cookie', 'BaseModel'
]
