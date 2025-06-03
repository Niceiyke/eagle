#main __init__.py
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
from .db import db, get_db
from .auth import User, get_current_superuser
from .middleware import MiddlewareManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
    
class EagleAPI(FastAPI):
    """Main application class for the Eagle framework."""
    
    def __init__(self, *args, **kwargs):
        # Extract middleware config
        self.middleware_config = kwargs.pop('middleware_config', {})
        super().__init__(*args, **kwargs)
        
        self.logger = logging.getLogger(__name__)
        self.middleware_manager = MiddlewareManager()
        self._setup()
        self.add_event_handler("startup", self.on_startup)
        self.add_event_handler("shutdown", self.on_shutdown)
    
    def _setup(self):
        """Set up the application with middleware and routes."""
        # Configure and apply middlewares
        self._setup_middlewares()
        
        # Set up admin dashboard
        self._admin = None
        if os.getenv("EAGLE_ADMIN_ENABLED", "true").lower() == "true":
            self.enable_admin()
    
    def _setup_middlewares(self):
        """Configure and apply middlewares based on configuration."""
        config = self.middleware_config
        
        # Configure logging middleware
        logging_config = config.get('logging', {})
        if logging_config.get('enabled', True):
            self.middleware_manager.configure_logging(
                enabled=True,
                log_methods=logging_config.get('methods', ['POST', 'PUT', 'DELETE']),
                excluded_paths=logging_config.get('excluded_paths', [
                    '/health', '/metrics', '/docs', '/redoc', '/openapi.json'
                ]),
                log_request_body=logging_config.get('log_request_body', True),
                log_response_body=logging_config.get('log_response_body', False),
                max_body_size=logging_config.get('max_body_size', 10 * 1024),
                batch_logging=logging_config.get('batch_logging', False)
            )
        
        # Configure CORS middleware
        cors_config = config.get('cors', {})
        if cors_config.get('enabled', True):
            self.middleware_manager.configure_cors(
                enabled=True,
                allow_origins=cors_config.get('origins', ["*"]),
                allow_methods=cors_config.get('methods', ["*"]),
                allow_credentials=cors_config.get('credentials', True)
            )
        
        # Configure rate limiting if enabled
        rate_limit_config = config.get('rate_limit', {})
        if rate_limit_config.get('enabled', False):
            self.middleware_manager.configure_rate_limit(
                enabled=True,
                calls=rate_limit_config.get('calls', 100),
                period=rate_limit_config.get('period', 60)
            )
            
        # Configure caching if enabled
        cache_config = config.get('cache', {})
        if cache_config.get('enabled', False):
            self.middleware_manager.add_middleware(
                'cache',
                default_ttl=cache_config.get('default_ttl', 300),
                cache_methods=cache_config.get('methods', ['GET']),
                cache_status_codes=cache_config.get('status_codes', [200]),
                excluded_paths=cache_config.get('excluded_paths', [
                    '/admin', '/docs', '/redoc', '/openapi.json'
                ]),
                vary_headers=cache_config.get('vary_headers', ['Authorization'])
            )
        
        # Apply all middlewares to the app
        try:
            self.middleware_manager.apply_to_app(self)
            self.logger.info("All middlewares applied successfully")
        except Exception as e:
            self.logger.error(f"Error applying middlewares: {e}")
            raise
    

    async def on_startup(self):
        """Handle application startup."""
        self.logger.info("Starting up Eagle application...")
        # Initialize database connection and create tables
        try:
            self.logger.info("Initializing database connection...")
            await db.create_tables()
            self.logger.info("Database tables created/verified")
            
            # Create superuser if it doesn't exist
            await self._create_initial_superuser()
            
            
        except Exception as e:
            self.logger.error(f"Error during database initialization: {e}")
            raise
    
    async def _create_initial_superuser(self):
        """Create initial superuser if it doesn't exist."""
        from .auth import get_user_by_email, create_user, UserCreate
        from .core.config import settings
        
        try:
            # Get an async session
            db_gen = get_db()
            db = await anext(db_gen)
            
            try:
                # Check if superuser already exists
                existing_user = await get_user_by_email(db, settings.SUPERUSER_EMAIL)
                
                if existing_user is None:
                    # Create superuser
                    user_data = {
                        'username': 'admin',
                        'email': settings.SUPERUSER_EMAIL,
                        'password': settings.SUPERUSER_PASSWORD,
                        'is_superuser': True,
                        'is_active': True,
                        'full_name': 'Admin User'
                    }
                    user_in = UserCreate(**user_data)
                    await create_user(db, user_in)
                    await db.commit()
                    self.logger.info("Initial superuser created successfully")
                else:
                    self.logger.info("Superuser already exists, skipping creation")
            finally:
                await db.close()
                
        except Exception as e:
            self.logger.error(f"Error creating initial superuser: {e}")
            raise
            

    async def on_shutdown(self):
        """Handle application shutdown events."""
        self.logger.info("Shutting down Eagle application...")
        
        try:
            self.logger.info("Eagle Application shutdown complete")
                
        except Exception as e:
            self.logger.error(f"Error during application shutdown: {e}", exc_info=True)

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



def create_app(
    title: str = "Eagle Framework",
    description: str = "A modern Python web framework built on FastAPI",
    version: str = __version__,
    docs_url: str = "/docs",
    redoc_url: str = "/redoc",
    openapi_url: str = "/openapi.json",
    debug: bool = False,
    middleware_config: Optional[Dict[str, Any]] = None,
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
        
        middleware_config: Middleware configuration dictionary. Example:
            {
                "logging": {
                    "enabled": True,
                    "methods": ["POST", "PUT", "DELETE"],
                    "excluded_paths": ["/health", "/metrics"],
                    "log_request_body": True,
                    "log_response_body": False,
                    "batch_logging": False
                },
                "cors": {
                    "enabled": True,
                    "origins": ["*"],
                    "methods": ["*"],
                    "credentials": True
                },
                "rate_limit": {
                    "enabled": False,
                    "calls": 100,
                    "period": 60
                }
            }
    Returns:
        EagleAPI: The configured Eagle application instance.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)

    # Set default middleware config
    default_middleware_config = {
        "logging": {"enabled": True},
        "cors": {"enabled": True},
        "rate_limit": {"enabled": False}
    }
    
    if middleware_config:
        # Merge with defaults
        for key, value in middleware_config.items():
            if key in default_middleware_config:
                default_middleware_config[key].update(value)
            else:
                default_middleware_config[key] = value
    
    try:
        logger.info(f"Creating {title} application (version: {version})")
        
        # Initialize the application
        app = EagleAPI(
            title=title,
            description=description,
            version=version,
            debug=debug,
            middleware_config=default_middleware_config,
            **kwargs
        )       
        # Import and include API routers
        try:
            from .auth import router as auth_router
            app.include_router(auth_router)
            logger.info("Auth router included")
        except ImportError as e:
            logger.warning(f"Could not import API router: {e}")
        
        logger.info("Application initialization complete")

        
        @app.get("/health", include_in_schema=True)
        async def health_check(current_user: User = Depends(get_current_superuser)):
            """Health check endpoint."""
            health_status = {
                "status": "ok",
                "database": "disconnected",
            }
            
            # Check database
            try:
                if hasattr(db, 'engine') and db.engine is not None:
                    async with db.engine.connect() as conn:
                        await conn.execute(text("SELECT 1"))
                    health_status["database"] = "connected"
                else:
                    health_status["message"] = "Database not initialized"
            except Exception as e:
                health_status["database"] = "disconnected"
                health_status["database_error"] = str(e)

            
            return health_status
                
        return app
        
    except Exception as e:
        logger.critical(f"Failed to create application: {e}", exc_info=True)
        raise

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

__all__ = [
    'EagleAPI', 'create_app', 'Request', 'Response', 'Depends', 
    'HTTPException', 'status', 'APIRouter', 'BackgroundTasks', 'UploadFile', 
    'File', 'Form', 'Query', 'Path', 'Body', 'Header', 'Cookie'
]
