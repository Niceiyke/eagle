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
from .core.config import settings
from .middleware.logging import LoggingMiddleware
from .cache import setup_eagle_cache, CacheConfig
from .cache.core import cache_manager
from .auth import User, get_current_superuser


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EagleAPI(FastAPI):
    """Main application class for the Eagle framework."""
    
    def __init__(self, *args, **kwargs):
        # Extract cache config before passing to parent
        self.cache_config = kwargs.pop('cache_config', None)
        super().__init__(*args, **kwargs)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self._setup()
        self.add_event_handler("startup", self.on_startup)
        self.add_event_handler("shutdown", self.on_shutdown)
    
    def _setup(self):
        """Set up the application with middleware and routes."""
        # Set up caching
        self._setup_cache()
        
        # Set up admin dashboard if enabled
        self._admin = None
        if os.getenv("EAGLE_ADMIN_ENABLED", "true").lower() == "true":
            self.enable_admin()
    
    def _setup_cache(self):
        """Set up caching system."""
        try:
            setup_eagle_cache(self, self.cache_config)
            self.logger.info("Cache system initialized successfully")
        except Exception as e:
            self.logger.warning(f"Cache initialization failed: {e}")
            # Continue without cache
    
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
            
            # Warm up cache if configured
            await self._warm_cache()
            
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
            
    async def _warm_cache(self):
        """Warm up cache with initial data."""
        try:
            cache = cache_manager.get_cache()
            if cache:
                # Add your cache warming logic here
                # Example: pre-load configuration
                self.logger.info("Cache warming completed")
        except Exception as e:
            self.logger.warning(f"Cache warming failed: {e}")
    
    async def on_shutdown(self):
        """Handle application shutdown events."""
        self.logger.info("Shutting down Eagle application...")
        
        try:
            # Close cache connections
            await cache_manager.close_all()
            self.logger.info("Cache connections closed")
            
            # Close database connections if they exist
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
    cache_config: Optional[Dict[str, Any]] = None,
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
        cache_config: Cache configuration dictionary. Examples:
            # In-memory cache (default)
            None or {"backend": "memory"}
            
            # Redis cache
            {
                "backend": "redis",
                "redis_url": "redis://localhost:6379/0",
                "serializer": "json",
                "default_ttl": 3600
            }
            
            # Memcached
            {
                "backend": "memcached",
                "servers": ["localhost:11211"],
                "serializer": "pickle"
            }
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
            cache_config=cache_config,  # Pass cache config
            **kwargs
        )
        
        # Configure CORS (removed duplicate from _setup)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        
        # Import and include API routers
        try:
            from .auth import router as auth_router
            app.include_router(auth_router)
            logger.info("Auth router included")
        except ImportError as e:
            logger.warning(f"Could not import API router: {e}")
        
        logger.info("Application initialization complete")

        
        
        # Enhanced health check with cache status
        @app.get("/health", include_in_schema=True)
        async def health_check(current_user: User = Depends(get_current_superuser)):
            """Health check endpoint."""
            health_status = {
                "status": "ok",
                "database": "disconnected",
                "cache": "unavailable"
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
            
            # Check cache
            try:
                cache = cache_manager.get_cache()
                if cache:
                    # Test cache operation
                    test_key = "health_check_test"
                    await cache.set(test_key, "ok", ttl=10)
                    result = await cache.get(test_key)
                    await cache.delete(test_key)
                    
                    if result == "ok":
                        health_status["cache"] = "connected"
                        cache_stats = await cache.stats()
                        health_status["cache_stats"] = {
                            "hit_rate": cache_stats.hit_rate,
                            "total_operations": cache_stats.hits + cache_stats.misses
                        }
                    else:
                        health_status["cache"] = "error"
            except Exception as e:
                health_status["cache"] = "error"
                health_status["cache_error"] = str(e)
            
            return health_status
                
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

__all__ = [
    'EagleAPI', 'create_app', 'Request', 'Response', 'Depends', 
    'HTTPException', 'status', 'APIRouter', 'BackgroundTasks', 'UploadFile', 
    'File', 'Form', 'Query', 'Path', 'Body', 'Header', 'Cookie'
]
