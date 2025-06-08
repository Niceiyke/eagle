#main __init__.py
"""
Eagle - A modern Python web framework built on FastAPI.

Eagle extends FastAPI with enterprise-grade features including database integration,
authentication, admin interface, and more, while maintaining performance and developer experience.
"""

__version__ = "0.1.0"

from fastapi import FastAPI, Depends, HTTPException, status
from typing import Optional, Dict, Any, List
from sqlalchemy import text
import os
import logging
from pathlib import Path
from .db import db, get_db
from .db.migrations import migration_manager,MigrationError
from .auth import AuthUser, get_current_superuser,AuthProvider
from .admin import AdminApp
from .middleware import MiddlewareManager
from .utils.routes import router as utils_router
import asyncio
import sys
from .core.config import settings
from .tasks.background import BackgroundTaskQueue

# Global background task queue instance
background_tasks = BackgroundTaskQueue()


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

    
    
    def _setup_migration_endpoints(self):
        """Add migration management endpoints"""
        # Prevent duplicate endpoint registration
        if hasattr(self, "_migration_endpoints_registered"):
            return
            
        from .auth import get_current_superuser
        from .db.migrations import get_migration_status, get_migration_history
        
        @self.get("/migrations/status", include_in_schema=True)
        async def migration_status(
            current_user: AuthUser = Depends(get_current_superuser)
        ):
            """Get migration status (superuser only)"""
            try:
                return get_migration_status()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get migration status: {e}"
                )
        
        @self.get("/migrations/history", include_in_schema=True)
        async def migration_history(
            current_user: AuthUser = Depends(get_current_superuser)
        ):
            """Get migration history (superuser only)"""
            try:
                history = get_migration_history()
                return [
                    {
                        "revision": m.revision,
                        "description": m.description,
                        "created_at": m.created_at.isoformat(),
                        "is_head": m.is_head,
                        "is_current": m.is_current
                    }
                    for m in history
                ]
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get migration history: {e}"
                )
        
        @self.post("/migrations/upgrade", include_in_schema=True)
        async def upgrade_migrations(
            revision: str = "head",
            current_user: AuthUser = Depends(get_current_superuser)
        ):
            """Upgrade migrations (superuser only)"""
            try:
                from .db.migrations import upgrade_database
                await asyncio.get_event_loop().run_in_executor(
                    None, upgrade_database, revision
                )
                return {"message": f"Upgraded to revision: {revision}"}
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Migration upgrade failed: {e}"
                )
        
        @self.post("/migrations/create", include_in_schema=True)
        async def create_migration(
            revision: str,
            description: str,
            current_user: AuthUser = Depends(get_current_superuser)
        ):
            """Create a new migration (superuser only)"""
            try:
                from .db.migrations import create_migration
                await asyncio.get_event_loop().run_in_executor(
                    None, create_migration, revision, description
                )
                return {"message": f"Created migration: {revision}"}
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Migration creation failed: {e}"
                )
                
        # Mark endpoints as registered
        self._migration_endpoints_registered = True
    def _setup(self):
        """Set up the application with middleware and routes."""
        # Configure and apply middlewares
        self._setup_middlewares()
        
        # Include utility routes
        self.include_router(utils_router)

        admin_app = AdminApp()
        self.include_router(admin_app.router,include_in_schema=False)


        
    
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
            
        # # Configure caching if enabled
        # cache_config = config.get('cache', {})
        # if cache_config.get('enabled', False):
        #     self.middleware_manager.add_middleware(
        #         'cache',
        #         default_ttl=cache_config.get('default_ttl', 300),
        #         cache_methods=cache_config.get('methods', ['GET']),
        #         cache_status_codes=cache_config.get('status_codes', [200]),
        #         excluded_paths=cache_config.get('excluded_paths', [
        #             '/admin', '/docs', '/redoc', '/openapi.json'
        #         ]),
        #         vary_headers=cache_config.get('vary_headers', ['Authorization'])
        #     )
        
        # Apply all middlewares to the app
        try:
            self.middleware_manager.apply_to_app(self)
            self.logger.info("All middlewares applied successfully")
        except Exception as e:
            self.logger.error(f"Error applying middlewares: {e}")
            raise
    
    async def on_startup(self):
        """Handle application startup with migrations."""
        self.logger.info("Starting up Eagle application...")
        
        try:
            self.logger.info("Initializing database connection...")
            
            # Auto-migrate if enabled
            auto_migrate = getattr(self, '_auto_migrate', True)
            
            # Only create initial migration if no migrations exist
            try:
                current_rev = migration_manager.current()
                if current_rev is None:
                    self.logger.info("No existing migrations found, creating initial migration")
                    migration_manager.create_migration("Initial migration")
                else:
                    self.logger.info(f"Existing migrations found, current revision: {current_rev}")
            except Exception as e:
                self.logger.warning(f"Could not check for existing migrations: {e}")
                if auto_migrate:
                    self.logger.info("Creating initial migration")
                    migration_manager.create_migration("Initial migration")
            
            
            
            self.logger.info("Database setup completed")
            
            # Create superuser
            await self._create_initial_superuser()
            
        except Exception as e:
            self.logger.error(f"Error during startup: {e}")
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
                        'confirm_password': settings.SUPERUSER_PASSWORD,  # Add confirm_password
                        'is_superuser': True,
                        'is_active': True,
                        'is_verified': True,
                        'auth_provider': AuthProvider.LOCAL,
                        'social_id': None,
                        'last_login': None,
                        'failed_login_attempts': 0,
                        'locked_until': None,
                        'full_name': 'Admin AuthUser'
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

  
    def enable_migrations(
            self, 
            migrations_dir: str = "migrations",
            auto_migrate: bool = True
        ):
        print("Enabling migrations...", settings.DATABASE_URL)
        """
        Enable database migrations.
        
        This method initializes the migrations directory and sets up Alembic configuration.
        If the migrations directory doesn't exist, it will be created along with the
        necessary Alembic configuration files.
        
        Args:
            migrations_dir: Directory to store migration files (relative to the project root)
            auto_migrate: Whether to auto-run migrations on startup
            
        Returns:
            MigrationManager instance
            
        Raises:
            MigrationError: If there's an error initializing the migrations
        """
        from pathlib import Path
        from .db.migrations import migration_manager
        
        try:
            # Convert to absolute path and ensure directory exists
            migrations_path = Path(migrations_dir).absolute()
            self.logger.info(f"Configuring migrations in: {migrations_path}")
            
            # Ensure the directory exists
            migrations_path.mkdir(parents=True, exist_ok=True)
            
            # Set the migrations directory before initializing
            migration_manager.migrations_dir = migrations_path
            
            # Initialize the migrations directory if it's empty
            if not any(file for file in migrations_path.iterdir() if file.name != '__pycache__'):
                self.logger.info(f"Initializing new migrations directory: {migrations_path}")
                if not migration_manager.init("Initial migration setup"):
                    self.logger.warning("Migrations directory already contains files, skipping initialization")
            
            self._auto_migrate = auto_migrate
            
            # Ensure the migrations directory is in Python path
            migrations_parent = str(migrations_path.parent)
            if migrations_parent not in sys.path:
                sys.path.insert(0, migrations_parent)
            
            # Add migration endpoints
            self._setup_migration_endpoints()
            
            self.logger.info(f"Database migrations enabled in {migrations_path}")
            
            if auto_migrate:
                self.logger.info("Auto-migration is enabled and will run on startup")
                
        except Exception as e:
            error_msg = f"Failed to enable migrations: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise MigrationError(error_msg) from e
            
        return migration_manager

def create_app(
    title: str = "Eagle Framework",
    description: str = "A modern Python web framework built on FastAPI",
    version: str = __version__,
    docs_url: str = "/docs",
    redoc_url: str = "/redoc",
    openapi_url: str = "/openapi.json",
    debug: bool = False,
    middleware_config: Optional[Dict[str, Any]] = None,
    enable_migrations: bool = True,
    migrations_dir: str = "migrations",
    auto_migrate: bool = True,
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
            docs_url=docs_url,
            redoc_url=redoc_url,
            openapi_url=openapi_url,
            debug=debug,
            middleware_config=default_middleware_config,
            **kwargs
        )
        
        # Enable migrations if requested
        if enable_migrations:
            app.enable_migrations(
                migrations_dir=migrations_dir,
                auto_migrate=auto_migrate
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
        async def health_check(current_user: AuthUser = Depends(get_current_superuser)):
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
from eagleapi.tasks.background import BackgroundTaskQueue  # noqa

background_tasks = BackgroundTaskQueue()

__all__ = [
    'background_tasks',
    'EagleAPI', 'create_app', 'Request', 'Response', 'Depends', 
    'HTTPException', 'status', 'APIRouter', 'BackgroundTasks', 'UploadFile', 
    'File', 'Form', 'Query', 'Path', 'Body', 'Header', 'Cookie'
]
