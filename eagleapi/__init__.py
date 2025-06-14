#main __init__.py
"""
Eagle - A modern Python web framework built on FastAPI.

Eagle extends FastAPI with enterprise-grade features including database integration,
authentication, admin interface, and more, while maintaining performance and developer experience.
"""

__version__ = "0.1.0"

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import text
import logging
import pathlib
from .db import db, get_db
from .auth import AuthUser, get_current_superuser,AuthProvider
from .admin import AdminApp
from .utils.routes import router as utils_router
from .migrations import MigrationManager
from .core.config import settings
import asyncio
from .middleware.timming import TimmingMiddleware

# Initialize module-level logger; logging configuration is handled in `create_app`
logger = logging.getLogger(__name__)


class MigrationConfig:
        def __init__(self, database_url: str, environment: str = "development"):
            self.DATABASE_URL = database_url
            self.ENVIRONMENT = environment

    
class EagleAPI(FastAPI):
    """Main application class for the Eagle framework."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.logger = logging.getLogger(__name__)
        self._setup()
        self.add_event_handler("startup", self.on_startup)
        self.add_event_handler("shutdown", self.on_shutdown)

    
    def _setup(self):
        """Set up the application with middleware and routes."""
        
        # Include utiliy routes
        self.include_router(utils_router)

        admin_app = AdminApp()
        self.include_router(admin_app.router,include_in_schema=False)
        self.add_middleware(TimmingMiddleware)
    async def on_startup(self):
        """Handle application startup with migrations."""
        self.logger.info("Starting up Eagle application...")
        
        try:
            self.logger.info("Initializing database connection...")
            
            # Run database migrations in background thread to avoid blocking
            await asyncio.to_thread(self._run_migrations)
            
            # Create superuser
            await self._create_initial_superuser()
            
        except Exception as e:
            self.logger.error(f"Error during startup: {e}")
            raise
    
    def _run_migrations(self):
        """Initialize and apply database migrations during startup.
        
        This method will:
        1. Check if migrations are enabled
        2. Check if migrations directory exists
        3. Initialize migrations if they don't exist
        4. Apply pending migrations if auto_migrate is True
        """
        # Access settings stored in app state (set during create_app)
        enable_migrations = getattr(self.state, "enable_migrations", False)
        if not enable_migrations:
            self.logger.info("Database migrations are disabled")
            return

        logger = self.logger
        debug = getattr(self.state, "debug_mode", False)
        migrations_dir = getattr(self.state, "migrations_dir", "migrations")
        auto_migrate = getattr(self.state, "auto_migrate", True)

        logger.info("Checking database migrations...",
                   extra={"debug": debug, "migrations_dir": migrations_dir, "auto_migrate": auto_migrate})

        try:
            # Lazy imports to avoid circular dependencies at module import time
            from .core.config import settings  # local import to prevent issues

            # Create migrations directory if it doesn't exist
            path_dir = pathlib.Path(migrations_dir)
            path_dir.mkdir(parents=True, exist_ok=True)

            migration_config = MigrationConfig(
                database_url=str(settings.DATABASE_URL),
                environment="development" if debug else "production",
            )

            logger.info("Migration config created", extra={"config": migration_config})

            # Initialize migration manager
            migration_manager = MigrationManager(
                config=migration_config,
                migrations_dir=migrations_dir,
                require_backup=not debug,
            )
            logger.info("Migration manager initialized")

            # Check if migrations are already initialized by looking for alembic.ini
            migrations_initialized = (path_dir / "alembic.ini").exists()

            if not migrations_initialized:
                logger.info("Initializing new migration environment")
                migration_manager.init(description="Initial database migration")
                
                logger.info("Creating initial migration")
                migration_manager.create_migration("Initial migration")
                logger.info("Successfully initialized migrations")
            else:
                logger.info("Existing migrations found")
                # Ensure versions directory contains at least one migration script
                versions_dir = path_dir / "versions"
                if not versions_dir.exists() or not any(versions_dir.glob("*.py")):
                    logger.warning("No migration scripts found; creating baseline migration")
                    migration_manager.create_migration("Baseline migration")

            if auto_migrate:
                logger.info("Applying pending migrations...")
                migration_manager.upgrade()
                logger.info("Database migrations applied successfully")

            # Store manager for later use
            self.state.migration_manager = migration_manager

        except Exception as e:
            logger.error(f"Failed to initialize migrations: {e}")
            if debug:
                raise
            logger.warning("Continuing without database migrations")

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
 
   

def create_app(
    title: str = "Eagle Framework",
    description: str = "A modern Python web framework built on FastAPI",
    version: str = __version__,
    docs_url: str = "/docs",
    redoc_url: str = "/redoc",
    openapi_url: str = "/openapi.json",
    debug: bool = False,
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
        
        # Store migration configuration in app state for use during startup
        app.state.enable_migrations = enable_migrations
        app.state.migrations_dir = migrations_dir
        app.state.auto_migrate = auto_migrate
        app.state.debug_mode = debug
        
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
from .core.config import settings
from .tasks.background import BackgroundTaskQueue

__all__ = [
    'EagleAPI', 'create_app', 'Request', 'Response', 'Depends', 
    'HTTPException', 'status', 'APIRouter', 'BackgroundTasks', 'UploadFile', 
    'File', 'Form', 'Query', 'Path', 'Body', 'Header', 'Cookie','settings','MigrationError'
]
