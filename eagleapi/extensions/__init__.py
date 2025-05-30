"""
Extensions module for Eagle Framework.

Provides a way to extend the framework's functionality through a plugin system.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Coroutine
from fastapi import FastAPI, Request, Response, Depends
from pydantic import BaseModel
import inspect
import logging

logger = logging.getLogger("eagle.extensions")

# Type variables
T = TypeVar('T')


class Extension:
    """Base class for all Eagle extensions."""
    
    def __init__(self, app: Optional[FastAPI] = None, **kwargs: Any) -> None:
        self.app = app
        self.config = kwargs
        self._name = self.__class__.__name__
        
        if app is not None:
            self.init_app(app, **kwargs)
    
    def init_app(self, app: FastAPI, **kwargs: Any) -> None:
        """Initialize the extension with the application."""
        self.app = app
        self.config.update(kwargs)
        
        # Register extension with the app
        if not hasattr(app, 'extensions'):
            app.extensions = {}
        
        app.extensions[self._name.lower()] = self
        logger.info(f"Initialized extension: {self._name}")
    
    def __repr__(self) -> str:
        return f"<{self._name} extension>"


class ExtensionManager:
    """Manager for handling extensions."""
    
    def __init__(self, app: Optional[FastAPI] = None) -> None:
        self.extensions: Dict[str, Extension] = {}
        self.app = app
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app: FastAPI) -> None:
        """Initialize the extension manager with the application."""
        self.app = app
        
        # Create extensions namespace if it doesn't exist
        if not hasattr(app, 'extensions'):
            app.extensions = {}
    
    def register_extension(self, extension: Extension) -> None:
        """Register an extension with the manager."""
        if not isinstance(extension, Extension):
            raise TypeError("Extension must be an instance of Extension class")
        
        name = extension._name.lower()
        if name in self.extensions:
            raise ValueError(f"Extension '{name}' is already registered")
        
        self.extensions[name] = extension
        
        # Also register with the app if it's available
        if self.app is not None:
            extension.init_app(self.app)
        
        logger.info(f"Registered extension: {name}")
    
    def get_extension(self, name: str) -> Extension:
        """Get a registered extension by name."""
        name = name.lower()
        if name not in self.extensions:
            raise KeyError(f"Extension '{name}' is not registered")
        return self.extensions[name]
    
    def __getitem__(self, name: str) -> Extension:
        """Get an extension using dictionary-style access."""
        return self.get_extension(name)
    
    def __contains__(self, name: str) -> bool:
        """Check if an extension is registered."""
        return name.lower() in self.extensions


# Built-in extensions

class DatabaseExtension(Extension):
    """Database extension for SQLAlchemy integration."""
    
    def init_app(self, app: FastAPI, **kwargs: Any) -> None:
        """Initialize the database extension."""
        super().init_app(app, **kwargs)
        
        # Import here to avoid circular imports
        from ..db import db as database
        
        # Initialize the database with the app's config
        db_url = kwargs.get('DATABASE_URL', 'sqlite+aiosqlite:///./eagle.db')
        database.url = db_url
        
        # Add startup and shutdown handlers
        @app.on_event("startup")
        async def startup_db() -> None:
            """Initialize database on startup."""
            logger.info("Initializing database...")
            await database.create_all()
        
        @app.on_event("shutdown")
        async def shutdown_db() -> None:
            """Clean up database on shutdown."""
            logger.info("Closing database connections...")
            await database.engine.dispose()
        
        logger.info("Database extension initialized")


class AuthExtension(Extension):
    """Authentication and authorization extension."""
    
    def init_app(self, app: FastAPI, **kwargs: Any) -> None:
        """Initialize the auth extension."""
        super().init_app(app, **kwargs)
        
        # Import here to avoid circular imports
        from ..auth import get_current_user, get_current_active_user
        
        # Add auth dependencies to the app
        app.dependency_overrides[get_current_user] = get_current_active_user
        
        logger.info("Auth extension initialized")


class AdminExtension(Extension):
    """Admin interface extension."""
    
    def init_app(self, app: FastAPI, **kwargs: Any) -> None:
        """Initialize the admin extension."""
        super().init_app(app, **kwargs)
        
        # Import here to avoid circular imports
        from ..admin import admin_config
        
        # Register default models if any
        self._register_default_models()
        
        logger.info("Admin extension initialized")
    
    def _register_default_models(self) -> None:
        """Register default models with the admin interface."""
        # This can be extended to register default models
        pass


# Export public API
__all__ = [
    'Extension', 'ExtensionManager',
    'DatabaseExtension', 'AuthExtension', 'AdminExtension'
]
