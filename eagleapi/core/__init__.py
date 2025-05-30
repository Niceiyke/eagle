"""
Core functionality for the Eagle Framework.

This module provides the base classes and utilities that power the framework.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Coroutine
from fastapi import FastAPI, Request, Response, Depends
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import asyncio
import inspect
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eagle")

# Type variables
T = TypeVar('T')


class EagleAPI(FastAPI):
    """Main application class for the Eagle framework."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._extensions: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
        self._startup_tasks: List[Callable[[], Coroutine[Any, Any, None]]] = []
        self._shutdown_tasks: List[Callable[[], Coroutine[Any, Any, None]]] = []
    
    def add_extension(self, name: str, extension: Any) -> None:
        """Register an extension with the application."""
        self._extensions[name] = extension
        logger.info(f"Added extension: {name}")
    
    def get_extension(self, name: str) -> Any:
        """Get a registered extension by name."""
        return self._extensions.get(name)
    
    def configure(self, **config: Any) -> None:
        """Update application configuration."""
        self._config.update(config)
        logger.info(f"Updated configuration: {', '.join(config.keys())}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config
    
    def on_startup(self, func: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Register a function to run on application startup."""
        self._startup_tasks.append(func)
    
    def on_shutdown(self, func: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Register a function to run on application shutdown."""
        self._shutdown_tasks.append(func)
    
    async def _run_startup_tasks(self) -> None:
        """Run all registered startup tasks."""
        if self._startup_tasks:
            logger.info("Running startup tasks...")
            await asyncio.gather(*(task() for task in self._startup_tasks))
    
    async def _run_shutdown_tasks(self) -> None:
        """Run all registered shutdown tasks."""
        if self._shutdown_tasks:
            logger.info("Running shutdown tasks...")
            await asyncio.gather(*(task() for task in self._shutdown_tasks))


class APIVersioningMiddleware(BaseHTTPMiddleware):
    """Middleware for API versioning support."""
    
    def __init__(
        self,
        app: FastAPI,
        default_version: str = "1.0",
        header_name: str = "X-API-Version",
    ) -> None:
        super().__init__(app)
        self.default_version = default_version
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Get version from header or use default
        version = request.headers.get(self.header_name, self.default_version)
        
        # Add version to request state
        request.state.api_version = version
        
        # Process the request
        response = await call_next(request)
        
        # Add version to response headers
        response.headers[self.header_name] = version
        
        return response


class Route(APIRoute):
    """Custom route class with additional functionality."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Add custom parameters here if needed
        super().__init__(*args, **kwargs)
    
    async def get_route_handler(self) -> Callable:
        """Get the route handler with custom behavior."""
        original_route_handler = await super().get_route_handler()
        
        async def custom_route_handler(request: Request) -> Response:
            # Add pre-request logic here
            logger.debug(f"Processing request: {request.method} {request.url}")
            
            # Process the request
            response = await original_route_handler(request)
            
            # Add post-request logic here
            logger.debug(f"Request completed: {request.method} {request.url} -> {response.status_code}")
            
            return response
        
        return custom_route_handler


class Pagination(BaseModel):
    """Pagination helper for API responses."""
    
    page: int = 1
    per_page: int = 20
    total: int = 0
    
    @property
    def offset(self) -> int:
        """Calculate the offset for database queries."""
        return (self.page - 1) * self.per_page
    
    @property
    def total_pages(self) -> int:
        """Calculate the total number of pages."""
        if self.total == 0:
            return 0
        return (self.total + self.per_page - 1) // self.per_page
    
    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Convert to dictionary with pagination metadata."""
        base = super().dict(**kwargs)
        base.update({
            "total_pages": self.total_pages,
            "has_prev": self.page > 1,
            "has_next": self.page < self.total_pages,
        })
        return base


# Helper functions
def get_dependency_return_annotation(dependency: Callable) -> Any:
    """Get the return annotation of a dependency function."""
    sig = inspect.signature(dependency)
    return sig.return_annotation


def is_async_callable(obj: Any) -> bool:
    """Check if an object is an async callable."""
    if inspect.iscoroutinefunction(obj):
        return True
    
    if inspect.isclass(obj):
        return False
    
    if hasattr(obj, "__call__"):
        return inspect.iscoroutinefunction(obj.__call__)
    
    return False


# Export public API
__all__ = [
    'Eagle', 'APIVersioningMiddleware', 'Route', 'Pagination',
    'get_dependency_return_annotation', 'is_async_callable'
]
