# middleware/__init__.py
"""
Eagle Middleware System

Provides a comprehensive middleware system for the Eagle framework.
"""
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI
import logging as log

from .base import EagleMiddleware
from .logging import LoggingMiddleware
from .auth import AuthMiddleware
from .cors import CORSMiddleware
from .rate_limit import RateLimitMiddleware
from .compression import CompressionMiddleware

log.basicConfig(level=log.INFO)

logger = log.getLogger("eagle.middleware")

class MiddlewareManager:
    """Manages middleware registration and configuration for Eagle apps."""
    
    def __init__(self):
        self.middlewares: List[Dict[str, Any]] = []
        self._builtin_middlewares = {
            'logging': LoggingMiddleware,
            'auth': AuthMiddleware,
            'cors': CORSMiddleware,
            'rate_limit': RateLimitMiddleware,
            'compression': CompressionMiddleware,
        }
    
    def add_middleware(
        self,
        middleware_class: Union[str, type],
        **options
    ) -> 'MiddlewareManager':
        """Add middleware to the stack."""
        if isinstance(middleware_class, str):
            if middleware_class not in self._builtin_middlewares:
                raise ValueError(f"Unknown middleware: {middleware_class}")
            middleware_class = self._builtin_middlewares[middleware_class]
        
        self.middlewares.append({
            'class': middleware_class,
            'options': options
        })
        return self
    
    def configure_logging(
        self,
        enabled: bool = True,
        log_methods: Optional[List[str]] = None,
        log_paths: Optional[List[str]] = None,
        excluded_paths: Optional[List[str]] = None,
        **kwargs
    ) -> 'MiddlewareManager':
        """Configure logging middleware."""
        if enabled:
            options = {
                'log_methods': set(log_methods) if log_methods else None,
                'log_paths': log_paths or [],
                'excluded_paths': excluded_paths or ['/health', '/metrics'],
                **kwargs
            }
            return self.add_middleware('logging', **options)
        return self
    
    def configure_cors(
        self,
        enabled: bool = True,
        allow_origins: List[str] = None,
        allow_methods: List[str] = None,
        **kwargs
    ) -> 'MiddlewareManager':
        """Configure CORS middleware."""
        if enabled:
            options = {
                'allow_origins': allow_origins or ["*"],
                'allow_methods': allow_methods or ["*"],
                'allow_credentials': True,
                'allow_headers': ["*"],
                **kwargs
            }
            return self.add_middleware('cors', **options)
        return self
    
    def configure_rate_limit(
        self,
        enabled: bool = False,
        calls: int = 100,
        period: int = 60,
        **kwargs
    ) -> 'MiddlewareManager':
        """Configure rate limiting middleware."""
        if enabled:
            options = {
                'calls': calls,
                'period': period,
                **kwargs
            }
            return self.add_middleware('rate_limit', **options)
        return self
    
    def apply_to_app(self, app: FastAPI) -> None:
        """Apply all configured middlewares to the FastAPI app."""
        # Apply middlewares in reverse order (LIFO stack)
        for middleware_config in reversed(self.middlewares):
            middleware_class = middleware_config['class']
            options = middleware_config['options']
            
            try:
                app.add_middleware(middleware_class, **options)
                logger.info(f"Added middleware: {middleware_class.__name__}")
            except Exception as e:
                logger.error(f"Failed to add middleware {middleware_class.__name__}: {e}")
                raise

middleware_manager = MiddlewareManager()

__all__ = [
    'MiddlewareManager',
    'EagleMiddleware', 
    'LoggingMiddleware',
    'middleware_manager',
    'EagleAPI',
    'create_app'
]