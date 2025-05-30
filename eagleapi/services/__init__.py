"""
Service layer for the Eagle Framework.

This module contains the business logic and services used by the application.
"""

# Import services to make them available when importing from .services
from . import auth  # noqa: F401

__all__ = ["auth"]
