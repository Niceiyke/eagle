"""
API routers for the Eagle Framework.

This module contains all the API routers for the application.
"""
from fastapi import APIRouter

# Create the main API router
router = APIRouter(prefix="/api/v1", tags=["api"])

# Import and include the auth router
from . import auth
router.include_router(auth.router, prefix="/auth", tags=["auth"])

__all__ = ["router"]
