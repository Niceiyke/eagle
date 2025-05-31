"""
Eagle Admin Dashboard

This module provides an automatic admin interface for Eagle API applications.
"""
from typing import Optional, Type, Dict, Any
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os

class Admin:
    def __init__(self, app, path: str = "/admin"):
        """
        Initialize the admin interface.
        
        Args:
            app: The EagleAPI instance
            path: The URL path where the admin dashboard will be mounted
        """
        self.app = app
        self.path = path
        self._setup()
    
    def _setup(self):
        """Set up the admin interface."""
        # Get the directory where this file is located
        current_dir = Path(__file__).parent
        
        # Mount static files
        self.static_dir = current_dir / "static"
        if not self.static_dir.exists():
            self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure templates
        self.templates_dir = current_dir / "templates"
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.templates = Jinja2Templates(directory=str(self.templates_dir))
        
        # Add admin routes to the main app
        self._add_routes()
        
        # Import and include API routes
        from . import api
        self.app.include_router(api.router, prefix=f"{self.path}/api")
        
        # Mount static files
        self.app.mount(
            f"{self.path}/static",
            StaticFiles(directory=str(self.static_dir)),
            name="admin_static"
        )
        
        self.app.logger.info(f"Admin dashboard available at {self.path}")
    
    def _add_routes(self):
        """Add admin routes to the main application."""
        @self.app.get(self.path, include_in_schema=False)
        async def admin_dashboard(request: Request):
            return self.templates.TemplateResponse(
                "admin.html",
                {
                    "request": request,
                    "title": "Eagle Admin Dashboard",
                    "api_url": f"{self.path}/api"
                }
            )

def setup_admin(app, path: str = "/admin") -> Admin:
    """
    Set up the admin dashboard for the Eagle application.
    
    Args:
        app: The EagleAPI instance
        path: The URL path where the admin dashboard will be mounted
    """
    return Admin(app, path)
