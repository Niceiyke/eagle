"""
Server management commands.

This module is loaded on-demand when server-related commands are invoked.
"""
import typer
from typing import Optional

# Import these only when needed to avoid loading the full application
from ..utils import console, print_success

# Create the command group
app = typer.Typer(help="Server management commands")

@app.command("run")
def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
) -> None:
    """Run the development server."""
    # Import uvicorn only when needed
    import uvicorn
    
    print_success(f"Starting Eagle server at http://{host}:{port}")
    # Change directory to the project root and use the correct import path
    import os
    import sys
    from pathlib import Path
    
    # Get the current working directory
    cwd = Path.cwd()
    
    # Add the current directory to Python path if it's not already there
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        reload_dirs=[str(cwd / "app")] if reload else None,
        app_dir=str(cwd)  # Tell uvicorn where the app is located
    )

@app.command("status")
def server_status() -> None:
    """Check server status."""
    # Import the actual implementation only when needed
    from eagleapi.core.config import settings
    from ..utils import print_info
    
    print_info("Server status:")
    print_info(f"  Environment: {settings.ENV}")
    print_info(f"  Debug mode: {settings.DEBUG}")
    print_info(f"  Docs: http://localhost:{settings.PORT}/docs" if settings.DOCS_ENABLED else "  Docs: Disabled")
