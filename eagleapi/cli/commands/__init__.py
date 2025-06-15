"""
Command registration for the Eagle CLI.

This module imports and registers all command groups and commands.
"""
"""
Main CLI command registration.

This module sets up the main CLI command group and registers all subcommands.
It's designed to avoid importing the main application code unless necessary.
"""
import typer

# Create the main command group
app = typer.Typer(help="Eagle Framework CLI")

# Import project commands at module level (they don't depend on the main app)
from . import project
app.add_typer(project.app, name="project", help="Project management commands")

# Lazy load other commands that depend on the main application
@app.callback()
def main_callback():
    """Eagle Framework command line interface."""
    pass

# Server commands will be loaded on demand
# Add server commands directly to the main app
from . import server as server_module
app.add_typer(server_module.app, name="server", help="Server management commands")
# Migration commands will be loaded on demand
@app.command("migrations")
def migrations_cmd():
    """Manage database migrations."""
    from . import migrations as migrations_module
    migrations_module.app()

__all__ = ['app']
