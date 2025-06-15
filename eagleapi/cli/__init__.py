"""
Command Line Interface for Eagle Framework.

This module provides the main entry point for the Eagle Framework CLI.
It imports and registers all command groups from the commands package.
"""
import typer

# Import the main command group from commands package
from .commands import app

# Re-export the app for backward compatibility
__all__ = ['app']

# This allows the module to be run directly with `python -m eagleapi.cli`
if __name__ == "__main__":
    app()
