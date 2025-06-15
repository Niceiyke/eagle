"""
Shared utilities for CLI commands.
"""
from typing import Any, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Global console instance
console = Console()

def create_progress() -> Progress:
    """Create a progress bar instance."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )

def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓ {message}[/green]")

def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]✗ {message}[/red]", err=True)

def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠ {message}[/yellow]")

def print_info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[blue]ℹ {message}[/blue]")
