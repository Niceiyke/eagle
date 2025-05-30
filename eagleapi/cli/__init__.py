"""
Command Line Interface for Eagle Framework.

Provides commands for project management, database operations, and development tasks.
"""
import os
import asyncio
import typer
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Awaitable, TypeVar, Sequence
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="Eagle Framework CLI")
console = Console()

# Create a new Eagle project
@app.command()
def new(name: str) -> None:
    """Create a new Eagle project.
    
    Args:
        name: The name of the project directory to create
    """
    project_dir = Path(name)
    
    if project_dir.exists():
        console.print(f"[red]Error: Directory '{name}' already exists![/red]")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Creating project '{name}'...", total=None)
        
        # Create project structure
        (project_dir / "app").mkdir(parents=True)
        (project_dir / "app" / "api").mkdir()
        (project_dir / "app" / "core").mkdir()
        (project_dir / "app" / "models").mkdir()
        (project_dir / "app" / "schemas").mkdir()
        (project_dir / "app" / "services").mkdir()
        (project_dir / "app" / "static").mkdir()
        (project_dir / "app" / "templates").mkdir()
        (project_dir / "tests").mkdir()
        (project_dir / "alembic").mkdir()
        
        # Create basic files
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class

# Environment variables
.env
.env.*
!.env.example

# Virtual Environment
venv/
env/
.venv/

# Database
*.db
*.sqlite3

# Logs
logs/
*.log

# IDE
.idea/
.vscode/
*.swp
*.swo

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
"""
        (project_dir / ".gitignore").write_text(gitignore_content)
        
        # Create README.md
        (project_dir / "README.md").write_text(f"# {name}\n\nAn Eagle Framework application.\n")
        
        # Create requirements.txt
        (project_dir / "requirements.txt").write_text("eagle-framework>=0.1.0\n")
        
        # Create .env file
        (project_dir / ".env.example").write_text("# Database\nDATABASE_URL=sqlite+aiosqlite:///./app.db\n\n# Security\nSECRET_KEY=your-secret-key-here\nALGORITHM=HS256\nACCESS_TOKEN_EXPIRE_MINUTES=30\n")
        
        # Create main.py
        (project_dir / "app" / "main.py").write_text('"""\nMain application module.\n"""\nfrom eagle import Eagle\n\napp = Eagle()\n\n@app.get("/")\nasync def read_root():\n    return {"message": "Welcome to Eagle Framework!"}\n')
        
        # Create __init__.py files
        for dir_path in [
            "app", "app/api", "app/core", "app/models", 
            "app/schemas", "app/services", "tests"
        ]:
            (project_dir / dir_path / "__init__.py").touch()
    
    console.print(f"[green]✓ Successfully created project '{name}'[/green]")
    console.print(f"\nTo get started, run:\n")
    console.print(f"  cd {name}")
    console.print("  python -m venv venv")
    console.print("  .\\venv\\Scripts\\activate  # On Windows")
    console.print("  pip install -r requirements.txt")
    console.print("  cp .env.example .env")
    console.print("  python -m uvicorn app.main:app --reload\n")
    console.print("Then open http://localhost:8000 in your browser.")


# Database commands
db = typer.Typer(help="Database operations")
app.add_typer(db, name="db")


@db.command()
def create():
    """Create database tables."""
    from ..db import db as database
    import asyncio
    
    async def _create_tables():
        await database.create_all()
    
    with console.status("Creating database tables..."):
        asyncio.run(_create_tables())
    console.print("[green]✓ Database tables created successfully[/green]")


@db.command()
def drop():
    """Drop all database tables."""
    from ..db import db as database
    import asyncio
    
    if typer.confirm("Are you sure you want to drop all database tables?"):
        async def _drop_tables():
            await database.drop_all()
        
        with console.status("Dropping database tables..."):
            asyncio.run(_drop_tables())
        console.print("[green]✓ Database tables dropped successfully[/green]")
    else:
        console.print("Operation cancelled.")


# Run the development server
@app.command()
def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
):
    """Run the development server."""
    import uvicorn
    
    console.print(f"Starting Eagle server at http://{host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


# Type variable for generic function return types
T = TypeVar('T')

# Export the CLI app
__all__ = ['app']
