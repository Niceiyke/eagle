"""
Project management commands.

This module is completely isolated from the main application to allow
project scaffolding without importing any application code.
"""
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from ..utils import console, print_success, print_error, create_progress

app = typer.Typer(help="Project management commands")

def validate_project_name(name: str) -> bool:
    """Validate that the project name is valid."""
    if not name.isidentifier():
        print_error(f"Project name '{name}' is not a valid Python package name.")
        print_error("Please use only alphanumeric characters and underscores.")
        return False
    return True

@app.command("new")
def new_project(name: str) -> None:
    """Create a new Eagle project.
    
    Args:
        name: The name of the project directory to create
    """
    if not validate_project_name(name):
        raise typer.Exit(1)
        
    project_dir = Path(name).absolute()
    
    if project_dir.exists():
        print_error(f"Directory '{name}' already exists!")
        raise typer.Exit(1)
    
    with create_progress() as progress:
        task = progress.add_task(description=f"Creating project '{name}'...", total=None)
        
        # Create project structure
        (project_dir / "app").mkdir(parents=True)
        (project_dir / "app" / "api").mkdir()
        (project_dir / "app" / "core").mkdir()
        (project_dir / "app" / "models").mkdir()
        (project_dir / "app" / "schemas").mkdir()
        (project_dir / "app" / "services").mkdir()
        (project_dir / "tests").mkdir()
        
        # Create basic files
        try:
            # Debug: Print project directory
            console.print(f"Creating files in: {project_dir}", style="yellow")
            
            # Debug: Check if directory exists and is writable
            if not project_dir.exists():
                console.print("Project directory does not exist!", style="red")
            elif not os.access(project_dir, os.W_OK):
                console.print("No write permission in project directory!", style="red")
                
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
            # Create files with error handling
            files_to_create = [
                (".gitignore", gitignore_content),
                ("README.md", f"# {name}\n\nAn Eagle Framework application.\n"),
                ("requirements.txt", "eagle-framework>=0.1.0\n"),
                (".env.example", 
                    "# Database\nDATABASE_URL=sqlite+aiosqlite:///./app.db\n\n"
                    "# Security\nSECRET_KEY=your-secret-key-here\nALGORITHM=HS256\nACCESS_TOKEN_EXPIRE_MINUTES=30\n"
                )
            ]
            
            for filename, content in files_to_create:
                filepath = project_dir / filename
                try:
                    filepath.write_text(content)
                    console.print(f"✓ Created {filepath}", style="green")
                except Exception as e:
                    console.print(f"✗ Failed to create {filepath}: {str(e)}", style="red")
                    
        except Exception as e:
            console.print(f"\nError creating project files: {str(e)}", style="red")
            if hasattr(e, 'errno'):
                console.print(f"Error code: {e.errno}", style="red")
            if hasattr(e, 'strerror'):
                console.print(f"Error message: {e.strerror}", style="red")
            raise
        
        # Create main.py
        (project_dir / "app" / "main.py").write_text(
            '"""\nMain application module.\n"""\n'
            'from eagleapi import create_app\n\n'
            'app = create_app()\n\n'
            '@app.get("/")\n'
            'async def read_root():\n'
            '    return {"message": "Welcome to Eagle Framework!"}\n'
        )

        # Create models.py
        (project_dir / "app" / "models" / "model.py").write_text(
            '"""\nModels for the application.\n"""\n'
            'from eagleapi.auth import AuthUser\n\n'
            'class User(AuthUser):\n'
            '    __table_args__ = {"extend_existing": True}\n'
            '    profile_picture: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)\n'
            '    phone_number: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)\n'
            '    date_of_birth: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)\n'
            '    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)\n'
            '    def __repr__(self) -> str:\n'
            '        return f"<User {self.username}>"\n'
        )
        
        # Create __init__.py files
        for dir_path in [
            "app", "app/api", "app/core", "app/models", 
            "app/schemas", "app/services", "tests"
        ]:
            (project_dir / dir_path / "__init__.py").touch()
        
        progress.update(task, completed=1)
    
    print_success(f"Created project '{name}'")
    console.print("\nTo get started, run:")
    console.print(f"\n  cd {name}")
    console.print("  python -m venv venv")
    console.print("  .\\venv\\Scripts\\activate  # On Windows")
    console.print("  pip install -r requirements.txt")
    console.print("  cp .env.example .env")
    console.print("  python -m uvicorn app.main:app --reload\n")
    console.print("Then open http://localhost:8000 in your browser.")
