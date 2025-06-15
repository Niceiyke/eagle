"""
Database migration commands.

This module is loaded on-demand when migration commands are invoked.
"""
import sys
import typer
from typing import Optional, cast

from ..utils import console, print_success, print_error, print_warning

# Create the command group
app = typer.Typer(help="Database migration commands")

def get_migrations(url: Optional[str] = None):
    """Initialize and return a Migrations instance.
    
    This is imported only when needed to avoid loading the full application.
    """
    from eagleapi.db.migrations import Migrations
    return Migrations(db_url=url)

@app.command("init")
def init_db(url: Optional[str] = None) -> None:
    """Initialize database and migrations."""
    migrations = get_migrations(url)
    
    try:
        # Create migrations directory structure
        migrations.init()
        print_success("Database migrations initialized")
        
        # Create tables
        import asyncio
        from eagleapi.db import db
        
        async def create_tables():
            await db.create_tables()
            
        asyncio.run(create_tables())
        print_success("Database tables created")
        
        # Create initial migration
        result = migrations.create_migration("Initial migration")
        if result.startswith("Error"):
            print_warning(result)
        else:
            print_success("Initial migration created")
            
        print_success("Database setup complete!")
        
    except Exception as e:
        print_error(f"Error initializing database: {str(e)}")
        sys.exit(1)

@app.command("create")
def create_migration(message: str, url: Optional[str] = None) -> None:
    """Create a new migration."""
    migrations = get_migrations(url)
    
    try:
        result = migrations.create_migration(message)
        if result.startswith("Error"):
            print_error(result)
            sys.exit(1)
        print_success(f"Created migration: {result}")
    except Exception as e:
        print_error(f"Error creating migration: {str(e)}")
        sys.exit(1)

@app.command("upgrade")
def upgrade_db(revision: str = "head", url: Optional[str] = None) -> None:
    """Upgrade database to a specific revision."""
    migrations = get_migrations(url)
    
    try:
        result = migrations.upgrade(revision)
        if result.startswith("Error"):
            print_error(result)
            sys.exit(1)
        print_success(f"Database upgraded to: {revision}")
    except Exception as e:
        print_error(f"Error upgrading database: {str(e)}")
        sys.exit(1)

@app.command("downgrade")
def downgrade_db(revision: str, url: Optional[str] = None) -> None:
    """Downgrade database to a specific revision."""
    if not typer.confirm(f"Are you sure you want to downgrade to revision {revision}?"):
        return
    
    migrations = get_migrations(url)
    
    try:
        result = migrations.downgrade(revision)
        if result.startswith("Error"):
            print_error(result)
            sys.exit(1)
        print_success(f"Database downgraded to revision: {revision}")
    except Exception as e:
        print_error(f"Error downgrading database: {str(e)}")
        sys.exit(1)

@app.command("status")
def migration_status(url: Optional[str] = None) -> None:
    """Show current migration status."""
    migrations = get_migrations(url)
    
    try:
        status = migrations.status()
        console.print("\n[bold]Migration Status:[/bold]")
        console.print(f"  Current: {status['current_revision'] or 'None'}")
        console.print(f"  Head: {status['head_revision'] or 'None'}")
        console.print(f"  Pending: {status['pending_migrations']}")
        
        if status['is_up_to_date']:
            print_success("  Database is up to date")
        else:
            print_warning("  Database is not up to date")
            
    except Exception as e:
        print_error(f"Error getting migration status: {str(e)}")
        sys.exit(1)

@app.command("history")
def migration_history(url: Optional[str] = None) -> None:
    """Show migration history."""
    migrations = get_migrations(url)
    
    try:
        history = migrations.history()
        console.print("\n[bold]Migration History:[/bold]")
        
        if not history:
            console.print("  No migrations found")
            return
            
        for migration in history:
            status = []
            if migration.is_current:
                status.append("current")
            if migration.is_head:
                status.append("head")
                
            status_str = f" [dim]({', '.join(status)})[/dim]" if status else ""
            console.print(f"  {migration.revision}: {migration.description}{status_str}")
            
    except Exception as e:
        print_error(f"Error getting migration history: {str(e)}")
        sys.exit(1)
