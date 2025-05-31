"""
Database management commands for Eagle Framework.

Provides CLI commands for database initialization, migrations, and management.
"""
import click
from typing import Optional
from pathlib import Path
import os
import sys

from eagleapi.db.migrations import Migrations
from eagleapi.db import db

@click.group()
def db_cli():
    """Database management commands."""
    pass

@db_cli.command()
@click.option('--url', help='Database URL. If not provided, uses EAGLE_DATABASE_URL or default SQLite URL.')
def init(url: Optional[str] = None):
    """Initialize database and migrations."""
    migrations = Migrations(db_url=url)
    try:
        # Create migrations directory structure
        migrations.init()
        click.echo("✓ Database migrations initialized")
        
        # Create tables
        import asyncio
        async def create_tables():
            await db.create_tables()
        asyncio.run(create_tables())
        click.echo("✓ Database tables created")
        
        # Create initial migration
        result = migrations.create_migration("Initial migration")
        if result.startswith("Error"):
            click.echo(f"⚠ {result}")
        else:
            click.echo("✓ Initial migration created")
            
        click.echo("\nDatabase setup complete!")
        
    except Exception as e:
        click.echo(f"Error initializing database: {str(e)}", err=True)
        sys.exit(1)

@db_cli.command()
@click.option('--message', '-m', help='Migration message', default='auto-generated')
@click.option('--url', help='Database URL. If not provided, uses EAGLE_DATABASE_URL or default SQLite URL.')
def migrate(message: str, url: Optional[str] = None):
    """Create and apply a new migration."""
    migrations = Migrations(db_url=url)
    
    # Create migration
    click.echo("Creating migration...")
    result = migrations.create_migration(message)
    if result.startswith("Error"):
        click.echo(f"Error: {result}", err=True)
        sys.exit(1)
    
    # Apply migration
    click.echo("Applying migration...")
    result = migrations.upgrade()
    if result.startswith("Error"):
        click.echo(f"Error: {result}", err=True)
        sys.exit(1)
    
    click.echo("✓ Migration created and applied successfully")

@db_cli.command()
@click.option('--url', help='Database URL. If not provided, uses EAGLE_DATABASE_URL or default SQLite URL.')
def upgrade(url: Optional[str] = None):
    """Upgrade database to the latest migration."""
    migrations = Migrations(db_url=url)
    result = migrations.upgrade()
    if result.startswith("Error"):
        click.echo(f"Error: {result}", err=True)
        sys.exit(1)
    click.echo("✓ Database upgraded to latest version")

@db_cli.command()
@click.argument('revision')
@click.option('--url', help='Database URL. If not provided, uses EAGLE_DATABASE_URL or default SQLite URL.')
def downgrade(revision: str, url: Optional[str] = None):
    """Downgrade database to a specific revision."""
    if not click.confirm(f"Are you sure you want to downgrade to revision {revision}?"):
        return
    
    migrations = Migrations(db_url=url)
    result = migrations.downgrade(revision)
    if result.startswith("Error"):
        click.echo(f"Error: {result}", err=True)
        sys.exit(1)
    click.echo(f"✓ Database downgraded to revision {revision}")

@db_cli.command()
@click.option('--url', help='Database URL. If not provided, uses EAGLE_DATABASE_URL or default SQLite URL.')
def current(url: Optional[str] = None):
    """Show current database revision."""
    migrations = Migrations(db_url=url)
    click.echo("Current database revision:")
    migrations.current()

@db_cli.command()
@click.option('--url', help='Database URL. If not provided, uses EAGLE_DATABASE_URL or default SQLite URL.')
def history(url: Optional[str] = None):
    """Show migration history."""
    migrations = Migrations(db_url=url)
    click.echo("Migration history:")
    migrations.history()

# Register the db command group with the main CLI
from ..cli import cli as main_cli
main_cli.add_command(db_cli, name="db")

# Add help text for the db command
db_cli.help = "Database management commands (create, migrate, upgrade, etc.)"
