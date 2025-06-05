"""
management.py - CLI commands for Eagle framework
"""
import click
import asyncio
from pathlib import Path
from ..db.migrations import migration_manager

@click.group()
def cli():
    """Eagle Framework Management Commands"""
    pass

@cli.group()
def migrate():
    """Database migration commands"""
    pass

@migrate.command()
@click.option('--message', '-m', required=True, help='Migration description')
@click.option('--auto/--no-auto', default=True, help='Auto-generate migration')
def create(message: str, auto: bool):
    """Create a new migration"""
    try:
        revision = migration_manager.create_migration(message, auto=auto)
        click.echo(f"✅ Created migration: {revision}")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

@migrate.command()
@click.option('--revision', '-r', default='head', help='Target revision')
def upgrade(revision: str):
    """Upgrade database to revision"""
    try:
        migration_manager.upgrade(revision)
        click.echo(f"✅ Upgraded to: {revision}")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

@migrate.command()
@click.option('--revision', '-r', required=True, help='Target revision')
def downgrade(revision: str):
    """Downgrade database to revision"""
    try:
        migration_manager.downgrade(revision)
        click.echo(f"✅ Downgraded to: {revision}")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

@migrate.command()
def status():
    """Show migration status"""
    try:
        status = migration_manager.status()
        click.echo("🔍 Migration Status:")
        click.echo(f"  Current: {status['current_revision'] or 'None'}")
        click.echo(f"  Head: {status['head_revision'] or 'None'}")
        click.echo(f"  Pending: {status['pending_migrations']}")
        click.echo(f"  Up to date: {'✅' if status['is_up_to_date'] else '❌'}")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

@migrate.command()
def history():
    """Show migration history"""
    try:
        history = migration_manager.history()
        click.echo("📚 Migration History:")
        for migration in history:
            status_icons = []
            if migration.is_current:
                status_icons.append("👉")
            if migration.is_head:
                status_icons.append("🔝")
            
            status_str = " ".join(status_icons)
            click.echo(f"  {migration.revision}: {migration.description} {status_str}")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

@migrate.command()
@click.option('--description', '-d', default='Initial migration', help='Initial migration description')
def init(description: str):
    """Initialize migrations"""
    try:
        if migration_manager.init(description):
            click.echo("✅ Migrations initialized successfully")
        else:
            click.echo("⚠️  Migrations already initialized")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

@migrate.command()
def validate():
    """Validate database schema against models"""
    try:
        is_valid = migration_manager.validate()
        if is_valid:
            click.echo("✅ Database schema is valid")
        else:
            click.echo("❌ Database schema validation failed")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

if __name__ == '__main__':
    cli()