# migrations.py
"""
Eagle Migration System - Professional Alembic Wrapper

A minimal, production-ready database migration system that integrates with Eagle's
database module. Provides automatic migration generation, version management,
and seamless integration with the existing SQLAlchemy setup.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Callable
from pathlib import Path
import os
import sys
import logging
import asyncio
from datetime import datetime
from dataclasses import dataclass
from eagleapi.config import settings
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from alembic.autogenerate import compare_metadata
from sqlalchemy import create_engine





logger = logging.getLogger(__name__)

@dataclass
class MigrationInfo:
    """Migration information container"""
    revision: str
    description: str
    created_at: datetime
    is_head: bool = False
    is_current: bool = False

class DatabaseError(Exception):
    """Base database exception with context"""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.context = context or {}

class MigrationError(DatabaseError):
    """Migration-specific errors"""
    pass



class MigrationManager:
    """Professional migration manager with Alembic integration"""
    
    def __init__(
        self, 
        migrations_dir: Optional[Union[str, Path]] = None,
        database_url: Optional[str] = None
    ):
        self.migrations_dir = Path(migrations_dir or "migrations")
        self.database_url = database_url or settings.DATABASE_URL
        self.config: Optional[Config] = None
        self.script_dir: Optional[ScriptDirectory] = None
        self._sync_engine = None
        self._logger = logging.getLogger(__name__)
        
        # Convert async URL to sync URL for Alembic
        self.sync_database_url = self._convert_to_sync_url(self.database_url)
        
    def _convert_to_sync_url(self, async_url: str) -> str:
        """Convert async database URL to sync URL for Alembic"""
        url_mappings = {
            'postgresql+asyncpg://': 'postgresql+psycopg2://',
            'mysql+aiomysql://': 'mysql+pymysql://',
            'sqlite+aiosqlite://': 'sqlite://'
        }
        
        for async_prefix, sync_prefix in url_mappings.items():
            if async_url.startswith(async_prefix):
                return async_url.replace(async_prefix, sync_prefix)
        
        # Fallback - might work for some databases
        return async_url.replace('+asyncpg', '+psycopg2').replace('+aiomysql', '+pymysql').replace('+aiosqlite', '')
    
    def _get_alembic_config(self) -> Config:
        """Get or create Alembic configuration"""
        if self.config is None:
            # Create config programmatically
            config = Config()
            config.set_main_option('script_location', str(self.migrations_dir))
            config.set_main_option('sqlalchemy.url', self.sync_database_url)
            config.set_main_option('file_template', '%%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s')
            
            # Configure logging
            config.set_main_option('logger.alembic', 'INFO')
            
            self.config = config
            
        return self.config
    
    def _get_script_directory(self) -> ScriptDirectory:
        """Get or create script directory"""
        if self.script_dir is None:
            config = self._get_alembic_config()
            self.script_dir = ScriptDirectory.from_config(config)
        return self.script_dir
    
    def _get_sync_engine(self):
        """Get synchronous engine for Alembic operations"""
        if self._sync_engine is None:
            print("Creating sync engine...",self.sync_database_url)
            self._sync_engine = create_engine(self.sync_database_url)
        return self._sync_engine
    
    def init(self, description: str = "Initial migration") -> bool:
        """
        Initialize migrations directory and create initial migration.
        
        This sets up the complete Alembic environment with all necessary files:
        - Creates migrations directory structure
        - Sets up env.py with proper configuration
        - Creates script.py.mako template for new migrations
        - Creates README with basic instructions
        - Creates an initial migration
        
        Args:
            description: Description for the initial migration
            
        Returns:
            bool: True if initialization was successful
            
        Raises:
            MigrationError: If initialization fails
        """
        try:
            # Ensure migrations_dir is a Path object
            if not isinstance(self.migrations_dir, Path):
                self.migrations_dir = Path(self.migrations_dir)
                
            self._logger.info(f"Initializing migrations in: {self.migrations_dir.absolute()}")
            
            # Create migrations directory if it doesn't exist
            self.migrations_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if directory is empty
            if any(self.migrations_dir.iterdir()):
                self._logger.warning(f"Migrations directory not empty: {self.migrations_dir}")
                return False
            
            # Create migrations directory and versions subdirectory
            self.migrations_dir.mkdir(parents=True, exist_ok=True)
            versions_dir = self.migrations_dir / 'versions'
            versions_dir.mkdir(exist_ok=True)
            
            # Create __init__.py in versions directory
            (versions_dir / '__init__.py').touch(exist_ok=True)
            
            # Create env.py with proper configuration
            env_py = self.migrations_dir / 'env.py'
            env_content = '''
# Alembic environment configuration

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import sys
import os

# Add the project root to the Python path
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

# Import your models here to ensure they are registered with SQLAlchemy
from eagleapi.db import Base

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode.
    
    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.
    
    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode.
    
    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
    '''
            env_py.write_text(env_content)
            
            # Create script.py.mako template - FIXED VERSION
            script_mako = self.migrations_dir / 'script.py.mako'
            script_content = '''"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
    '''
            script_mako.write_text(script_content)
            
            # Create README
            readme = self.migrations_dir / 'README'
            readme_content = """Migrations Directory
    ==================

    This directory contains database migration scripts managed by Alembic.

    - `versions/` - Contains individual migration scripts
    - `env.py` - Main migration environment configuration
    - `script.py.mako` - Template for new migration files

    To create a new migration:
        python -m alembic revision --autogenerate -m "description"

    To upgrade database:
        python -m alembic upgrade head

    To downgrade database:
        python -m alembic downgrade -1
    """
            readme.write_text(readme_content)
            
            # Create initial migration
            self._logger.info("Creating initial migration...")
            self.create_migration(description, auto=True)
            
            self._logger.info(f"Successfully initialized migrations in {self.migrations_dir}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize migrations: {e}", exc_info=True)
            raise MigrationError(
                f"Migration initialization failed: {e}",
                context={"migrations_dir": str(self.migrations_dir)}
            ) 
    def create_migration(
        self, 
        message: str, 
        auto: bool = True,
        head: str = "head",
        **kwargs
    ) -> Optional[str]:
        """Create a new migration"""
        try:
            config = self._get_alembic_config()
            
            if auto:
                # Auto-generate migration by comparing metadata
                revision = command.revision(
                    config, 
                    message=message,
                    autogenerate=True,
                    head=head,
                    **kwargs
                )
            else:
                # Create empty migration
                revision = command.revision(
                    config,
                    message=message,
                    head=head,
                    **kwargs
                )
            
            # Extract revision ID from the returned script
            script_dir = self._get_script_directory()
            revision_id = script_dir.get_current_head()

            self.upgrade()
            
            self._logger.info(f"Created migration: {revision_id} - {message}")
            return revision_id
            
        except Exception as e:
            self._logger.error(f"Failed to create migration: {e}")
            raise MigrationError(f"Migration creation failed: {e}")
    
    def upgrade(self, revision: str = "head") -> None:
        """Upgrade database to a specific revision"""
        try:
            config = self._get_alembic_config()
            command.upgrade(config, revision)
            self._logger.info(f"Upgraded database to revision: {revision}")
            
        except Exception as e:
            self._logger.error(f"Failed to upgrade database: {e}")
            raise MigrationError(f"Database upgrade failed: {e}")
    
    def downgrade(self, revision: str) -> None:
        """Downgrade database to a specific revision"""
        try:
            config = self._get_alembic_config()
            command.downgrade(config, revision)
            self._logger.info(f"Downgraded database to revision: {revision}")
            
        except Exception as e:
            self._logger.error(f"Failed to downgrade database: {e}")
            raise MigrationError(f"Database downgrade failed: {e}")
    
    def current(self) -> Optional[str]:
        """Get current database revision"""
        try:
            engine = self._get_sync_engine()
            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                return current_rev
                
        except Exception as e:
            self._logger.error(f"Failed to get current revision: {e}")
            return None
    
    def history(self, verbose: bool = False) -> List[MigrationInfo]:
        """Get migration history"""
        try:
            script_dir = self._get_script_directory()
            current_rev = self.current()
            head_rev = script_dir.get_current_head()
            
            revisions = []
            for script in script_dir.walk_revisions():
                migration_info = MigrationInfo(
                    revision=script.revision,
                    description=script.doc or "No description",
                    created_at=datetime.fromtimestamp(os.path.getctime(script.path)),
                    is_head=(script.revision == head_rev),
                    is_current=(script.revision == current_rev)
                )
                revisions.append(migration_info)
            
            return revisions
            
        except Exception as e:
            self._logger.error(f"Failed to get migration history: {e}")
            raise MigrationError(f"Failed to get migration history: {e}")
    
    def status(self) -> Dict[str, Any]:
        """Get detailed migration status"""
        try:
            current_rev = self.current()
            script_dir = self._get_script_directory()
            head_rev = script_dir.get_current_head()
            
            # Check if there are pending migrations
            if current_rev is None:
                pending_count = len(list(script_dir.walk_revisions()))
            else:
                pending_count = len(list(script_dir.walk_revisions(current_rev, head_rev)))
            
            return {
                "current_revision": current_rev,
                "head_revision": head_rev,
                "pending_migrations": pending_count,
                "is_up_to_date": current_rev == head_rev,
                "database_url": self.sync_database_url,
                "migrations_dir": str(self.migrations_dir)
            }
            
        except Exception as e:
            self._logger.error(f"Failed to get migration status: {e}")
            raise MigrationError(f"Failed to get migration status: {e}")
    
    def check_pending(self) -> bool:
        """Check if there are pending migrations"""
        try:
            status = self.status()
            return status["pending_migrations"] > 0
            
        except Exception:
            return False
    
    def stamp(self, revision: str) -> None:
        """Stamp database with a specific revision without running migrations"""
        try:
            config = self._get_alembic_config()
            command.stamp(config, revision)
            self._logger.info(f"Stamped database with revision: {revision}")
            
        except Exception as e:
            self._logger.error(f"Failed to stamp database: {e}")
            raise MigrationError(f"Database stamp failed: {e}")
    
    async def auto_upgrade(self) -> bool:
        """Automatically upgrade database if there are pending migrations"""
        try:
            if self.check_pending():
                self._logger.info("Pending migrations detected, upgrading...")
                await asyncio.get_event_loop().run_in_executor(
                    None, self.upgrade, "head"
                    )
                print("Upgrading database... pending migrations")
                return True
            else:
                self._logger.info("Database is up to date")
                return False
                
        except Exception as e:
            self._logger.error(f"Auto-upgrade failed: {e}")
            raise MigrationError(f"Auto-upgrade failed: {e}")
    
    def validate(self) -> bool:
        """Validate current database schema against models"""
        try:
            engine = self._get_sync_engine()
            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                
                # Compare current database schema with models
                diff = compare_metadata(context, Base.metadata)
                
                if diff:
                    self._logger.warning("Database schema differs from models")
                    for change in diff:
                        self._logger.warning(f"  - {change}")
                    return False
                else:
                    self._logger.info("Database schema is in sync with models")
                    return True
                    
        except Exception as e:
            self._logger.error(f"Schema validation failed: {e}")
            return False

# Global migration manager instance
migration_manager = MigrationManager()

# Convenience functions
def init_migrations(description: str = "Initial migration") -> bool:
    """Initialize migrations directory"""
    return migration_manager.init(description)

def create_migration(message: str, auto: bool = True) -> Optional[str]:
    """Create a new migration"""
    return migration_manager.create_migration(message, auto=auto)

def upgrade_database(revision: str = "head") -> None:
    """Upgrade database to specific revision"""
    migration_manager.upgrade(revision)

def downgrade_database(revision: str) -> None:
    """Downgrade database to specific revision"""
    migration_manager.downgrade(revision)

def get_migration_status() -> Dict[str, Any]:
    """Get current migration status"""
    return migration_manager.status()

def get_migration_history() -> List[MigrationInfo]:
    """Get migration history"""
    return migration_manager.history()

async def auto_upgrade_database() -> bool:
    """Automatically upgrade database if needed"""
    return await migration_manager.auto_upgrade()