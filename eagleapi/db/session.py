"""
Database session management and connection handling.
"""
from __future__ import annotations
import os
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy import text

from .exceptions import ConnectionError, DatabaseError

class Database:
    """Database connection and session management."""
    
    def __init__(
        self, 
        database_url: Optional[str] = None, 
        **kwargs: Any
    ) -> None:
        """Initialize the database connection.
        
        Args:
            database_url: Database connection URL. If not provided, will use
                         EAGLE_DATABASE_URL environment variable or default to SQLite.
            **kwargs: Additional keyword arguments passed to create_async_engine.
        """
        from eagleapi.core.config import settings
        self.database_url = database_url or settings.DATABASE_URL
        self.echo_sql = bool(kwargs.pop('echo_sql', False) or settings.ECHO_SQL)
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._logger = logging.getLogger(__name__)
        
        # Initialize the engine
        self._setup_engine(**kwargs)
    
    def _setup_engine(self, **kwargs: Any) -> None:
        """Set up the SQLAlchemy async engine."""
        if not self.database_url:
            raise ValueError("Database URL is required")
        
        # Configure engine options
        engine_options: Dict[str, Any] = {
            "echo": self.echo_sql,
            "future": True,
            "pool_pre_ping": True,
            "pool_recycle": 300,  # Recycle connections after 5 minutes
            **kwargs
        }
        
        # SQLite specific options
        if "sqlite" in self.database_url:
            engine_options.update({
                "connect_args": {"check_same_thread": False},
                "poolclass": NullPool
            })
        
        try:
            self.engine = create_async_engine(
                self.database_url,
                **engine_options
            )
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                class_=AsyncSession
            )
            self._logger.info("Database engine initialized successfully")
        except Exception as e:
            self._logger.error(f"Failed to initialize database engine: {e}")
            raise ConnectionError(
                f"Failed to connect to database: {e}",
                context={"database_url": self._obfuscate_url(self.database_url)}
            )
    
    @staticmethod
    def _obfuscate_url(url: str) -> str:
        """Obfuscate sensitive information in database URLs for logging."""
        if not url:
            return ""
        
        if "@" in url:
            # Obfuscate username:password in URL
            parts = url.split("@", 1)
            auth_part = parts[0].split("//", 1)[-1]
            if ":" in auth_part:
                user_pass = auth_part.split(":", 1)
                obfuscated = f"{user_pass[0]}:****@"
                return url.replace(auth_part + "@", obfuscated)
        return url
    
    async def health_check(self) -> bool:
        """Check if the database is reachable."""
        if not self.engine:
            return False
            
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            self._logger.error(f"Database health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session with proper cleanup."""
        if not self.session_factory:
            raise ConnectionError("Database session factory not initialized")
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as e:
            await session.rollback()
            self._logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            await session.close()
    
    async def close(self) -> None:
        """Close all database connections."""
        if self.engine:
            await self.engine.dispose()
            self._logger.info("Database connections closed")

# Global database instance
db = Database()

# FastAPI dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get a database session."""
    if not db.session_factory:
        raise RuntimeError("Database not initialized")
    
    session = db.session_factory()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError as e:
        await session.rollback()
        raise
    finally:
        await session.close()
