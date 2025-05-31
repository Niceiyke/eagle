"""
Eagle Database Module

Provides a simple, powerful database interface with:
- Async SQLAlchemy 2.0+ support
- Automatic session management
- Built-in model base classes
- Easy migration support
"""
from __future__ import annotations
from typing import AsyncGenerator, Optional,TypeVar
from datetime import datetime
import os

from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import (
    DeclarativeBase, 
    declared_attr, 
    Mapped, 
    mapped_column,
    sessionmaker
)
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, 
    ForeignKey, func, select, update, delete
)
from sqlalchemy.orm import relationship

# Type variables for better type hints
ModelType = TypeVar("ModelType", bound="BaseModel")

class Database:
    """Database connection and session management."""
    
    def __init__(self, database_url: Optional[str] = None, echo_sql: bool = False):
        self.database_url = database_url or os.getenv("EAGLE_DATABASE_URL", "sqlite+aiosqlite:///./eagle.db")
        self.echo_sql = echo_sql or os.getenv("EAGLE_ECHO_SQL", "").lower() in ("1", "true", "yes")
        self.engine: Optional[AsyncEngine] = None
        self.session_factory = None
        self._setup_engine()
    
    def _setup_engine(self) -> None:
        """Initialize the database engine and session factory."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Initializing database connection to {self.database_url}")
            
            # Configure connection arguments based on database type
            connect_args = {}
            if "sqlite" in self.database_url:
                connect_args["check_same_thread"] = False
            
            # Create the async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=self.echo_sql,
                future=True,
                pool_pre_ping=True,  # Enable connection health checks
                pool_recycle=3600,    # Recycle connections after 1 hour
                pool_size=20,         # Maximum number of connections in the pool
                max_overflow=10,      # Maximum overflow connections
                connect_args=connect_args
            )
            
            # Create the session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False
            )
            
            logger.info("Database connection pool initialized")
            
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}", exc_info=True)
            self.engine = None
            self.session_factory = None
            raise
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a new database session.
        
        Yields:
            AsyncSession: A new database session
            
        Raises:
            RuntimeError: If the database is not properly initialized
            Exception: Any database errors that occur during the session
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not self.session_factory:
            error_msg = "Database session factory is not initialized. Please check your database configuration."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        session = self.session_factory()
        try:
            logger.debug("Yielding new database session")
            yield session
            await session.commit()
            logger.debug("Database session committed successfully")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error occurred: {e}", exc_info=True)
            raise
            
        finally:
            try:
                await session.close()
                logger.debug("Database session closed")
            except Exception as e:
                logger.error(f"Error closing database session: {e}", exc_info=True)
    
    async def create_tables(self) -> None:
        """
        Create all database tables.
        
        Raises:
            RuntimeError: If the database engine is not properly initialized
            Exception: If there's an error during table creation
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not self.engine:
            error_msg = "Database engine is not initialized. Cannot create tables."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            logger.info("Creating database tables...")
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}", exc_info=True)
            raise
    
    async def drop_tables(self) -> None:
        """
        Drop all database tables.
        
        WARNING: This will permanently delete all data in the database!
        
        Raises:
            RuntimeError: If the database engine is not properly initialized
            Exception: If there's an error during table deletion
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not self.engine:
            error_msg = "Database engine is not initialized. Cannot drop tables."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            logger.warning("Dropping all database tables...")
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.warning("All database tables have been dropped")
            
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}", exc_info=True)
            raise

# Global database instance
db = Database()

# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.
    
    Yields:
        AsyncSession: A database session
        
    Example:
        ```python
        from fastapi import Depends
        from .db import get_db, AsyncSession
        
        @app.get("/items/")
        async def read_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
        ```
    """
    if not db.session_factory:
        raise RuntimeError("Database not initialized. Call db.setup() first.")
        
    session = db.session_factory()
    try:
        yield session
    except Exception as e:
        await session.rollback()
        raise
    finally:
        await session.close()

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower()


class BaseModel(Base):
    """Base model with common fields and methods."""
    __abstract__ = True
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    
    @classmethod
    async def get(cls, id: int) -> Optional[ModelType]:
        """Get a model instance by ID."""
        async with db.get_session() as session:
            result = await session.get(cls, id)
            return result
    
    async def save(self) -> None:
        """Save the current instance."""
        async with db.get_session() as session:
            session.add(self)
            await session.commit()
            await session.refresh(self)
    
    async def delete(self) -> None:
        """Delete the current instance."""
        async with db.get_session() as session:
            await session.delete(self)
            await session.commit()

# Export common components
# Import all models to ensure they are registered with SQLAlchemy


__all__ = [
    # Core database classes
    'Database', 'db', 'BaseModel',
    
    # SQLAlchemy core components
    'Column', 'Integer', 'String', 'DateTime', 'Boolean', 
    'ForeignKey', 'relationship', 'select', 'update', 'delete',
    'func', 'declared_attr', 'sessionmaker', 'AsyncSession',
    
    # Type hints and utilities
    'ModelType', 'Mapped', 'mapped_column',
    
    # Session management
    'get_db', 'async_sessionmaker'
]
