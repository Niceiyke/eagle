"""
Database module for Eagle Framework.

Provides SQLAlchemy integration, async database support, and model utilities.
"""
from typing import Any, AsyncGenerator, Optional, Type, TypeVar, Generic
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, declared_attr, relationship
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, func
from datetime import datetime
import os

from ..core.config import settings

# Import models to ensure they are registered with SQLAlchemy
from .user_model import User  # noqa: F401

ModelType = TypeVar("ModelType", bound="BaseModel")

class Base(DeclarativeBase):
    """Base class for all database models."""
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate __tablename__ automatically."""
        return cls.__name__.lower()


class BaseModel(Base):
    """Base model with common fields and methods."""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    @classmethod
    async def get(cls, db: AsyncSession, id: int) -> Optional[ModelType]:
        """Get a model instance by ID."""
        return await db.get(cls, id)
    
    async def save(self, db: AsyncSession) -> None:
        """Save the current instance."""
        db.add(self)
        await db.commit()
        await db.refresh(self)
    
    async def delete(self, db: AsyncSession) -> None:
        """Delete the current instance."""
        await db.delete(self)
        await db.commit()


class Database:
    """Database connection and session management."""
    
    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./eagle.db")
        self.engine = create_async_engine(
            self.url,
            echo=True,
            future=True
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_all(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_all(self) -> None:
        """Drop all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# Database instance
db = Database(settings.DATABASE_URL)

# Dependency to get DB session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that provides a database session."""
    async with db.session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_models() -> None:
    """Initialize database models by creating all tables."""
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Export common database components
__all__ = [
    'Base', 'BaseModel', 'Database', 'db', 'get_db', 'init_models', 'Column', 
    'Integer', 'String', 'DateTime', 'Boolean', 'ForeignKey', 'relationship', 
    'func', 'select', 'update', 'delete'
]

# Import SQLAlchemy types for easier access
from sqlalchemy import String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import select, update, delete
