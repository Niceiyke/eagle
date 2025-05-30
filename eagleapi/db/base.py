"""
Base database models and utilities.
"""
from datetime import datetime
from typing import Any, Optional, Type, TypeVar

from sqlalchemy import Column, Integer, DateTime, select
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

ModelType = TypeVar("ModelType", bound="Base")

@as_declarative()
class Base:
    """Base class for all database models."""
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    
    @declared_attr
    def __tablename__(cls) -> str:
        """
        Generate table name automatically.
        Convert CamelCase class name to snake_case table name.
        """
        name = ''
        for i, char in enumerate(cls.__name__):
            if i > 0 and char.isupper() and not cls.__name__[i-1].isupper():
                name += '_' + char.lower()
            else:
                name += char.lower()
        return name
    
    @classmethod
    async def get(cls: Type[ModelType], db: AsyncSession, id: int) -> Optional[ModelType]:
        """Get a model instance by ID."""
        result = await db.execute(select(cls).filter(cls.id == id))
        return result.scalar_one_or_none()
    
    async def save(self, db: AsyncSession) -> None:
        """Save the current instance."""
        db.add(self)
        await db.commit()
        await db.refresh(self)
    
    async def delete(self, db: AsyncSession) -> None:
        """Delete the current instance."""
        await db.delete(self)
        await db.commit()
