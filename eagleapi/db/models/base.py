"""
Base database models and utilities.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, TypeVar
from datetime import datetime
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower() + 's'
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    def to_dict(self, exclude: Optional[set[str]] = None) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        exclude = exclude or set()
        return {
            c.name: getattr(self, c.name)
            for c in sa_inspect(self).mapper.column_attrs
            if c.key not in exclude
        }
    
    def __repr__(self) -> str:
        """String representation of the model instance."""
        attrs = [
            f"{k}={v!r}" 
            for k, v in self.to_dict().items()
            if not k.startswith('_')
        ]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

# Type variable for model classes
T = TypeVar('T', bound=Base)

class TimestampMixin:
    """Mixin that adds timestamp fields to models."""
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

class SoftDeleteMixin:
    """Mixin that adds soft delete functionality to models."""
    is_deleted: Mapped[bool] = mapped_column(default=False, nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(default=None, nullable=True)
    
    async def soft_delete(self, session) -> None:
        """Mark the record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        session.add(self)
        await session.flush()
    
    @classmethod
    def filter_not_deleted(cls, query):
        """Filter out soft-deleted records."""
        return query.filter(
            (getattr(cls, 'is_deleted') == False) |  # noqa: E712
            (getattr(cls, 'is_deleted').is_(None))
        )
