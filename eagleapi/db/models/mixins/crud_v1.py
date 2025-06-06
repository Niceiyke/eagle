"""
CRUD (Create, Read, Update, Delete) operations mixin for SQLAlchemy models.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Type, TypeVar, cast
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ...types import PaginationResult, ColumnType

T = TypeVar('T', bound='CRUDMixin')

class CRUDMixin:
    """Mixin that adds CRUD operations to SQLAlchemy models."""
    
    @classmethod
    async def get(cls: Type[T], session: AsyncSession, id: int) -> Optional[T]:
        """Get a single record by ID."""
        result = await session.get(cls, id)
        return cast(Optional[T], result)
    
    @classmethod
    async def get_by(cls: Type[T], session: AsyncSession, **filters: Any) -> Optional[T]:
        """Get a single record by filters."""
        query = select(cls).filter_by(**filters).limit(1)
        result = await session.execute(query)
        return result.scalars().first()
    
    @classmethod
    async def get_all(
        cls: Type[T], 
        session: AsyncSession, 
        **filters: Any
    ) -> Sequence[T]:
        """Get all records matching the given filters."""
        query = select(cls)
        if filters:
            query = query.filter_by(**filters)
        result = await session.execute(query)
        return result.scalars().all()
    
    @classmethod
    async def get_many(
        cls: Type[T], 
        session: AsyncSession, 
        ids: Sequence[int]
    ) -> Sequence[T]:
        """Get multiple records by their IDs."""
        if not ids:
            return []
        query = select(cls).where(cls.id.in_(ids))  # type: ignore
        result = await session.execute(query)
        return result.scalars().all()
    
    @classmethod
    async def paginate(
        cls: Type[T],
        session: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        order_by: Optional[ColumnType] = None,
        **filters: Any
    ) -> PaginationResult[T]:
        """Paginate through records with optional filtering and ordering."""
        # Count total records
        count_query = select([func.count()]).select_from(cls)  # type: ignore
        if filters:
            count_query = count_query.filter_by(**filters)
        
        total = (await session.execute(count_query)).scalar() or 0
        
        # Calculate pagination
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1
        
        # Get paginated results
        query = select(cls)
        if filters:
            query = query.filter_by(**filters)
        
        if order_by is not None:
            if isinstance(order_by, str):
                order_by = getattr(cls, order_by)
            query = query.order_by(order_by)
        
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await session.execute(query)
        items = result.scalars().all()
        
        return PaginationResult(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev
        )
    
    @classmethod
    async def exists(cls, session: AsyncSession, **filters: Any) -> bool:
        """Check if a record exists matching the given filters."""
        query = select([1]).select_from(cls).filter_by(**filters).limit(1)  # type: ignore
        result = await session.execute(query)
        return result.scalar() is not None
    
    @classmethod
    async def count(cls, session: AsyncSession, **filters: Any) -> int:
        """Count records matching the given filters."""
        query = select([func.count()]).select_from(cls)  # type: ignore
        if filters:
            query = query.filter_by(**filters)
        result = await session.execute(query)
        return cast(int, result.scalar() or 0)
    
    @classmethod
    async def create(cls: Type[T], session: AsyncSession, **data: Any) -> T:
        """Create a new record."""
        instance = cls(**data)
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return cast(T, instance)
    
    @classmethod
    async def bulk_create(
        cls: Type[T],
        session: AsyncSession,
        data_list: Sequence[Dict[str, Any]]
    ) -> Sequence[T]:
        """Create multiple records in a single operation."""
        if not data_list:
            return []
            
        instances = [cls(**data) for data in data_list]
        session.add_all(instances)
        await session.flush()
        return cast(Sequence[T], instances)
    
    async def update(self, session: AsyncSession, **data: Any) -> None:
        """Update the current instance with new data."""
        for key, value in data.items():
            setattr(self, key, value)
        self.updated_at = datetime.utcnow()
        session.add(self)
        await session.flush()
    
    async def delete(self, session: AsyncSession) -> None:
        """Delete the current instance."""
        await session.delete(self)
        await session.flush()
    
    async def refresh(self, session: AsyncSession) -> None:
        """Refresh the current instance from the database."""
        await session.refresh(self)
    
    def to_dict(self, exclude: Optional[set[str]] = None) -> Dict[str, Any]:
        """Convert the instance to a dictionary."""
        exclude = exclude or set()
        return {
            c.name: getattr(self, c.name)
            for c in self.__table__.columns  # type: ignore
            if c.name not in exclude
        }
