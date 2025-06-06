"""
Enhanced CRUD (Create, Read, Update, Delete) operations mixin for SQLAlchemy models.
Production-ready with advanced features for FastAPI applications.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Type, TypeVar, Union, List, Callable
from datetime import datetime
from contextlib import asynccontextmanager
import logging
from sqlalchemy import select, func, desc, asc, and_, or_, text, inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.sql import Select
from enum import Enum

from ...types import PaginationResult, ColumnType
from ...exceptions import (
    RecordNotFoundError, 
    ValidationError, 
    ConflictError,
    DatabaseError
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='CRUDMixin')

class SortOrder(Enum):
    ASC = "asc"
    DESC = "desc"

class CRUDMixin:
    """Enhanced mixin that adds comprehensive CRUD operations to SQLAlchemy models."""
    
    # Class-level configuration
    _soft_delete_field: Optional[str] = "deleted_at"
    _timestamp_fields = {"created_at", "updated_at"}
    _audit_fields = {"created_by", "updated_by"}
    
    # ============================================================================
    # READ OPERATIONS
    # ============================================================================
    
    @classmethod
    async def get(
        cls: Type[T], 
        session: AsyncSession, 
        id: int,
        raise_not_found: bool = False,
        load_relationships: Optional[List[str]] = None
    ) -> Optional[T]:
        """Get a single record by ID with optional relationship loading."""
        try:
            query = select(cls).where(cls.id == id)
            
            # Apply relationship loading
            if load_relationships:
                for rel in load_relationships:
                    query = query.options(selectinload(getattr(cls, rel)))
            
            # Apply soft delete filter
            query = cls._apply_soft_delete_filter(query)
            
            result = await session.execute(query)
            record = result.scalars().first()
            
            if raise_not_found and record is None:
                raise RecordNotFoundError(f"{cls.__name__} with id {id} not found")
                
            return record
        except SQLAlchemyError as e:
            logger.error(f"Error getting {cls.__name__} by id {id}: {e}")
            raise DatabaseError(f"Failed to retrieve record: {str(e)}")
    
    @classmethod
    async def get_by(
        cls: Type[T], 
        session: AsyncSession, 
        raise_not_found: bool = False,
        load_relationships: Optional[List[str]] = None,
        **filters: Any
    ) -> Optional[T]:
        """Get a single record by filters with enhanced error handling."""
        try:
            query = select(cls)
            query = cls._apply_filters(query, **filters)
            query = cls._apply_soft_delete_filter(query)
            
            if load_relationships:
                for rel in load_relationships:
                    query = query.options(selectinload(getattr(cls, rel)))
            
            result = await session.execute(query)
            record = result.scalars().first()
            
            if raise_not_found and record is None:
                raise RecordNotFoundError(f"{cls.__name__} not found with filters: {filters}")
                
            return record
        except SQLAlchemyError as e:
            logger.error(f"Error getting {cls.__name__} by filters {filters}: {e}")
            raise DatabaseError(f"Failed to retrieve record: {str(e)}")
    
    @classmethod
    async def get_all(
        cls: Type[T], 
        session: AsyncSession,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        load_relationships: Optional[List[str]] = None,
        **filters: Any
    ) -> Sequence[T]:
        """Get all records with enhanced filtering and ordering."""
        try:
            query = select(cls)
            query = cls._apply_filters(query, **filters)
            query = cls._apply_soft_delete_filter(query)
            query = cls._apply_ordering(query, order_by)
            
            if load_relationships:
                for rel in load_relationships:
                    query = query.options(selectinload(getattr(cls, rel)))
            
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
                
            result = await session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {cls.__name__}: {e}")
            raise DatabaseError(f"Failed to retrieve records: {str(e)}")
    
    @classmethod
    async def get_many(
        cls: Type[T], 
        session: AsyncSession, 
        ids: Sequence[int],
        load_relationships: Optional[List[str]] = None,
        maintain_order: bool = False
    ) -> Sequence[T]:
        """Get multiple records by their IDs with order preservation option."""
        if not ids:
            return []
            
        try:
            query = select(cls).where(cls.id.in_(ids))
            query = cls._apply_soft_delete_filter(query)
            
            if load_relationships:
                for rel in load_relationships:
                    query = query.options(selectinload(getattr(cls, rel)))
            
            result = await session.execute(query)
            records = result.scalars().all()
            
            if maintain_order:
                # Preserve the order of input IDs
                id_to_record = {record.id: record for record in records}
                return [id_to_record[id] for id in ids if id in id_to_record]
            
            return records
        except SQLAlchemyError as e:
            logger.error(f"Error getting many {cls.__name__}: {e}")
            raise DatabaseError(f"Failed to retrieve records: {str(e)}")
    
    @classmethod
    async def search(
        cls: Type[T],
        session: AsyncSession,
        search_term: str,
        search_fields: List[str],
        limit: Optional[int] = None,
        **filters: Any
    ) -> Sequence[T]:
        """Full-text search across specified fields."""
        try:
            search_conditions = []
            for field in search_fields:
                if hasattr(cls, field):
                    column = getattr(cls, field)
                    search_conditions.append(column.ilike(f"%{search_term}%"))
            
            if not search_conditions:
                return []
            
            query = select(cls).where(or_(*search_conditions))
            query = cls._apply_filters(query, **filters)
            query = cls._apply_soft_delete_filter(query)
            
            if limit:
                query = query.limit(limit)
                
            result = await session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Error searching {cls.__name__}: {e}")
            raise DatabaseError(f"Search failed: {str(e)}")
    
    @classmethod
    async def paginate(
        cls: Type[T],
        session: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        order_by: Optional[Union[str, List[str]]] = None,
        load_relationships: Optional[List[str]] = None,
        **filters: Any
    ) -> PaginationResult[T]:
        """Enhanced pagination with better performance and features."""
        try:
            # Validate pagination parameters
            if page < 1:
                page = 1
            if page_size < 1:
                page_size = 20
            if page_size > 1000:  # Prevent excessive page sizes
                page_size = 1000
            
            # Build base query
            base_query = select(cls)
            base_query = cls._apply_filters(base_query, **filters)
            base_query = cls._apply_soft_delete_filter(base_query)
            
            # Count total records
            count_query = select(func.count()).select_from(base_query.subquery())
            total = (await session.execute(count_query)).scalar() or 0
            
            # Calculate pagination metadata
            total_pages = (total + page_size - 1) // page_size if page_size > 0 else 1
            has_next = page < total_pages
            has_prev = page > 1
            
            # Get paginated results
            query = base_query
            query = cls._apply_ordering(query, order_by)
            
            if load_relationships:
                for rel in load_relationships:
                    query = query.options(selectinload(getattr(cls, rel)))
            
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
        except SQLAlchemyError as e:
            logger.error(f"Error paginating {cls.__name__}: {e}")
            raise DatabaseError(f"Pagination failed: {str(e)}")
    
    @classmethod
    async def exists(
        cls, 
        session: AsyncSession, 
        **filters: Any
    ) -> bool:
        """Check if a record exists with optimized query."""
        try:
            query = select(1).select_from(cls)
            query = cls._apply_filters(query, **filters)
            query = cls._apply_soft_delete_filter(query)
            query = query.limit(1)
            
            result = await session.execute(query)
            return result.scalar() is not None
        except SQLAlchemyError as e:
            logger.error(f"Error checking existence of {cls.__name__}: {e}")
            raise DatabaseError(f"Existence check failed: {str(e)}")
    
    @classmethod
    async def count(
        cls, 
        session: AsyncSession, 
        **filters: Any
    ) -> int:
        """Count records with enhanced filtering."""
        try:
            query = select(func.count()).select_from(cls)
            query = cls._apply_filters(query, **filters)
            query = cls._apply_soft_delete_filter(query)
            
            result = await session.execute(query)
            return result.scalar() or 0
        except SQLAlchemyError as e:
            logger.error(f"Error counting {cls.__name__}: {e}")
            raise DatabaseError(f"Count failed: {str(e)}")
    
    # ============================================================================
    # CREATE OPERATIONS
    # ============================================================================
    
    @classmethod
    async def create(
        cls: Type[T], 
        session: AsyncSession,
        commit: bool = False,
        refresh: bool = True,
        user_id: Optional[int] = None,
        **data: Any
    ) -> T:
        """Create a new record with enhanced features."""
        try:
            # Apply pre-create validation
            cls._validate_create_data(data)
            
            # Add audit fields
            if user_id and "created_by" in cls._audit_fields:
                data["created_by"] = user_id
            
            # Add timestamp
            if "created_at" in cls._timestamp_fields:
                data["created_at"] = datetime.utcnow()
            
            instance = cls(**data)
            session.add(instance)
            
            await session.flush()
            
            if refresh:
                await session.refresh(instance)
            
            if commit:
                await session.commit()
                
            return instance
        except IntegrityError as e:
            await session.rollback()
            logger.error(f"Integrity error creating {cls.__name__}: {e}")
            raise ConflictError(f"Record creation failed due to constraint violation")
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Error creating {cls.__name__}: {e}")
            raise DatabaseError(f"Failed to create record: {str(e)}")
    
    @classmethod
    async def bulk_create(
        cls: Type[T],
        session: AsyncSession,
        data_list: Sequence[Dict[str, Any]],
        batch_size: int = 1000,
        commit: bool = False,
        user_id: Optional[int] = None
    ) -> Sequence[T]:
        """Create multiple records with batching for better performance."""
        if not data_list:
            return []
            
        try:
            all_instances = []
            current_time = datetime.utcnow()
            
            # Process in batches
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                batch_instances = []
                
                for data in batch:
                    # Apply validation and preprocessing
                    cls._validate_create_data(data)
                    
                    # Add audit fields
                    if user_id and "created_by" in cls._audit_fields:
                        data["created_by"] = user_id
                    
                    # Add timestamp
                    if "created_at" in cls._timestamp_fields:
                        data["created_at"] = current_time
                    
                    batch_instances.append(cls(**data))
                
                session.add_all(batch_instances)
                await session.flush()
                all_instances.extend(batch_instances)
            
            if commit:
                await session.commit()
                
            return all_instances
        except IntegrityError as e:
            await session.rollback()
            logger.error(f"Integrity error bulk creating {cls.__name__}: {e}")
            raise ConflictError(f"Bulk creation failed due to constraint violation")
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Error bulk creating {cls.__name__}: {e}")
            raise DatabaseError(f"Bulk creation failed: {str(e)}")
    
    @classmethod
    async def get_or_create(
        cls: Type[T],
        session: AsyncSession,
        defaults: Optional[Dict[str, Any]] = None,
        commit: bool = False,
        user_id: Optional[int] = None,
        **filters: Any
    ) -> tuple[T, bool]:
        """Get existing record or create new one. Returns (instance, created)."""
        try:
            # Try to get existing record
            instance = await cls.get_by(session, **filters)
            
            if instance is not None:
                return instance, False
            
            # Create new record
            create_data = {**filters}
            if defaults:
                create_data.update(defaults)
                
            instance = await cls.create(
                session, 
                commit=commit, 
                user_id=user_id,
                **create_data
            )
            return instance, True
        except SQLAlchemyError as e:
            logger.error(f"Error in get_or_create for {cls.__name__}: {e}")
            raise DatabaseError(f"Get or create failed: {str(e)}")
    
    # ============================================================================
    # UPDATE OPERATIONS
    # ============================================================================
    
    async def update(
        self, 
        session: AsyncSession,
        commit: bool = False,
        user_id: Optional[int] = None,
        **data: Any
    ) -> None:
        """Update the current instance with enhanced features."""
        try:
            # Apply pre-update validation
            self._validate_update_data(data)
            
            # Add audit fields
            if user_id and "updated_by" in self._audit_fields:
                data["updated_by"] = user_id
            
            # Add timestamp
            if "updated_at" in self._timestamp_fields:
                data["updated_at"] = datetime.utcnow()
            
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            session.add(self)
            await session.flush()
            
            if commit:
                await session.commit()
                
        except IntegrityError as e:
            await session.rollback()
            logger.error(f"Integrity error updating {self.__class__.__name__}: {e}")
            raise ConflictError(f"Update failed due to constraint violation")
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Error updating {self.__class__.__name__}: {e}")
            raise DatabaseError(f"Update failed: {str(e)}")
    
    @classmethod
    async def bulk_update(
        cls: Type[T],
        session: AsyncSession,
        updates: List[Dict[str, Any]],
        commit: bool = False,
        user_id: Optional[int] = None
    ) -> int:
        """Bulk update multiple records. Each dict should contain 'id' and update fields."""
        if not updates:
            return 0
            
        try:
            updated_count = 0
            current_time = datetime.utcnow()
            
            for update_data in updates:
                if 'id' not in update_data:
                    continue
                    
                record_id = update_data.pop('id')
                
                # Add audit fields
                if user_id and "updated_by" in cls._audit_fields:
                    update_data["updated_by"] = user_id
                
                # Add timestamp
                if "updated_at" in cls._timestamp_fields:
                    update_data["updated_at"] = current_time
                
                # Perform update
                stmt = (
                    cls.__table__.update()
                    .where(cls.id == record_id)
                    .values(**update_data)
                )
                
                result = await session.execute(stmt)
                updated_count += result.rowcount
            
            if commit:
                await session.commit()
                
            return updated_count
        except IntegrityError as e:
            await session.rollback()
            logger.error(f"Integrity error bulk updating {cls.__name__}: {e}")
            raise ConflictError(f"Bulk update failed due to constraint violation")
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Error bulk updating {cls.__name__}: {e}")
            raise DatabaseError(f"Bulk update failed: {str(e)}")
    
    # ============================================================================
    # DELETE OPERATIONS
    # ============================================================================
    
    async def delete(
        self, 
        session: AsyncSession,
        soft: Optional[bool] = None,
        commit: bool = False,
        user_id: Optional[int] = None
    ) -> None:
        """Delete the current instance with soft delete support."""
        try:
            # Determine if soft delete should be used
            use_soft_delete = (
                soft if soft is not None 
                else (self._soft_delete_field is not None and hasattr(self, self._soft_delete_field))
            )
            
            if use_soft_delete and self._soft_delete_field:
                # Soft delete
                update_data = {self._soft_delete_field: datetime.utcnow()}
                if user_id and "deleted_by" in self._audit_fields:
                    update_data["deleted_by"] = user_id
                    
                await self.update(session, commit=commit, **update_data)
            else:
                # Hard delete
                await session.delete(self)
                await session.flush()
                
                if commit:
                    await session.commit()
                    
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Error deleting {self.__class__.__name__}: {e}")
            raise DatabaseError(f"Delete failed: {str(e)}")
    
    @classmethod
    async def bulk_delete(
        cls: Type[T],
        session: AsyncSession,
        ids: Sequence[int],
        soft: Optional[bool] = None,
        commit: bool = False,
        user_id: Optional[int] = None
    ) -> int:
        """Bulk delete multiple records by IDs."""
        if not ids:
            return 0
            
        try:
            use_soft_delete = (
                soft if soft is not None 
                else (cls._soft_delete_field is not None)
            )
            
            if use_soft_delete and cls._soft_delete_field:
                # Soft delete
                update_data = {cls._soft_delete_field: datetime.utcnow()}
                if user_id and "deleted_by" in cls._audit_fields:
                    update_data["deleted_by"] = user_id
                
                stmt = (
                    cls.__table__.update()
                    .where(cls.id.in_(ids))
                    .values(**update_data)
                )
            else:
                # Hard delete
                stmt = cls.__table__.delete().where(cls.id.in_(ids))
            
            result = await session.execute(stmt)
            deleted_count = result.rowcount
            
            if commit:
                await session.commit()
                
            return deleted_count
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Error bulk deleting {cls.__name__}: {e}")
            raise DatabaseError(f"Bulk delete failed: {str(e)}")
    
    async def restore(
        self, 
        session: AsyncSession,
        commit: bool = False,
        user_id: Optional[int] = None
    ) -> None:
        """Restore a soft-deleted record."""
        if not self._soft_delete_field or not hasattr(self, self._soft_delete_field):
            raise ValueError("Soft delete not configured for this model")
            
        try:
            update_data = {self._soft_delete_field: None}
            if user_id and "restored_by" in self._audit_fields:
                update_data["restored_by"] = user_id
                
            await self.update(session, commit=commit, **update_data)
        except SQLAlchemyError as e:
            logger.error(f"Error restoring {self.__class__.__name__}: {e}")
            raise DatabaseError(f"Restore failed: {str(e)}")
    
    # ============================================================================
    # UTILITY & HELPER METHODS
    # ============================================================================
    
    async def refresh(self, session: AsyncSession) -> None:
        """Refresh the current instance from the database."""
        try:
            await session.refresh(self)
        except SQLAlchemyError as e:
            logger.error(f"Error refreshing {self.__class__.__name__}: {e}")
            raise DatabaseError(f"Refresh failed: {str(e)}")
    
    def to_dict(
        self, 
        exclude: Optional[set[str]] = None,
        include_relationships: bool = False,
        date_format: str = "iso"
    ) -> Dict[str, Any]:
        """Convert the instance to a dictionary with enhanced options."""
        exclude = exclude or set()
        result = {}
        
        # Include column attributes
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                
                # Format datetime objects
                if isinstance(value, datetime):
                    if date_format == "iso":
                        value = value.isoformat()
                    elif date_format == "timestamp":
                        value = value.timestamp()
                
                result[column.name] = value
        
        # Include relationships if requested
        if include_relationships:
            mapper = inspect(self.__class__)
            for rel in mapper.relationships:
                if rel.key not in exclude:
                    rel_value = getattr(self, rel.key, None)
                    if rel_value is not None:
                        if hasattr(rel_value, '__iter__') and not isinstance(rel_value, (str, bytes)):
                            # Collection relationship
                            result[rel.key] = [item.to_dict(exclude=exclude) if hasattr(item, 'to_dict') else str(item) for item in rel_value]
                        else:
                            # Single relationship
                            result[rel.key] = rel_value.to_dict(exclude=exclude) if hasattr(rel_value, 'to_dict') else str(rel_value)
        
        return result
    
    @classmethod
    def _apply_filters(cls, query: Select, **filters: Any) -> Select:
        """Apply filters to a query with advanced filtering support."""
        for key, value in filters.items():
            if not hasattr(cls, key):
                continue
                
            column = getattr(cls, key)
            
            if isinstance(value, dict):
                # Advanced filtering: {"gte": 10, "lt": 20}
                for op, val in value.items():
                    if op == "eq":
                        query = query.where(column == val)
                    elif op == "ne":
                        query = query.where(column != val)
                    elif op == "gt":
                        query = query.where(column > val)
                    elif op == "gte":
                        query = query.where(column >= val)
                    elif op == "lt":
                        query = query.where(column < val)
                    elif op == "lte":
                        query = query.where(column <= val)
                    elif op == "in":
                        query = query.where(column.in_(val))
                    elif op == "notin":
                        query = query.where(~column.in_(val))
                    elif op == "like":
                        query = query.where(column.like(val))
                    elif op == "ilike":
                        query = query.where(column.ilike(val))
                    elif op == "is_null":
                        query = query.where(column.is_(None) if val else column.isnot(None))
            elif isinstance(value, (list, tuple)):
                # IN clause
                query = query.where(column.in_(value))
            else:
                # Simple equality
                query = query.where(column == value)
                
        return query
    
    @classmethod
    def _apply_ordering(
        cls, 
        query: Select, 
        order_by: Optional[Union[str, List[str]]]
    ) -> Select:
        """Apply ordering to a query."""
        if not order_by:
            return query
            
        if isinstance(order_by, str):
            order_by = [order_by]
            
        for order_field in order_by:
            desc_order = order_field.startswith("-")
            field_name = order_field.lstrip("-")
            
            if hasattr(cls, field_name):
                column = getattr(cls, field_name)
                query = query.order_by(desc(column) if desc_order else asc(column))
                
        return query
    
    @classmethod
    def _apply_soft_delete_filter(cls, query: Select) -> Select:
        """Apply soft delete filter to exclude deleted records."""
        if cls._soft_delete_field and hasattr(cls, cls._soft_delete_field):
            column = getattr(cls, cls._soft_delete_field)
            query = query.where(column.is_(None))
        return query
    
    @classmethod
    def _validate_create_data(cls, data: Dict[str, Any]) -> None:
        """Validate data before creating a record. Override in subclasses."""
        pass
    
    def _validate_update_data(self, data: Dict[str, Any]) -> None:
        """Validate data before updating a record. Override in subclasses."""
        pass
    
    # ============================================================================
    # TRANSACTION CONTEXT MANAGERS
    # ============================================================================
    
    @classmethod
    @asynccontextmanager
    async def transaction(cls, session: AsyncSession):
        """Context manager for database transactions."""
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
    
    # ============================================================================
    # AGGREGATION METHODS
    # ============================================================================
    
    @classmethod
    async def aggregate(
        cls,
        session: AsyncSession,
        field: str,
        operation: str = "sum",
        **filters: Any
    ) -> Optional[float]:
        """Perform aggregation operations on a field."""
        try:
            if not hasattr(cls, field):
                raise ValueError(f"Field {field} not found in {cls.__name__}")
                
            column = getattr(cls, field)
            
            # Map operation to SQLAlchemy function
            agg_functions = {
                "sum": func.sum,
                "avg": func.avg,
                "min": func.min,
                "max": func.max,
                "count": func.count
            }
            
            if operation not in agg_functions:
                raise ValueError(f"Unsupported operation: {operation}")
                
            agg_func = agg_functions[operation]
            query = select(agg_func(column))
            
            if filters:
                # Create a subquery with filters
                subquery = select(cls)
                subquery = cls._apply_filters(subquery, **filters)
                subquery = cls._apply_soft_delete_filter(subquery)
                query = select(agg_func(column)).select_from(subquery.subquery())
            else:
                query = query.select_from(cls)
                query = cls._apply_soft_delete_filter(query)
            
            result = await session.execute(query)
            return result.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error in aggregation for {cls.__name__}: {e}")
            raise DatabaseError(f"Aggregation failed: {str(e)}")