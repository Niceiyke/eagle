"""
Type definitions and protocols for the database module.
"""
from __future__ import annotations
from typing import (
    TypeVar, Generic, Type, Dict, Any, List, Optional, Union, Callable,
    Sequence, Mapping, Awaitable, ClassVar, get_type_hints, get_origin, get_args,
    cast, overload, Protocol, runtime_checkable, AsyncGenerator, AnyStr
)
from typing_extensions import Self, ParamSpec, Concatenate
from datetime import datetime
from enum import Enum
from pydantic import BaseModel as PydanticBaseModel
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql import Select, Update, Delete

# Type variables
T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')

# Model and schema type variables
ModelType = TypeVar('ModelType', bound='Base')
SchemaType = TypeVar('SchemaType', bound=PydanticBaseModel)

# Type aliases
ColumnType = Union[InstrumentedAttribute[Any], str]
FilterDict = Dict[str, Any]
SchemaCache = Dict[str, Type[PydanticBaseModel]]

# Protocols
@runtime_checkable
class DatabaseProtocol(Protocol):
    """Protocol for database connection objects."""
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]: ...
    async def health_check(self) -> bool: ...

@runtime_checkable
class BaseModelProtocol(Protocol):
    """Protocol for model-like objects."""
    id: int
    created_at: datetime
    updated_at: datetime
    
    @classmethod
    def __tablename__(cls) -> str: ...

# Data structures
class PaginationResult(Generic[T]):
    """Container for paginated query results."""
    
    def __init__(
        self,
        items: Sequence[T],
        total: int,
        page: int,
        page_size: int,
        total_pages: int,
        has_next: bool,
        has_prev: bool
    ) -> None:
        """Initialize pagination result.
        
        Args:
            items: List of items in the current page
            total: Total number of items across all pages
            page: Current page number (1-based)
            page_size: Number of items per page
            total_pages: Total number of pages
            has_next: Whether there is a next page
            has_prev: Whether there is a previous page
        """
        self.items = items
        self.total = total
        self.page = page
        self.page_size = page_size
        self.total_pages = total_pages
        self.has_next = has_next
        self.has_prev = has_prev
        
        # Validate inputs
        if page < 1:
            raise ValueError("Page must be >= 1")
        if page_size < 1:
            raise ValueError("Page size must be >= 1")
        if total < 0:
            raise ValueError("Total must be >= 0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "items": self.items,
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
            "has_next": self.has_next,
            "has_prev": self.has_prev
        }

# SQLAlchemy types
class SortDirection(str, Enum):
    """Sort direction for queries."""
    ASC = "asc"
    DESC = "desc"

class JoinType(str, Enum):
    """SQL join types."""
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    OUTER = "outer"

# Utility types for query building
class FilterOperator(str, Enum):
    """Operators for building complex filters."""
    EQ = "eq"  # Equal to
    NE = "ne"  # Not equal to
    GT = "gt"  # Greater than
    LT = "lt"  # Less than
    GE = "ge"  # Greater than or equal to
    LE = "le"  # Less than or equal to
    IN = "in"  # In list
    NOT_IN = "not_in"  # Not in list
    LIKE = "like"  # String contains (case-sensitive)
    ILIKE = "ilike"  # String contains (case-insensitive)
    IS_NULL = "is_null"  # Is NULL
    NOT_NULL = "not_null"  # Is not NULL

# Type for filter conditions
FilterCondition = Dict[str, Union[Any, Dict[str, Any]]]
