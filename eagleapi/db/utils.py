"""
Utility functions for database operations.
"""
from __future__ import annotations
import logging
import inspect
import functools
import asyncio
from typing import (
    Any, Callable, TypeVar, Optional, Dict, List, Type, Union, ParamSpec,
    Sequence, Awaitable, cast, overload
)
from datetime import datetime

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.exc import SQLAlchemyError

from . import types
from .exceptions import DatabaseError, QueryError

# Type variables
T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')
ModelType = TypeVar('ModelType', bound='Base')

# Logger
logger = logging.getLogger(__name__)

def retry_on_db_error(
    max_retries: int = 3, 
    delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (SQLAlchemyError,)
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator for retrying database operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for delay between retries
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            current_delay = delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                        
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_multiplier
            
            # If we get here, all retries failed
            error_msg = f"Failed after {max_retries} attempts"
            if last_exception:
                error_msg += f": {last_exception}"
            raise DatabaseError(error_msg) from last_exception
        
        return wrapper
    return decorator

def get_model_columns(model: Type[ModelType]) -> List[str]:
    """Get all column names from a SQLAlchemy model.
    
    Args:
        model: SQLAlchemy model class
        
    Returns:
        List of column names
    """
    return [column.name for column in sa_inspect(model).columns]

def get_primary_key(model: Type[ModelType]) -> str:
    """Get the primary key column name from a SQLAlchemy model.
    
    Args:
        model: SQLAlchemy model class
        
    Returns:
        Name of the primary key column
        
    Raises:
        ValueError: If no primary key is found or multiple primary keys exist
    """
    inspector = sa_inspect(model)
    pk_columns = inspector.primary_key
    
    if not pk_columns:
        raise ValueError(f"No primary key found for model {model.__name__}")
    if len(pk_columns) > 1:
        raise ValueError(
            f"Composite primary keys are not supported. "
            f"Model {model.__name__} has multiple primary key columns: {pk_columns}"
        )
    
    return pk_columns[0].name

async def execute_with_retry(
    session: AsyncSession, 
    statement: types.Select | types.Update | types.Delete,
    max_retries: int = 3
) -> Any:
    """Execute a SQL statement with retry logic.
    
    Args:
        session: Async database session
        statement: SQLAlchemy statement to execute
        max_retries: Maximum number of retry attempts
        
    Returns:
        Result of the statement execution
    """
    @retry_on_db_error(max_retries=max_retries)
    async def _execute() -> Any:
        result = await session.execute(statement)
        return result
        
    return await _execute()

def get_column_expression(
    model: Type[ModelType], 
    column_name: str
) -> InstrumentedAttribute[Any]:
    """Get a column expression from a model by name.
    
    Args:
        model: SQLAlchemy model class
        column_name: Name of the column to get
        
    Returns:
        SQLAlchemy column expression
        
    Raises:
        AttributeError: If the column does not exist on the model
    """
    if not hasattr(model, column_name):
        raise AttributeError(
            f"Model {model.__name__} has no column '{column_name}'"
        )
    
    column = getattr(model, column_name)
    if not isinstance(column, InstrumentedAttribute):
        raise AttributeError(
            f"'{column_name}' is not a column on model {model.__name__}"
        )
    
    return column

def model_to_dict(
    instance: ModelType, 
    exclude: Optional[set[str]] = None
) -> Dict[str, Any]:
    """Convert a SQLAlchemy model instance to a dictionary.
    
    Args:
        instance: SQLAlchemy model instance
        exclude: Set of field names to exclude
        
    Returns:
        Dictionary representation of the model
    """
    exclude = exclude or set()
    return {
        c.name: getattr(instance, c.name)
        for c in sa_inspect(instance).mapper.column_attrs
        if c.key not in exclude
    }

class DatabaseTransaction:
    """Context manager for database transactions with proper typing."""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._committed = False
    
    async def __aenter__(self) -> DatabaseTransaction:
        """Enter the transaction context."""
        if self.session.in_transaction() and not self.session.in_nested_transaction():
            self.session.begin_nested()
        return self
    
    async def __aexit__(
        self, 
        exc_type: Optional[Type[BaseException]], 
        exc_val: Optional[BaseException], 
        exc_tb: Any
    ) -> None:
        """Exit the transaction context and handle commit/rollback."""
        if exc_type is not None:
            await self.rollback()
        elif not self._committed:
            await self.commit()
    
    async def commit(self) -> None:
        """Commit the transaction."""
        try:
            await self.session.commit()
            self._committed = True
        except SQLAlchemyError as e:
            await self.rollback()
            raise DatabaseError("Failed to commit transaction") from e
    
    async def rollback(self) -> None:
        """Rollback the transaction."""
        if self.session.in_transaction():
            await self.session.rollback()
        self._committed = False
