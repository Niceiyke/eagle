"""
Eagle Database Module - Enhanced Version with Complete Type Safety

This module provides a robust, production-ready database interface with:
- Comprehensive type hints for all methods and functions
- Async SQLAlchemy 2.0+ support with connection retry
- Enhanced error handling and validation
- Advanced query utilities and pagination
- Auto-generated Pydantic schemas with proper validation
- Built-in migration support hooks
- Performance monitoring and health checks
- Generic type support for better IDE experience
"""
from __future__ import annotations
import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_type_hints, cast
from typing_extensions import ParamSpec, TypeVar, Callable, Final, Awaitable, Self, Mapping
from datetime import datetime
from sqlalchemy import func, select, update, delete, text, and_, or_
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import inspect as sa_inspect
from functools import wraps
# Re-export core components
from .models.base import Base, TimestampMixin, SoftDeleteMixin
from .models.mixins import CRUDMixin
from .session import Database, db, get_db
from .exceptions import (
    DatabaseError, ConnectionError, ValidationError, 
    QueryError, NotFoundError, IntegrityError, MigrationError, TimeoutError
)
from .schemas import SchemaConfig,generate_schema
from .types import (
    ModelType, SchemaType, ColumnType, FilterDict, SchemaCache,PaginationResult)
from .utils import (
    retry_on_db_error, execute_with_retry, DatabaseTransaction)
from .migrations import MigrationManager



# Type variables for generic typing
T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')

# SQLAlchemy imports
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, func, select, update, delete, text, and_, or_
from sqlalchemy.orm import Mapped, mapped_column, InstrumentedAttribute, sessionmaker
from sqlalchemy.sql import Select, Update, Delete, text
from sqlalchemy.sql.sqltypes import Integer as SQLInteger, String as SQLString, DateTime as SQLDateTime, Boolean as SQLBoolean, Text, Float, JSON, ARRAY, Enum as SQLEnum
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.engine import Result, CursorResult
from sqlalchemy.exc import OperationalError
from typing import AsyncGenerator, Generic, Sequence

# Re-export SQLAlchemy types for convenience
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, declared_attr

# Re-export Pydantic for schema definitions
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, validator

# For backward compatibility
__all__ = [
    # Core components
    'Database', 'db', 'Base', 'BaseModel', 'get_db',
    
    # Models and mixins
    'TimestampMixin', 'SoftDeleteMixin', 'CRUDMixin',
    
    # Schemas
    'SchemaConfig', 'generate_schema', 'PydanticBaseModel',
    
    # Exceptions
    'DatabaseError', 'ConnectionError', 'ValidationError', 
    'QueryError', 'NotFoundError', 'IntegrityError', 'MigrationError', 'TimeoutError',
    
    # Types
    'ModelType', 'SchemaType', 'ColumnType', 'FilterDict', 'SchemaCache',
    'PaginationResult', 'SortDirection', 'JoinType', 'FilterOperator', 'FilterCondition',
    
    # Utilities
    'retry_on_db_error', 'execute_with_retry', 'DatabaseTransaction',
    
    # SQLAlchemy types and functions
    'AsyncSession', 'DeclarativeBase', 'declared_attr', 'Mapped', 
    'mapped_column', 'InstrumentedAttribute', 'Column', 'Integer', 
    'String', 'DateTime', 'Boolean', 'ForeignKey', 'func', 'select', 
    'update', 'delete', 'text', 'and_', 'or_', 'ConfigDict', 'Field', 'validator'
]

# For backward compatibility
BaseModel = Base  # Alias for backward compatibility


def retry_on_db_error(
    max_retries: int = 3, 
    delay: float = 1.0,
    backoff_multiplier: float = 2.0
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Enhanced decorator for retrying database operations with exponential backoff"""
    
    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (OperationalError, ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff_multiplier ** attempt)
                        await asyncio.sleep(sleep_time)
                        continue
                    break
                except Exception:
                    # Don't retry on non-connection errors
                    raise
                    
            if last_exception:
                raise ConnectionError(
                    f"Database operation failed after {max_retries} attempts"
                ) from last_exception
            
            # This should never be reached, but for type safety
            raise ConnectionError("Unexpected error in retry logic")
        
        return wrapper
    return decorator

class Database:
    """Enhanced database connection and session management with comprehensive typing"""
    
    def __init__(
        self, 
        database_url: Optional[str] = None, 
        **kwargs: Any
    ) -> None:
        from eagleapi.core.config import settings
        self.database_url: str = database_url or settings.DATABASE_URL
        self.echo_sql: bool = bool(kwargs.get('echo_sql', False) or settings.ECHO_SQL)
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._health_check_query: text = text("SELECT 1")
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._setup_engine(**kwargs)
        self.migration_manager = MigrationManager(
            database_url=self.database_url,
            migrations_dir=kwargs.get('migrations_dir', 'migrations')
        )
    
    def _validate_database_url(self, url: str) -> None:
        """Validate database URL format with enhanced checks"""
        if not url or not isinstance(url, str):
            raise ValueError("Database URL must be a non-empty string")
        
        if '://' not in url:
            raise ValueError("Database URL must include a scheme (protocol)")
        
        valid_schemes: List[str] = [
            'postgresql+asyncpg', 
            'mysql+aiomysql', 
            'sqlite+aiosqlite'
        ]
        
        scheme = url.split('://')[0]
        if not any(url.startswith(valid_scheme) for valid_scheme in valid_schemes):
            self._logger.warning(
                f"Database URL scheme may not be supported: {scheme}"
            )
    
    def _setup_engine(self, **kwargs: Any) -> None:
        """Enhanced engine setup with validation and comprehensive error handling"""
        try:
            self._validate_database_url(self.database_url)
            self._logger.info(f"Initializing database connection to {self.database_url}")
            
            connect_args: Dict[str, Any] = {}
            if "sqlite" in self.database_url:
                connect_args.update({
                    "check_same_thread": False,
                    "timeout": 30,
                })
            
            # Enhanced engine configuration with type-safe defaults
            self.engine = create_async_engine(
                self.database_url,
                echo=self.echo_sql,
                future=True,
                pool_pre_ping=True,
                pool_recycle=kwargs.get('pool_recycle', 3600),
                pool_size=kwargs.get('pool_size', 20),
                max_overflow=kwargs.get('max_overflow', 10),
                pool_timeout=kwargs.get('pool_timeout', 30),
                connect_args=connect_args
            )
            
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False
            )
            
            self._logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise ConnectionError(f"Database initialization failed: {e}")
    
    @retry_on_db_error()
    async def health_check(self) -> bool:
        """Check database connection health with proper error handling"""
        if not self.engine:
            return False
            
        try:
            async with self.engine.begin() as conn:
                await conn.execute(self._health_check_query)
            return True
        except Exception as e:
            self._logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Enhanced session management with proper cleanup and error handling"""
        if not self.session_factory:
            raise ConnectionError("Database session factory not initialized")
        
        session: AsyncSession = self.session_factory()
        try:
            yield session
        except Exception as e:
            # Only rollback if there's an active transaction
            try:
                if session.in_transaction():
                    await session.rollback()
            except Exception as rollback_error:
                self._logger.warning(f"Error during rollback: {rollback_error}")
            
            # Re-raise the original exception with proper error handling
            if isinstance(e, IntegrityError):
                raise ValidationError(f"Data integrity error: {e}") from e
            elif isinstance(e, SQLAlchemyError):
                raise DatabaseError(f"Database error: {e}") from e
            raise
        finally:
            # Enhanced session cleanup with proper state checking
            try:
                # Check if session is still valid before closing
                if not session.is_active:
                    return
                
                # If there's still an active transaction, make sure it's completed
                if session.in_transaction():
                    try:
                        # Don't auto-commit here - let the caller handle transactions
                        await session.rollback()
                    except Exception as tx_error:
                        self._logger.warning(f"Error rolling back transaction: {tx_error}")
                
                # Now safely close the session
                await session.close()
                
            except Exception as close_error:
                # Only log as debug since this is cleanup - don't fail the request
                self._logger.debug(f"Session cleanup warning: {close_error}")

    async def auto_migrate(self) -> None:
        """Automatically run pending migrations on startup"""
        try:
            upgraded = await self.migration_manager.auto_upgrade()
            if upgraded:
                self._logger.info("Database migrations applied successfully")
        except Exception as e:
            self._logger.error(f"Migration failed: {e}")
            raise
    
    async def create_tables(self, auto_migrate: bool = True) -> None:
        """Create tables with optional auto-migration"""
        if not self.engine:
            raise ConnectionError("Database engine not initialized")
            
        self._logger.info("Setting up database schema...")
        
        try:
            # Check if migrations directory exists
            if auto_migrate and self.migration_manager.migrations_dir.exists():
                print("Auto-migrating database...")
                # Use migrations
                await self.auto_migrate()
            else:
                print("Auto-migration not enabled or migrations directory does not exist. Creating tables...")
                # Fallback to direct table creation
                async with self.engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                self._logger.info("Database tables created successfully")
                
        except Exception as e:
            self._logger.error(f"Failed to setup database: {e}")
            raise DatabaseError(f"Database setup failed: {e}")

    @asynccontextmanager
    async def session(self):
        """Async context manager for database sessions"""
        if not self.session_factory:
            raise ConnectionError("Database session factory not initialized")
        
        session = self.session_factory()
        try:
            yield session
        except Exception as e:
            # Only rollback if there's an active transaction
            try:
                if session.in_transaction():
                    await session.rollback()
            except Exception as rollback_error:
                self._logger.warning(f"Error during rollback: {rollback_error}")
            
            # Re-raise the original exception with proper error handling
            if isinstance(e, IntegrityError):
                raise ValidationError(f"Data integrity error: {e}") from e
            elif isinstance(e, SQLAlchemyError):
                raise DatabaseError(f"Database error: {e}") from e
            raise
        finally:
            # Enhanced session cleanup with proper state checking
            try:
                # Check if session is still valid before closing
                if not session.is_active:
                    return
                
                # If there's still an active transaction, make sure it's completed
                if session.in_transaction():
                    try:
                        # Don't auto-commit here - let the caller handle transactions
                        await session.rollback()
                    except Exception as tx_error:
                        self._logger.warning(f"Error rolling back transaction: {tx_error}")
                
                # Now safely close the session
                await session.close()
                
            except Exception as close_error:
                # Only log as debug since this is cleanup - don't fail the request
                self._logger.debug(f"Session cleanup warning: {close_error}")

db: Database = Database()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Enhanced FastAPI dependency with proper typing and error handling"""
    async for session in db.get_session():
        yield session

class Base(DeclarativeBase):
    """Enhanced base class with better table naming and type hints"""
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Convert CamelCase to snake_case for table names"""
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

class BaseModel(Base, CRUDMixin, Generic[ModelType]):
    """Production-ready base model with comprehensive functionality and strong typing.
    
    This model includes built-in CRUD operations, timestamps, and common fields.
    All models should inherit from this class to get these features automatically.
    """
    __abstract__ = True
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(), 
        nullable=False,
        doc="Timestamp when the record was created"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(), 
        nullable=False,
        doc="Timestamp when the record was last updated"
    )
    
    # Schema generation methods with proper return types
    @classmethod
    def get_create_schema(cls,config:SchemaConfig=None) -> Type[PydanticBaseModel]:
        """Get creation schema with proper typing"""
        return generate_schema(cls, mode="create",config=config)
    
    @classmethod
    def get_update_schema(cls,config:SchemaConfig=None) -> Type[PydanticBaseModel]:
        """Get update schema with proper typing"""
        return generate_schema(cls, mode="update",config=config)
    
    @classmethod
    def get_response_schema(cls,config:SchemaConfig=None) -> Type[PydanticBaseModel]:
        """Get response schema with proper typing"""
        return generate_schema(cls, mode="response",config=config)
    

    

async def execute_with_retry(
    session: AsyncSession, 
    statement: Union[Select[Any], Update, Delete], 
    max_retries: int = 3
) -> Union[Result[Any], CursorResult[Any]]:
    """Execute statement with retry logic and proper typing"""
    for attempt in range(max_retries):
        try:
            return await session.execute(statement)
        except OperationalError as e:
            if attempt == max_retries - 1:
                raise QueryError(f"Query failed after {max_retries} attempts") from e
            await asyncio.sleep(0.5 * (2 ** attempt))
            continue

def inspect_model_schemas(model_class: Type[BaseModel[Any]]) -> None:
    """Enhanced schema inspection with error handling and better formatting"""
    print(f"\n=== {model_class.__name__} Schemas ===")
    
    try:
        schema_methods = {
            'create': model_class.get_create_schema,
            'update': model_class.get_update_schema,
            'response': model_class.get_response_schema,
        }
        
        for schema_type, schema_method in schema_methods.items():
            try:
                schema_class = schema_method()
                print(f"\n{schema_type.upper()} ({schema_class.__name__}):")
                
                if hasattr(schema_class, '__annotations__'):
                    for field_name, field_type in schema_class.__annotations__.items():
                        print(f"  {field_name}: {field_type}")
                else:
                    print("  No field annotations found")
            except Exception as e:
                print(f"  Error generating {schema_type} schema: {e}")
                
    except Exception as e:
        print(f"Error inspecting schemas: {e}")

# Context manager for database transactions
class DatabaseTransaction:
    """Context manager for database transactions with proper typing"""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._committed = False
    
    async def __aenter__(self) -> AsyncSession:
        return self.session
    
    async def __aexit__(
        self, 
        exc_type: Optional[Type[BaseException]], 
        exc_val: Optional[BaseException], 
        exc_tb: Any
    ) -> None:
        if exc_type is not None:
            await self.session.rollback()
        elif not self._committed:
            await self.session.commit()
            self._committed = True
    
    async def commit(self) -> None:
        """Explicitly commit the transaction"""
        await self.session.commit()
        self._committed = True
    
    async def rollback(self) -> None:
        """Explicitly rollback the transaction"""
        await self.session.rollback()

# Export all components with proper typing
__all__ = [
    # Core classes
    'Database', 'db', 'BaseModel', 'Base',
    
    # Enhanced features
    'PaginationResult', 'SchemaTypeEnum', 'SchemaGenerator',
    'DatabaseError', 'ConnectionError', 'ValidationError', 'QueryError',
    'DatabaseTransaction',
    
    # Protocols
    'DatabaseProtocol', 'BaseModelProtocol',
    
    # Type aliases
    'FilterDict', 'SchemaCache', 'ColumnType',
    
    # Utilities
    'inspect_model_schemas', 'execute_with_retry', 'retry_on_db_error',
    
    # SQLAlchemy components
    'Column', 'Integer', 'String', 'DateTime', 'Boolean', 
    'ForeignKey', 'select', 'update', 'delete', 'func',
    'declared_attr', 'AsyncSession', 'Mapped', 'mapped_column',
    
    # Session management
    'get_db',
    
    # Pydantic
    'PydanticBaseModel', 'Field',
    
    # Type variables
    'ModelType', 'SchemaType_T', 'T', 'R'

    # Migration components
    'MigrationManager', 'MigrationInfo',
    'init_migrations', 'create_migration', 'upgrade_database',
    'downgrade_database', 'auto_upgrade_database', 'get_migration_status','get_migration_history'
]