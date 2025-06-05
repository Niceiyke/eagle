# db __init__.py
"""
Eagle Database Module - Enhanced Version with Complete Type Safety

Provides a robust, production-ready database interface with:
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
from typing import (
    AsyncGenerator, Optional, TypeVar, Type, Dict, Any, List, Union, Callable, 
    Generic, Protocol, Sequence, Mapping, Awaitable, ClassVar, Final,
    get_type_hints, get_origin, get_args, cast, overload
)
from typing_extensions import Self, ParamSpec, Concatenate
from datetime import datetime, timezone
from functools import lru_cache, wraps
from enum import Enum
import os
import logging
import asyncio
from dataclasses import dataclass
import time
from abc import ABC, abstractmethod
from .migrations import (
    MigrationManager, migration_manager,
    init_migrations, create_migration, upgrade_database, 
    downgrade_database, auto_upgrade_database, get_migration_status,get_migration_history
)

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
    InstrumentedAttribute
)
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, 
    ForeignKey, func, select, update, delete, 
    inspect as sa_inspect, text, and_, or_,
    Result, ScalarResult, CursorResult
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy.sql.sqltypes import (
    Integer as SQLInteger, String as SQLString, DateTime as SQLDateTime, 
    Boolean as SQLBoolean, Float, Text, JSON, ARRAY, Enum as SQLEnum
)
from sqlalchemy.sql import Select, Update, Delete
from sqlalchemy.sql.elements import BinaryExpression

# Pydantic imports
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, create_model, ValidationError

# Enhanced type variables with better constraints
ModelType = TypeVar("ModelType", bound="BaseModel")
SchemaType_T = TypeVar("SchemaType_T", bound=PydanticBaseModel)
P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")

# Protocol definitions for better type safety
class DatabaseProtocol(Protocol):
    """Protocol for database-like objects"""
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]: ...
    async def health_check(self) -> bool: ...

class BaseModelProtocol(Protocol):
    """Protocol for model-like objects"""
    id: int
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class PaginationResult(Generic[T]):
    """Strongly typed pagination result container"""
    items: Sequence[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
    
    def __post_init__(self) -> None:
        """Validate pagination parameters"""
        if self.page < 1:
            raise ValueError("Page must be >= 1")
        if self.page_size < 1:
            raise ValueError("Page size must be >= 1")
        if self.total < 0:
            raise ValueError("Total must be >= 0")

class SchemaTypeEnum(str, Enum):
    """Schema generation types with better naming"""
    CREATE = "create"
    UPDATE = "update"
    RESPONSE = "response"
    LIST = "list"
    PARTIAL = "partial"

class DatabaseError(Exception):
    """Base database exception with context"""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.context = context or {}

class ConnectionError(DatabaseError):
    """Database connection related errors"""
    pass

class ValidationError(DatabaseError):
    """Data validation errors"""
    pass

class QueryError(DatabaseError):
    """Query execution errors"""
    pass

# Type aliases for better readability
FilterDict = Dict[str, Any]
SchemaCache = Dict[str, Type[PydanticBaseModel]]
ColumnType = Union[InstrumentedAttribute[Any], str]

class SchemaGenerator:
    """Enhanced schema generator with comprehensive type hints"""
    
    def __init__(self, max_cache_size: int = 1000) -> None:
        self._schema_cache: SchemaCache = {}
        self._max_cache_size: Final[int] = max_cache_size
        self._logger: logging.Logger = logging.getLogger(__name__)
    
    def generate_schema(
        self, 
        model_class: Type[ModelType], 
        schema_type: SchemaTypeEnum
    ) -> Type[PydanticBaseModel]:
        """Generate a specific schema type for a model with enhanced error handling"""
        
        model_name: str = model_class.__name__
        cache_key: str = f"{model_name}_{schema_type.value}"
        
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]
        
        # Prevent unbounded cache growth
        if len(self._schema_cache) >= self._max_cache_size:
            self._evict_oldest_cache_entry()
        
        try:
            schema_class = self._create_schema(model_class, schema_type)
            self._schema_cache[cache_key] = schema_class
            return schema_class
        except Exception as e:
            self._logger.error(
                f"Failed to generate schema for {model_name} ({schema_type}): {e}"
            )
            raise ValidationError(f"Schema generation failed: {e}", context={
                "model": model_name,
                "schema_type": schema_type.value
            })
    
    def _evict_oldest_cache_entry(self) -> None:
        """Remove oldest cache entry (simple FIFO)"""
        if self._schema_cache:
            oldest_key = next(iter(self._schema_cache))
            del self._schema_cache[oldest_key]
    
    def _create_schema(
        self, 
        model_class: Type[ModelType], 
        schema_type: SchemaTypeEnum
    ) -> Type[PydanticBaseModel]:
        """Create the actual Pydantic schema with proper typing"""
        
        schema_name: str = f"{model_class.__name__}{schema_type.value.title()}Schema"
        
        try:
            mapper = sa_inspect(model_class)
        except Exception as e:
            raise ValidationError(f"Cannot inspect model {model_class.__name__}: {e}")
        
        field_definitions: Dict[str, tuple[Type[Any], Any]] = {}
        
        for column in mapper.columns:
            field_name: str = column.name
            
            if self._should_skip_field(field_name, schema_type):
                continue
                
            try:
                python_type = self._sqlalchemy_to_python_type(column.type)
                is_optional = self._is_field_optional(column, schema_type)
                
                if is_optional:
                    field_definitions[field_name] = (Optional[python_type], Field(default=None))
                else:
                    field_definitions[field_name] = (python_type, Field())
                    
            except Exception as e:
                self._logger.warning(
                    f"Skipping field {field_name} in {model_class.__name__}: {e}"
                )
                continue
        
        if not field_definitions:
            raise ValidationError(f"No valid fields found for schema {schema_name}")
        
        return create_model(
            schema_name,
            **field_definitions,
            __config__=ConfigDict(
                from_attributes=True,
                validate_assignment=True,
                arbitrary_types_allowed=True,
                str_strip_whitespace=True,
                validate_default=True,
                frozen=schema_type == SchemaTypeEnum.RESPONSE  # Immutable response schemas
            )
        )
    
    def _should_skip_field(self, field_name: str, schema_type: SchemaTypeEnum) -> bool:
        """Enhanced logic for field skipping with type safety"""
        skip_patterns: Dict[SchemaTypeEnum, set[str]] = {
            SchemaTypeEnum.CREATE: {'id', 'created_at', 'updated_at'},
            SchemaTypeEnum.UPDATE: {'id', 'created_at'},
            SchemaTypeEnum.LIST: set(),  # Include all fields for list view
            SchemaTypeEnum.RESPONSE: set(),  # Include all fields for response
            SchemaTypeEnum.PARTIAL: set()  # Include all fields but make optional
        }
        
        return field_name in skip_patterns.get(schema_type, set())
    
    def _is_field_optional(self, column: Any, schema_type: SchemaTypeEnum) -> bool:
        """Enhanced field optionality logic with type safety"""
        if schema_type in {SchemaTypeEnum.UPDATE, SchemaTypeEnum.PARTIAL}:
            return True
        
        return bool(
            column.nullable or 
            column.default is not None or 
            column.server_default is not None or
            getattr(column, 'autoincrement', False)
        )
    
    def _sqlalchemy_to_python_type(self, sql_type: Any) -> Type[Any]:
        """Enhanced type conversion with comprehensive mapping"""
        type_mapping: Dict[Type[Any], Type[Any]] = {
            SQLInteger: int,
            SQLString: str,
            Text: str,
            SQLBoolean: bool,
            Float: float,
            SQLDateTime: datetime,
            JSON: dict,
        }
        
        for sql_class, python_type in type_mapping.items():
            if isinstance(sql_type, sql_class):
                return python_type
                
        if isinstance(sql_type, SQLEnum):
            return str
        if isinstance(sql_type, ARRAY):
            return list
        
        # Log unknown types for debugging
        self._logger.debug(f"Unknown SQLAlchemy type: {sql_type}, defaulting to Any")
        return Any

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
        self.database_url: str = database_url or os.getenv(
            "EAGLE_DATABASE_URL", 
            "sqlite+aiosqlite:///./eagle.db"
        )
        self.echo_sql: bool = bool(kwargs.get('echo_sql', False) or 
                                 os.getenv("EAGLE_ECHO_SQL", "").lower() in ("1", "true", "yes"))
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
    
    # @retry_on_db_error()
    # async def create_tables(self) -> None:
    #     """Create tables with retry logic and proper error handling"""
    #     if not self.engine:
    #         raise ConnectionError("Database engine not initialized")
            
    #     self._logger.info("Creating database tables...")
        
    #     try:
    #         async with self.engine.begin() as conn:
    #             await conn.run_sync(Base.metadata.create_all)
    #         self._logger.info("Database tables created successfully")
    #     except Exception as e:
    #         self._logger.error(f"Failed to create tables: {e}")
    #         raise DatabaseError(f"Table creation failed: {e}")
    
    # async def close(self) -> None:
    #     """Properly close database connections"""
    #     if self.engine:
    #         await self.engine.dispose()
    #         self._logger.info("Database connections closed")

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

# Global instances with proper typing
_schema_generator: SchemaGenerator = SchemaGenerator()
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

class BaseModel(Base, Generic[ModelType]):
    """Production-ready base model with comprehensive functionality and strong typing"""
    __abstract__ = True
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc), 
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc), 
        nullable=False
    )
    
    # Schema generation methods with proper return types
    @classmethod
    def get_create_schema(cls) -> Type[PydanticBaseModel]:
        """Get creation schema with proper typing"""
        return _schema_generator.generate_schema(cls, SchemaTypeEnum.CREATE)
    
    @classmethod
    def get_update_schema(cls) -> Type[PydanticBaseModel]:
        """Get update schema with proper typing"""
        return _schema_generator.generate_schema(cls, SchemaTypeEnum.UPDATE)
    
    @classmethod
    def get_response_schema(cls) -> Type[PydanticBaseModel]:
        """Get response schema with proper typing"""
        return _schema_generator.generate_schema(cls, SchemaTypeEnum.RESPONSE)
    
    @classmethod
    def get_list_schema(cls) -> Type[PydanticBaseModel]:
        """Get list schema with proper typing"""
        return _schema_generator.generate_schema(cls, SchemaTypeEnum.LIST)
    
    @classmethod
    def get_partial_schema(cls) -> Type[PydanticBaseModel]:
        """Get partial update schema with proper typing"""
        return _schema_generator.generate_schema(cls, SchemaTypeEnum.PARTIAL)
    
    # Enhanced database operations with comprehensive type hints
    @classmethod
    async def get(cls, session: AsyncSession, id: int) -> Optional[Self]:
        """Get by ID with proper type checking and return type"""
        if not isinstance(id, int) or id <= 0:
            return None
        result = await session.get(cls, id)
        return cast(Optional[Self], result)
    
    @classmethod
    async def get_by(cls, session: AsyncSession, **filters: Any) -> Optional[Self]:
        """Get single record by filters with proper typing"""
        query: Select[tuple[Self]] = select(cls).filter_by(**filters)
        result: Result[tuple[Self]] = await session.execute(query)
        return result.scalar_one_or_none()
    
    @classmethod
    async def get_all(cls, session: AsyncSession, **filters: Any) -> List[Self]:
        """Get all records with optional filters and proper typing"""
        query: Select[tuple[Self]] = select(cls)
        if filters:
            query = query.filter_by(**filters)
        result: Result[tuple[Self]] = await session.execute(query)
        return list(result.scalars().all())
    
    @classmethod
    async def get_many(
        cls, 
        session: AsyncSession, 
        ids: Sequence[int]
    ) -> List[Self]:
        """Get multiple records by IDs with proper typing"""
        if not ids:
            return []
        
        query: Select[tuple[Self]] = select(cls).where(cls.id.in_(ids))
        result: Result[tuple[Self]] = await session.execute(query)
        return list(result.scalars().all())
    
    @classmethod
    async def paginate(
        cls, 
        session: AsyncSession, 
        page: int = 1, 
        page_size: int = 20,
        order_by: Optional[ColumnType] = None,
        **filters: Any
    ) -> PaginationResult[Self]:
        """Paginated query with metadata and enhanced typing"""
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 1000:  # Prevent abuse
            page_size = 20
            
        offset: int = (page - 1) * page_size
        
        # Build base query with proper typing
        query: Select[tuple[Self]] = select(cls)
        if filters:
            query = query.filter_by(**filters)
        
        # Add ordering if specified
        if order_by is not None:
            if isinstance(order_by, str):
                order_by = getattr(cls, order_by, cls.id)
            query = query.order_by(order_by)
        else:
            query = query.order_by(cls.id)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total: int = await session.scalar(count_query) or 0
        
        # Get paginated results
        paginated_query = query.offset(offset).limit(page_size)
        result: Result[tuple[Self]] = await session.execute(paginated_query)
        items: List[Self] = list(result.scalars().all())
        
        total_pages: int = (total + page_size - 1) // page_size
        
        return PaginationResult(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
    
    @classmethod
    async def exists(cls, session: AsyncSession, **filters: Any) -> bool:
        """Check if record exists with proper typing"""
        query: Select[tuple[int]] = select(cls.id).filter_by(**filters).limit(1)
        result: Optional[int] = await session.scalar(query)
        return result is not None
    
    @classmethod
    async def count(cls, session: AsyncSession, **filters: Any) -> int:
        """Count records matching filters with proper typing"""
        query = select(func.count(cls.id))
        if filters:
            query = query.filter_by(**filters)
        result: Optional[int] = await session.scalar(query)
        return result or 0
    
    @classmethod
    async def create(cls, session: AsyncSession, **data: Any) -> Self:
        """Create with validation and proper typing"""
        try:
            instance: Self = cls(**data)
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance
        except IntegrityError as e:
            await session.rollback()
            raise ValidationError(f"Data integrity violation: {e}")
        except Exception as e:
            await session.rollback()
            raise DatabaseError(f"Create operation failed: {e}")
    
    @classmethod
    async def bulk_create(
        cls, 
        session: AsyncSession, 
        data_list: Sequence[Mapping[str, Any]]
    ) -> List[Self]:
        """Efficient bulk creation with proper typing"""
        if not data_list:
            return []
            
        try:
            instances: List[Self] = [cls(**data) for data in data_list]
            session.add_all(instances)
            await session.flush()
            
            # Refresh all instances to get IDs
            for instance in instances:
                await session.refresh(instance)
                
            return instances
        except IntegrityError as e:
            await session.rollback()
            raise ValidationError(f"Bulk create failed: {e}")
        except Exception as e:
            await session.rollback()
            raise DatabaseError(f"Bulk create operation failed: {e}")
    
    async def update(self, session: AsyncSession, **data: Any) -> Self:
        """Update with validation and proper return typing"""
        try:
            for field, value in data.items():
                if hasattr(self, field) and field not in {'id', 'created_at'}:
                    setattr(self, field, value)
            
            # Update the updated_at field
            self.updated_at = datetime.now(timezone.utc)
            await session.flush()
            await session.refresh(self)
            return self
        except IntegrityError as e:
            await session.rollback()
            raise ValidationError(f"Update failed: {e}")
        except Exception as e:
            await session.rollback()
            raise DatabaseError(f"Update operation failed: {e}")
    
    async def delete(self, session: AsyncSession) -> None:
        """Delete with proper error handling"""
        try:
            await session.delete(self)
            await session.flush()
        except Exception as e:
            await session.rollback()
            raise DatabaseError(f"Delete operation failed: {e}")
    
    async def refresh(self, session: AsyncSession) -> Self:
        """Refresh instance from database with proper typing"""
        await session.refresh(self)
        return self
    
    def to_dict(self, exclude: Optional[set[str]] = None) -> Dict[str, Any]:
        """Convert instance to dictionary with proper typing"""
        exclude = exclude or set()
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
            if column.name not in exclude
        }

# Enhanced utility functions with proper typing
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
            'list': model_class.get_list_schema,
            'partial': model_class.get_partial_schema
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