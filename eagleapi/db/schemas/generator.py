"""
Schema generation utilities for database models.
"""
from __future__ import annotations
from typing import Dict, Type, Any, Optional, get_origin, get_args, cast
from enum import Enum
import logging
from pydantic import BaseModel as PydanticBaseModel, create_model, ValidationError
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Mapper
from .base import BaseSchema

from ..types import ModelType, SchemaCache

class SchemaTypeEnum(str, Enum):
    """Schema generation types with better naming"""
    CREATE = "create"
    UPDATE = "update"
    RESPONSE = "response"
    LIST = "list"
    PARTIAL = "partial"

class SchemaGenerator:
    """Enhanced schema generator with comprehensive type hints"""
    
    def __init__(self, max_cache_size: int = 1000) -> None:
        self._schema_cache: SchemaCache = {}
        self._max_cache_size: int = max_cache_size
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
                f"Failed to generate schema for {model_name} ({schema_type}): {e}",
                exc_info=True
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
                
            field_type = self._map_column_type(column.type)
            default_value = ...  # Required by default
            
            # Handle optional fields for update schemas
            if schema_type in (SchemaTypeEnum.UPDATE, SchemaTypeEnum.PARTIAL):
                default_value = None
                field_type = Optional[field_type]
            
            field_definitions[field_name] = (field_type, default_value)
        
        # Create the model with proper base classes
        base_classes = (BaseSchema,)
        return create_model(schema_name, __base__=base_classes, **field_definitions)
    
    def _should_skip_field(self, field_name: str, schema_type: SchemaTypeEnum) -> bool:
        """Determine if a field should be skipped based on schema type"""
        # Skip certain fields for specific schema types
        if field_name in ('id', 'created_at', 'updated_at'):
            if schema_type == SchemaTypeEnum.CREATE:
                return True
        return False
    
    def _map_column_type(self, column_type: Any) -> type:
        """Map SQLAlchemy column types to Python types"""
        from sqlalchemy import Integer, String, Boolean, DateTime, Float, Text, JSON, ARRAY, Enum as SQLEnum
        
        if isinstance(column_type, (Integer, )):
            return int
        elif isinstance(column_type, (String, Text)):
            return str
        elif isinstance(column_type, Boolean):
            return bool
        elif isinstance(column_type, DateTime):
            from datetime import datetime
            return datetime
        elif isinstance(column_type, Float):
            return float
        elif isinstance(column_type, JSON):
            return dict
        elif isinstance(column_type, ARRAY):
            return list
        elif isinstance(column_type, SQLEnum):
            return str  # Or return the actual Enum type if needed
        else:
            # Default to string for unknown types
            return str
