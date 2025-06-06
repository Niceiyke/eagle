"""
Utility functions for generating TypeScript types from SQLAlchemy models.
"""
from typing import Dict, Type, Any, List, Optional, Set, TypeVar, Union
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Float, Date, 
    ForeignKey, Table, Text, Enum, JSON, ARRAY, Numeric, inspect as sqla_inspect
)
from sqlalchemy.orm import Relationship, Mapped, RelationshipProperty
from datetime import datetime, date
import inspect
import re
import logging

logger = logging.getLogger(__name__)

# Type variable for generic type hints
T = TypeVar('T')

# Set to track processed models to prevent infinite recursion
_processed_models: Set[str] = set()

# Map SQLAlchemy types to TypeScript types
TYPE_MAPPING = {
    Integer: "number",
    String: "string",
    Boolean: "boolean",
    DateTime: "string",  # ISO format string
    Date: "string",      # ISO date string
    Float: "number",
    Numeric: "number",
    Text: "string",
    JSON: "Record<string, any>",
    ARRAY: "any[]",
    type(None): "null",
    bool: "boolean",
    int: "number",
    float: "number",
    str: "string",
    dict: "Record<string, any>",
    list: "any[]",
    set: "any[]",
    tuple: "any[]"
}

def _get_ts_type(column_type: Any) -> str:
    """Convert SQLAlchemy column type to TypeScript type."""
    try:
        if column_type is None:
            return "any"
            
        # Handle SQLAlchemy type with impl
        if hasattr(column_type, 'impl'):
            return _get_ts_type(column_type.impl)
        
        # Handle Python built-in types
        if column_type in TYPE_MAPPING:
            return TYPE_MAPPING[column_type]
        
        # Handle enums
        if hasattr(column_type, 'enums'):
            enum_values = " | ".join(f'"{v}"' for v in column_type.enums)
            return enum_values
        
        # Handle arrays
        if hasattr(column_type, 'item_type'):
            item_type = _get_ts_type(column_type.item_type)
            return f"{item_type}[]"
        
        # Handle foreign keys
        if hasattr(column_type, 'foreign_keys') and column_type.foreign_keys:
            return "number | string"  # Could be more specific with model names
            
        # Handle SQLAlchemy types by class
        for sql_type, ts_type in TYPE_MAPPING.items():
            if isinstance(column_type, sql_type) or (hasattr(column_type, '__class__') and column_type.__class__ == sql_type):
                return ts_type
        
        # Fallback to any for unknown types
        logger.warning(f"Unknown column type: {column_type}, defaulting to 'any'")
        return "any"
        
    except Exception as e:
        logger.error(f"Error getting TypeScript type for {column_type}: {str(e)}")
        return "any"

def _get_model_fields(model: Type[Any]) -> Dict[str, str]:
    """Extract fields from a SQLAlchemy model."""
    global _processed_models
    fields = {}
    
    try:
        # Prevent infinite recursion for circular imports
        model_name = model.__name__
        if model_name in _processed_models:
            return {}
            
        _processed_models.add(model_name)
        
        # Handle columns
        if hasattr(model, '__table__') and model.__table__ is not None:
            for column in model.__table__.columns:
                try:
                    column_type = _get_ts_type(column.type)
                    field_name = column.name
                    fields[field_name] = column_type
                except Exception as e:
                    logger.error(f"Error processing column {column.name} in {model_name}: {str(e)}")
        
        # Handle relationships
        if hasattr(model, '__mapper__') and hasattr(model.__mapper__, 'relationships'):
            for name, relationship in model.__mapper__.relationships.items():
                try:
                    related_model_name = relationship.mapper.class_.__name__
                    if relationship.direction.name == 'MANYTOONE':
                        fields[name] = f"{related_model_name} | null"
                    elif relationship.direction.name in ('ONETOMANY', 'MANYTOMANY'):
                        fields[name] = f"{related_model_name}[]"
                except Exception as e:
                    logger.error(f"Error processing relationship {name} in {model_name}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error getting fields for model {model_name}: {str(e)}")
    finally:
        _processed_models.discard(model_name)
    
    return fields

def model_to_typescript(model: Type[Any], include_imports: bool = True) -> str:
    """Convert a SQLAlchemy model to a TypeScript interface."""
    lines = []
    model_name = model.__name__
    
    if include_imports:
        lines.append("/** Auto-generated TypeScript interfaces from SQLAlchemy models */")
    
    # Get model fields
    fields = _get_model_fields(model)
    
    # Generate interface
    lines.append(f"export interface {model_name} {{")
    
    for field_name, field_type in fields.items():
        # Handle optional fields (nullable columns)
        column = getattr(model, field_name, None)
        is_optional = False
        if hasattr(column, 'property') and hasattr(column.property, 'columns'):
            is_optional = any(col.nullable for col in column.property.columns)
        elif hasattr(column, 'nullable'):
            is_optional = column.nullable
            
        optional_marker = "?" if is_optional else ""
        lines.append(f"  {field_name}{optional_marker}: {field_type};")
    
    lines.append("}")
    return "\n".join(lines)

def get_all_models() -> List[Type[Any]]:
    """Get all SQLAlchemy models in the application."""
    try:
        # Import inside function to prevent circular imports
        from sqlalchemy.orm import DeclarativeBase
        from sqlalchemy.inspection import inspect as sqla_inspect
        
        # Get all classes that inherit from BaseModel
        from eagleapi.db import BaseModel
        
        models = []
        
        # Get all subclasses of BaseModel
        def get_subclasses(cls):
            subclasses = set(cls.__subclasses__())
            return subclasses.union(
                [s for c in subclasses for s in get_subclasses(c)]
            )
        
        for model in get_subclasses(BaseModel):
            try:
                # Check if it's a proper SQLAlchemy model
                if hasattr(model, '__tablename__') and hasattr(model, '__table__'):
                    models.append(model)
            except Exception as e:
                logger.warning(f"Skipping potential model {model.__name__}: {str(e)}")
        
        return models
        
    except ImportError as e:
        logger.error(f"Error importing SQLAlchemy: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return []

def generate_all_typescript_interfaces() -> str:
    """Generate TypeScript interfaces for all SQLAlchemy models."""
    models = get_all_models()
    output = [
        "/**",
        " * Auto-generated TypeScript interfaces from SQLAlchemy models",
        " * This file is auto-generated. Do not edit manually.",
        " */",
        "",
        "// Base types",
        "type Nullable<T> = T | null;",
        ""
    ]
    
    # Generate interfaces for each model
    for model in models:
        output.append(f"// {model.__name__} model")
        output.append(model_to_typescript(model, include_imports=False))
        output.append("")  # Add empty line between models
    
    return "\n".join(output)
