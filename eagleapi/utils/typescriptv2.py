"""
Enhanced utility functions for generating TypeScript types from SQLAlchemy models.
Includes comprehensive type mapping, relationship handling, enum generation, and more.
"""
from typing import Dict, Type, Any, List, Optional, Set, TypeVar, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Float, Date, Time, Interval,
    ForeignKey, Table, Text, Enum, JSON, ARRAY, Numeric, BigInteger, 
    LargeBinary, inspect as sqla_inspect
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Relationship, Mapped, RelationshipProperty
from datetime import datetime, date
import inspect
import re
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Type variable for generic type hints
T = TypeVar('T')

# Set to track processed models to prevent infinite recursion
_processed_models: Set[str] = set()

@dataclass
class TypeScriptConfig:
    """Configuration for TypeScript generation."""
    use_strict_null_checks: bool = True
    generate_enums: bool = True
    include_relationships: bool = True
    use_interfaces: bool = True  # vs types
    camel_case_fields: bool = False
    include_json_schema: bool = False
    include_utility_types: bool = True
    output_format: str = "interface"  # "interface" | "type" | "class"
    generate_create_update_types: bool = True
    include_pagination_types: bool = True

class TypeScriptGenerationError(Exception):
    """Custom exception for TypeScript generation errors."""
    pass

# Enhanced type mapping with more SQLAlchemy types
TYPE_MAPPING = {
    Integer: "number",
    BigInteger: "bigint",
    String: "string",
    Boolean: "boolean",
    DateTime: "string",  # ISO format string
    Date: "string",      # ISO date string
    Time: "string",      # Time string
    Float: "number",
    Numeric: "number",
    Text: "string",
    JSON: "Record<string, any>",
    ARRAY: "any[]",
    UUID: "string",
    LargeBinary: "Uint8Array | string",
    Interval: "number",  # Duration in milliseconds
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

@contextmanager
def model_processing_context(model_name: str):
    """Context manager for tracking model processing to prevent infinite recursion."""
    if model_name in _processed_models:
        yield False
    else:
        _processed_models.add(model_name)
        try:
            yield True
        finally:
            _processed_models.discard(model_name)

def validate_model(model: Type[Any]) -> bool:
    """Validate that a model can be converted to TypeScript."""
    try:
        return (
            hasattr(model, '__tablename__') and 
            hasattr(model, '__table__') and
            model.__table__ is not None
        )
    except Exception:
        return False

def _get_numeric_type(column_type) -> str:
    """Get appropriate numeric type based on precision and scale."""
    if hasattr(column_type, 'precision') and hasattr(column_type, 'scale'):
        if column_type.scale == 0:
            return "bigint" if (column_type.precision and column_type.precision > 15) else "number"
    return "number"

def _snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(word.capitalize() for word in components[1:])

def _get_ts_type(column_type: Any, config: Optional[TypeScriptConfig] = None) -> str:
    """Convert SQLAlchemy column type to TypeScript type."""
    config = config or TypeScriptConfig()
    
    try:
        if column_type is None:
            return "any"
            
        # Handle SQLAlchemy type with impl
        if hasattr(column_type, 'impl'):
            return _get_ts_type(column_type.impl, config)
        
        # Handle Python built-in types
        if column_type in TYPE_MAPPING:
            return TYPE_MAPPING[column_type]
        
        # Handle numeric types with precision
        if isinstance(column_type, (Numeric, Float)):
            return _get_numeric_type(column_type)
        
        # Handle enums
        if hasattr(column_type, 'enums') and column_type.enums:
            if config.generate_enums:
                return f"EnumType_{hash(str(column_type.enums)) % 10000}"
            else:
                enum_values = " | ".join(f'"{v}"' for v in column_type.enums)
                return enum_values
        
        # Handle arrays
        if hasattr(column_type, 'item_type'):
            item_type = _get_ts_type(column_type.item_type, config)
            return f"{item_type}[]"
        
        # Handle foreign keys
        if hasattr(column_type, 'foreign_keys') and column_type.foreign_keys:
            return "number | string"
            
        # Handle SQLAlchemy types by class
        for sql_type, ts_type in TYPE_MAPPING.items():
            if isinstance(column_type, sql_type) or (
                hasattr(column_type, '__class__') and column_type.__class__ == sql_type
            ):
                return ts_type
        
        # Handle type annotations
        if hasattr(column_type, '__origin__'):
            if column_type.__origin__ is Union:
                args = getattr(column_type, '__args__', [])
                if len(args) == 2 and type(None) in args:
                    # Optional type
                    non_none_type = next(arg for arg in args if arg is not type(None))
                    return f"{_get_ts_type(non_none_type, config)} | null"
        
        # Fallback to any for unknown types
        logger.warning(f"Unknown column type: {column_type}, defaulting to 'any'")
        return "any"
        
    except Exception as e:
        logger.error(f"Error getting TypeScript type for {column_type}: {str(e)}")
        return "any"

def _get_relationship_type(relationship, config: Optional[TypeScriptConfig] = None) -> str:
    """Get TypeScript type for SQLAlchemy relationships with forward references."""
    config = config or TypeScriptConfig()
    
    try:
        related_model_name = relationship.mapper.class_.__name__
        
        if relationship.uselist:
            return f"{related_model_name}[]"
        else:
            # Check if relationship is nullable
            nullable = True
            if hasattr(relationship, 'nullable'):
                nullable = relationship.nullable
            elif hasattr(relationship, 'local_columns'):
                # Check if foreign key columns are nullable
                nullable = any(col.nullable for col in relationship.local_columns)
            
            return f"{related_model_name}{' | null' if nullable else ''}"
    except Exception as e:
        logger.error(f"Error processing relationship: {str(e)}")
        return "any"

def _is_field_optional(model, field_name: str) -> bool:
    """Determine if a field should be optional in TypeScript."""
    try:
        # Check table columns
        if hasattr(model, '__table__'):
            for column in model.__table__.columns:
                if column.name == field_name:
                    return column.nullable and column.default is None and not column.server_default
        
        # Check relationships
        if hasattr(model, '__mapper__'):
            for name, rel in model.__mapper__.relationships.items():
                if name == field_name:
                    # One-to-many relationships are always optional in the parent
                    if rel.uselist:
                        return True
                    # Many-to-one relationships depend on foreign key nullability
                    if hasattr(rel, 'local_columns'):
                        return any(col.nullable for col in rel.local_columns)
                    return True
        
        return False
    except Exception as e:
        logger.debug(f"Error checking if field {field_name} is optional: {str(e)}")
        return True  # Default to optional if unsure

def _get_model_fields(model: Type[Any], config: Optional[TypeScriptConfig] = None) -> Dict[str, str]:
    """Extract fields from a SQLAlchemy model."""
    config = config or TypeScriptConfig()
    fields = {}
    model_name = model.__name__
    
    with model_processing_context(model_name) as should_process:
        if not should_process:
            return {}
        
        try:
            # Handle columns
            if hasattr(model, '__table__') and model.__table__ is not None:
                for column in model.__table__.columns:
                    try:
                        column_type = _get_ts_type(column.type, config)
                        field_name = column.name
                        if config.camel_case_fields:
                            field_name = _snake_to_camel(field_name)
                        fields[field_name] = column_type
                    except Exception as e:
                        logger.error(f"Error processing column {column.name} in {model_name}: {str(e)}")
            
            # Handle relationships
            if config.include_relationships and hasattr(model, '__mapper__'):
                for name, relationship in model.__mapper__.relationships.items():
                    try:
                        field_name = name
                        if config.camel_case_fields:
                            field_name = _snake_to_camel(name)
                        fields[field_name] = _get_relationship_type(relationship, config)
                    except Exception as e:
                        logger.error(f"Error processing relationship {name} in {model_name}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error getting fields for model {model_name}: {str(e)}")
    
    return fields

def _generate_enum_types(models: List[Type[Any]]) -> str:
    """Generate TypeScript enums from SQLAlchemy enum columns."""
    enums = {}
    
    for model in models:
        if hasattr(model, '__table__'):
            for column in model.__table__.columns:
                if hasattr(column.type, 'enums') and column.type.enums:
                    enum_name = f"{model.__name__}{column.name.title()}Enum"
                    enums[enum_name] = list(column.type.enums)
    
    if not enums:
        return ""
    
    output = ["// Generated Enums"]
    for enum_name, values in enums.items():
        output.append(f"export enum {enum_name} {{")
        for value in values:
            # Handle special characters in enum values
            key = re.sub(r'[^a-zA-Z0-9_]', '_', str(value).upper())
            output.append(f"  {key} = '{value}',")
        output.append("}")
        output.append("")
    
    return "\n".join(output)

def generate_utility_types() -> str:
    """Generate useful TypeScript utility types."""
    return '''// Utility types for database operations
export type Nullable<T> = T | null;
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// Database entity base types
export interface BaseEntity {
  id: number;
  created_at: string;
  updated_at: string;
}

// CRUD operation types
export type CreateInput<T> = Omit<T, 'id' | 'created_at' | 'updated_at'>;
export type UpdateInput<T> = Partial<CreateInput<T>>;
export type DatabaseEntity<T> = T & BaseEntity;

// Pagination types
export interface PaginationParams {
  page?: number;
  limit?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

// Filter and search types
export type FilterOperator = 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge' | 'like' | 'ilike' | 'in' | 'not_in';
export interface Filter {
  field: string;
  operator: FilterOperator;
  value: any;
}

export interface QueryParams extends PaginationParams {
  filters?: Filter[];
  search?: string;
}'''

def model_to_typescript(
    model: Type[Any], 
    config: Optional[TypeScriptConfig] = None,
    include_imports: bool = True
) -> str:
    """Convert a SQLAlchemy model to a TypeScript interface."""
    config = config or TypeScriptConfig()
    lines = []
    model_name = model.__name__
    
    if include_imports and config.include_utility_types:
        lines.append("/** Auto-generated TypeScript interfaces from SQLAlchemy models */")
    
    # Get model fields
    fields = _get_model_fields(model, config)
    
    if not fields:
        logger.warning(f"No fields found for model {model_name}")
        return ""
    
    # Generate interface or type
    if config.use_interfaces:
        lines.append(f"export interface {model_name} {{")
    else:
        lines.append(f"export type {model_name} = {{")
    
    for field_name, field_type in fields.items():
        # Handle optional fields
        is_optional = _is_field_optional(model, field_name)
        optional_marker = "?" if is_optional else ""
        lines.append(f"  {field_name}{optional_marker}: {field_type};")
    
    lines.append("}")
    
    # Generate additional types if requested
    if config.generate_create_update_types:
        lines.append("")
        lines.append(f"export type {model_name}CreateInput = CreateInput<{model_name}>;")
        lines.append(f"export type {model_name}UpdateInput = UpdateInput<{model_name}>;")
    
    return "\n".join(lines)

def get_all_models() -> List[Type[Any]]:
    """Get all SQLAlchemy models in the application."""
    try:
        # Try to get all classes that inherit from BaseModel
        try:
            from eagleapi.db import BaseModel
            base_class = BaseModel
        except ImportError:
            # Fallback to finding DeclarativeBase subclasses
            try:
                from sqlalchemy.ext.declarative import declarative_base
                Base = declarative_base()
                base_class = Base.__class__
            except ImportError:
                logger.error("Could not find SQLAlchemy base class")
                return []
        
        models = []
        
        # Get all subclasses recursively
        def get_subclasses(cls):
            subclasses = set(cls.__subclasses__())
            return subclasses.union(
                [s for c in subclasses for s in get_subclasses(c)]
            )
        
        for model in get_subclasses(base_class):
            if validate_model(model):
                models.append(model)
        
        return models
        
    except ImportError as e:
        logger.error(f"Error importing SQLAlchemy: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return []

def generate_json_schema(model: Type[Any], config: Optional[TypeScriptConfig] = None) -> Dict[str, Any]:
    """Generate JSON Schema for validation."""
    config = config or TypeScriptConfig()
    fields = _get_model_fields(model, config)
    properties = {}
    required = []
    
    def ts_type_to_json_schema(ts_type: str) -> Dict[str, Any]:
        """Convert TypeScript type to JSON Schema type."""
        if "number" in ts_type:
            return {"type": "number"}
        elif "string" in ts_type:
            return {"type": "string"}
        elif "boolean" in ts_type:
            return {"type": "boolean"}
        elif "[]" in ts_type:
            item_type = ts_type.replace("[]", "")
            return {"type": "array", "items": ts_type_to_json_schema(item_type)}
        elif "Record<" in ts_type:
            return {"type": "object"}
        elif "|" in ts_type:
            types = [t.strip() for t in ts_type.split("|")]
            return {"anyOf": [ts_type_to_json_schema(t) for t in types if t != "null"]}
        else:
            return {"type": "string"}  # Default fallback
    
    for field_name, field_type in fields.items():
        if not _is_field_optional(model, field_name):
            required.append(field_name)
        
        properties[field_name] = ts_type_to_json_schema(field_type)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False
    }

def _generate_header() -> str:
    """Generate file header with metadata."""
    return '''/**
 * Auto-generated TypeScript interfaces from SQLAlchemy models
 * 
 * This file is auto-generated. Do not edit manually.
 * Generated at: {timestamp}
 */
'''.format(timestamp=datetime.now().isoformat())

def generate_typescript_definitions(
    models: Optional[List[Type[Any]]] = None,
    config: Optional[TypeScriptConfig] = None,
    output_file: Optional[str] = None
) -> str:
    """
    Generate comprehensive TypeScript definitions.
    
    Args:
        models: List of models to process (None for all)
        config: Generation configuration
        output_file: Optional file path to write output
    
    Returns:
        Generated TypeScript code
    """
    config = config or TypeScriptConfig()
    models = models or get_all_models()
    
    # Reset processed models set
    global _processed_models
    _processed_models.clear()
    
    # Validate models
    valid_models = [m for m in models if validate_model(m)]
    
    if not valid_models:
        raise TypeScriptGenerationError("No valid SQLAlchemy models found")
    
    output_parts = [_generate_header()]
    
    if config.include_utility_types:
        output_parts.append(generate_utility_types())
    
    if config.generate_enums:
        enum_types = _generate_enum_types(valid_models)
        if enum_types:
            output_parts.append(enum_types)
    
    # Generate model interfaces
    model_outputs = []
    for model in valid_models:
        try:
            interface = model_to_typescript(model, config, include_imports=False)
            if interface:
                model_outputs.append(f"// {model.__name__} model")
                model_outputs.append(interface)
        except Exception as e:
            logger.error(f"Failed to generate interface for {model.__name__}: {e}")
            continue
    
    if model_outputs:
        output_parts.append("\n".join(model_outputs))
    
    result = "\n\n".join(filter(None, output_parts))
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        logger.info(f"TypeScript definitions written to {output_file}")
    
    return result

def generate_all_typescript_interfaces(
    config: Optional[TypeScriptConfig] = None,
    output_file: Optional[str] = None
) -> str:
    """
    Legacy function for backwards compatibility.
    Generate TypeScript interfaces for all SQLAlchemy models.
    """
    return generate_typescript_definitions(config=config, output_file=output_file)

# CLI functionality
def main():
    """Command line interface for the TypeScript generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate TypeScript interfaces from SQLAlchemy models')
    parser.add_argument('--output', '-o', help='Output file path', default='types.ts')
    parser.add_argument('--no-enums', action='store_true', help='Skip enum generation')
    parser.add_argument('--no-relationships', action='store_true', help='Skip relationship fields')
    parser.add_argument('--camel-case', action='store_true', help='Convert field names to camelCase')
    parser.add_argument('--use-types', action='store_true', help='Use type aliases instead of interfaces')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    config = TypeScriptConfig(
        generate_enums=not args.no_enums,
        include_relationships=not args.no_relationships,
        camel_case_fields=args.camel_case,
        use_interfaces=not args.use_types
    )
    
    try:
        result = generate_typescript_definitions(config=config, output_file=args.output)
        print(f"Successfully generated TypeScript definitions: {args.output}")
        if args.verbose:
            print(f"Generated {len(result.splitlines())} lines of TypeScript")
    except Exception as e:
        logger.error(f"Error generating TypeScript definitions: {e}")
        exit(1)

if __name__ == "__main__":
    main()