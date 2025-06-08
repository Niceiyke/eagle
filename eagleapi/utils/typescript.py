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

# Dictionary to track enum mappings for consistent naming
_enum_mappings: Dict[str, Dict[str, Any]] = {}

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
    handle_inheritance: bool = True  # Generate inheritance with extends
    generate_abstract_types: bool = True  # Generate types for abstract models

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
        # Abstract models are valid too if they have fields
        is_abstract = getattr(model, '__abstract__', False)
        
        return (
            (hasattr(model, '__tablename__') or is_abstract) and 
            (hasattr(model, '__table__') or hasattr(model, '__mapper__') or is_abstract)
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

def _get_enum_name(model_name: str, column_name: str, enum_values: List[str]) -> str:
    """Generate consistent enum names and track mappings."""
    # Create a consistent enum name
    enum_name = f"{model_name}{column_name.title()}Enum"
    
    # Store the mapping for later use in enum generation
    enum_key = str(sorted(enum_values))  # Use sorted values as key for consistency
    _enum_mappings[enum_key] = {
        'name': enum_name,
        'values': enum_values
    }
    
    return enum_name

def _get_ts_type(column_type: Any, config: Optional[TypeScriptConfig] = None, model_name: str = "", column_name: str = "") -> str:
    """Convert SQLAlchemy column type to TypeScript type."""
    config = config or TypeScriptConfig()
    
    try:
        if column_type is None:
            return "any"
            
        # Handle SQLAlchemy type with impl
        if hasattr(column_type, 'impl'):
            return _get_ts_type(column_type.impl, config, model_name, column_name)
        
        # Handle Python built-in types
        if column_type in TYPE_MAPPING:
            return TYPE_MAPPING[column_type]
        
        # Handle numeric types with precision
        if isinstance(column_type, (Numeric, Float)):
            return _get_numeric_type(column_type)
        
        # Handle enums - FIXED VERSION
        if hasattr(column_type, 'enums') and column_type.enums:
            if config.generate_enums and model_name and column_name:
                return _get_enum_name(model_name, column_name, list(column_type.enums))
            else:
                enum_values = " | ".join(f'"{v}"' for v in column_type.enums)
                return enum_values
        
        # Handle arrays
        if hasattr(column_type, 'item_type'):
            item_type = _get_ts_type(column_type.item_type, config, model_name, column_name)
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
                    return f"{_get_ts_type(non_none_type, config, model_name, column_name)} | null"
        
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

def _get_model_inheritance_info(model: Type[Any]) -> Tuple[List[str], bool]:
    """
    Get inheritance information for a model.
    Returns: (parent_model_names, is_abstract)
    """
    parent_models = []
    is_abstract = False
    
    try:
        # Check if model is abstract
        is_abstract = getattr(model, '__abstract__', False)
        
        # Get parent models (excluding DeclarativeBase and other SQLAlchemy base classes)
        for base in model.__bases__:
            if (hasattr(base, '__tablename__') or hasattr(base, '__table__')) and base.__name__ != 'DeclarativeBase':
                # Only include if it's a proper SQLAlchemy model
                if hasattr(base, '__mapper__') or hasattr(base, '__table__'):
                    parent_models.append(base.__name__)
    except Exception as e:
        logger.debug(f"Error getting inheritance info for {model.__name__}: {str(e)}")
    
    return parent_models, is_abstract

def _get_model_fields(model: Type[Any], config: Optional[TypeScriptConfig] = None) -> Dict[str, str]:
    """Extract fields from a SQLAlchemy model."""
    config = config or TypeScriptConfig()
    fields = {}
    model_name = model.__name__
    
    with model_processing_context(model_name) as should_process:
        if not should_process:
            return {}
        
        try:
            # Handle columns - use mapper to get all columns including inherited ones
            if hasattr(model, '__mapper__'):
                for column in model.__mapper__.columns:
                    try:
                        # Pass model_name and column_name for proper enum handling
                        column_type = _get_ts_type(column.type, config, model_name, column.name)
                        field_name = column.name
                        if config.camel_case_fields:
                            field_name = _snake_to_camel(field_name)
                        fields[field_name] = column_type
                    except Exception as e:
                        logger.error(f"Error processing column {column.name} in {model_name}: {str(e)}")
            
            # Fallback to table columns if mapper is not available
            elif hasattr(model, '__table__') and model.__table__ is not None:
                for column in model.__table__.columns:
                    try:
                        # Pass model_name and column_name for proper enum handling
                        column_type = _get_ts_type(column.type, config, model_name, column.name)
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

def _generate_enum_types_from_mappings() -> str:
    """Generate TypeScript enums from collected enum mappings."""
    if not _enum_mappings:
        return ""
    
    output = ["// Generated Enums"]
    
    # Use the collected mappings instead of scanning models again
    for enum_info in _enum_mappings.values():
        enum_name = enum_info['name']
        values = enum_info['values']
        
        output.append(f"export enum {enum_name} {{")
        for value in values:
            # Handle special characters in enum values
            key = re.sub(r'[^a-zA-Z0-9_]', '_', str(value).upper())
            output.append(f"  {key} = '{value}',")
        output.append("}")
        output.append("")
    
    return "\n".join(output)

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
    
    # Check if this is an abstract model
    parent_models, is_abstract = _get_model_inheritance_info(model)
    
    # Skip abstract models unless explicitly requested
    if is_abstract and not config.generate_abstract_types:
        return ""
    
    # Get model fields
    fields = _get_model_fields(model, config)
    
    if not fields and not parent_models:
        logger.warning(f"No fields or inheritance found for model {model_name}")
        return ""
    
    # Generate interface or type with inheritance
    extends_clause = ""
    if config.handle_inheritance and parent_models:
        extends_clause = f" extends {', '.join(parent_models)}"
    
    if config.use_interfaces:
        lines.append(f"export interface {model_name}{extends_clause} {{")
    else:
        if extends_clause:
            # For type aliases with inheritance, we need intersection types
            parent_intersection = " & ".join(parent_models)
            lines.append(f"export type {model_name} = {parent_intersection} & {{")
        else:
            lines.append(f"export type {model_name} = {{")
    
    # Only include fields that are not inherited (unless we can't determine inheritance)
    fields_to_include = fields
    if config.handle_inheritance and parent_models:
        # For now, include all fields. In a more sophisticated implementation,
        # we could track which fields come from which parent and exclude inherited ones
        pass
    
    for field_name, field_type in fields_to_include.items():
        # Handle optional fields
        is_optional = _is_field_optional(model, field_name)
        optional_marker = "?" if is_optional else ""
        lines.append(f"  {field_name}{optional_marker}: {field_type};")
    
    lines.append("}")
    
    # Add comment for abstract models
    if is_abstract:
        lines.insert(-len(fields) - 1, "  // Abstract model - used as base for other models")
    
    # Generate additional types if requested (but not for abstract models)
    if config.generate_create_update_types and not is_abstract:
        lines.append("")
        lines.append(f"export type {model_name}CreateInput = CreateInput<{model_name}>;")
        lines.append(f"export type {model_name}UpdateInput = UpdateInput<{model_name}>;")
    
    return "\n".join(lines)

def get_all_models() -> List[Type[Any]]:
    """Get all SQLAlchemy models in the application."""
    try:
        # Import inside function to prevent circular imports
        from sqlalchemy.orm import DeclarativeBase
        
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
                # Try to find any DeclarativeBase
                import sys
                base_class = None
                for name, obj in sys.modules.items():
                    if hasattr(obj, '__dict__'):
                        for attr_name, attr_obj in obj.__dict__.items():
                            if (isinstance(attr_obj, type) and 
                                hasattr(attr_obj, '__tablename__') and 
                                len(attr_obj.__subclasses__()) > 0):
                                base_class = attr_obj
                                break
                        if base_class:
                            break
                
                if not base_class:
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
        
        # Sort models to handle inheritance order (parents before children)
        def sort_by_inheritance(models_list):
            """Sort models so that parent classes come before child classes."""
            sorted_models = []
            remaining_models = models_list.copy()
            
            while remaining_models:
                # Find models with no unprocessed parents
                ready_models = []
                for model in remaining_models:
                    parent_names, _ = _get_model_inheritance_info(model)
                    processed_parent_names = [m.__name__ for m in sorted_models]
                    
                    # Check if all parents are already processed or not in our model list
                    all_parents_ready = all(
                        parent_name in processed_parent_names or 
                        not any(m.__name__ == parent_name for m in remaining_models)
                        for parent_name in parent_names
                    )
                    
                    if all_parents_ready:
                        ready_models.append(model)
                
                if not ready_models:
                    # Circular dependency or missing parent, just add remaining models
                    ready_models = remaining_models
                
                sorted_models.extend(ready_models)
                for model in ready_models:
                    remaining_models.remove(model)
            
            return sorted_models
        
        return sort_by_inheritance(models)
        
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
    
    # Reset processed models set AND enum mappings
    global _processed_models, _enum_mappings
    _processed_models.clear()
    _enum_mappings.clear()
    
    # Validate models
    valid_models = [m for m in models if validate_model(m)]
    
    if not valid_models:
        raise TypeScriptGenerationError("No valid SQLAlchemy models found")
    
    output_parts = [_generate_header()]
    
    if config.include_utility_types:
        output_parts.append(generate_utility_types())
    
    # Generate model interfaces first to collect enum mappings
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
    
    # Now generate enums from collected mappings
    if config.generate_enums:
        enum_types = _generate_enum_types_from_mappings()
        if enum_types:
            output_parts.append(enum_types)
    
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
    parser.add_argument('--no-inheritance', action='store_true', help='Skip inheritance handling')
    parser.add_argument('--include-abstract', action='store_true', help='Include abstract models')
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
        use_interfaces=not args.use_types,
        handle_inheritance=not args.no_inheritance,
        generate_abstract_types=args.include_abstract
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