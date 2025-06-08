from __future__ import annotations
import datetime
from typing import List, Optional, Union, Any, Dict, Set
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
import sqlalchemy
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text, Numeric, Date, Time
from sqlalchemy.sql.sqltypes import TypeDecorator


class SchemaConfig:
    """Configuration for schema generation"""
    def __init__(
        self,
        include_relationships: bool = False,
        max_recursion_depth: int = 2,
        exclude_columns: Set[str] = None,
        include_columns: Set[str] = None,
        field_aliases: Dict[str, str] = None,
        custom_validators: Dict[str, Any] = None
    ):
        self.include_relationships = include_relationships
        self.max_recursion_depth = max_recursion_depth
        self.exclude_columns = exclude_columns or set()
        self.include_columns = include_columns
        self.field_aliases = field_aliases or {}
        self.custom_validators = custom_validators or {}


def _is_auto_generated_field(col: Column) -> bool:
    """Check if a column is auto-generated and should be excluded from create schemas"""
    # Primary keys are typically auto-generated
    if col.primary_key:
        return True
    
    # Fields with server defaults (like CURRENT_TIMESTAMP)
    if col.server_default is not None:
        return True
    
    # Common auto-generated field patterns
    auto_field_names = {
        'id', 'created_at', 'updated_at', 'modified_at', 'date_created', 
        'date_modified', 'timestamp', 'uuid'
    }
    
    if col.name.lower() in auto_field_names:
        return True
    
    # UUID fields with defaults are typically auto-generated
    if 'uuid' in col.name.lower() and col.default is not None:
        return True
        
    # Timestamp fields with defaults
    if any(keyword in col.name.lower() for keyword in ['created', 'updated', 'modified']) and col.default is not None:
        return True
    
    return False


def generate_schema(
    model, 
    mode: str = "response", 
    config: SchemaConfig = None,
    generated_schemas: Dict[str, type] = None,
    current_depth: int = 0
) -> type[BaseModel]:
    """
    Enhanced schema generator that:
    - Converts SQLAlchemy models to Pydantic schemas
    - Handles complex column types and relationships
    - Supports configurable depth control for recursive relationships
    - Provides better type mapping and validation
    """
    if config is None:
        config = SchemaConfig()
    
    if generated_schemas is None:
        generated_schemas = {}

    model_name = f"{model.__name__}{mode.capitalize()}Schema"
    
    # Return existing schema if already generated
    if model_name in generated_schemas:
        return generated_schemas[model_name]

    # Check recursion depth
    if current_depth > config.max_recursion_depth:
        # Return a minimal schema for deep recursion
        minimal_schema = create_minimal_schema(model, mode)
        generated_schemas[model_name] = minimal_schema
        return minimal_schema

    # Create field definitions dictionary
    field_definitions = {}
    
    # Process columns
    for col in model.__table__.columns:
        field_name = col.name
        
        # Skip excluded columns
        if field_name in config.exclude_columns:
            continue
            
        # Include only specified columns if include_columns is set
        if config.include_columns and field_name not in config.include_columns:
            continue
        
        # Skip primary keys in create mode
        if mode == "create" and col.primary_key:
            continue
            
        # Skip auto-generated fields in create mode
        if mode == "create" and _is_auto_generated_field(col):
            continue

        # Get Python type from SQLAlchemy column
        base_type = python_type_from_sqla(col.type)
        
        # Determine if field should be optional
        is_nullable = col.nullable
        is_update_mode = mode == "update"
        
        # Build the field type and default
        if mode == "create":
            if col.default is not None:
                # Has a default value
                if hasattr(col.default, 'arg'):
                    if callable(col.default.arg):
                        # Default factory
                        if is_nullable:
                            field_type = Optional[base_type]
                            field_definitions[field_name] = (field_type, Field(default_factory=col.default.arg))
                        else:
                            field_definitions[field_name] = (base_type, Field(default_factory=col.default.arg))
                    else:
                        # Static default value
                        if is_nullable:
                            field_type = Optional[base_type]
                            field_definitions[field_name] = (field_type, col.default.arg)
                        else:
                            field_definitions[field_name] = (base_type, col.default.arg)
                else:
                    # Default but not easily extractable
                    if is_nullable:
                        field_type = Optional[base_type]
                        field_definitions[field_name] = (field_type, None)
                    else:
                        field_definitions[field_name] = (base_type, ...)
            else:
                # No default
                if is_nullable:
                    field_type = Optional[base_type]
                    field_definitions[field_name] = (field_type, None)
                else:
                    # Required field
                    field_definitions[field_name] = (base_type, ...)
                    
        elif mode == "update":
            # All fields optional in update
            field_type = Optional[base_type]
            field_definitions[field_name] = (field_type, None)
            
        else:  # response mode
            if is_nullable:
                field_type = Optional[base_type]
                field_definitions[field_name] = (field_type, None)
            else:
                # Required field in response
                field_definitions[field_name] = (base_type, ...)

    # Process relationships if enabled and not too deep
    if config.include_relationships and current_depth < config.max_recursion_depth:
        try:
            inspector = sqlalchemy.inspect(model)
            for rel in inspector.relationships:
                rel_name = rel.key
                
                # Skip if excluded
                if rel_name in config.exclude_columns:
                    continue
                    
                related_model = rel.mapper.class_
                
                # Generate related schema with increased depth
                related_schema = generate_schema(
                    related_model, 
                    mode="response",  # Always use response for relationships
                    config=config,
                    generated_schemas=generated_schemas,
                    current_depth=current_depth + 1
                )
                
                if rel.uselist:
                    # One-to-many or many-to-many
                    field_definitions[rel_name] = (Optional[List[related_schema]], None)
                else:
                    # Many-to-one or one-to-one
                    field_definitions[rel_name] = (Optional[related_schema], None)
                    
        except Exception as e:
            # Gracefully handle relationship processing errors
            print(f"Warning: Could not process relationships for {model.__name__}: {e}")

    # Create the dynamic model using create_model approach
    NewModel = create_pydantic_model(model_name, field_definitions)
    
    # Register in cache
    generated_schemas[model_name] = NewModel
    
    return NewModel


def create_pydantic_model(model_name: str, field_definitions: Dict[str, tuple]) -> type[BaseModel]:
    """Create a Pydantic model from field definitions"""
    
    # Separate annotations and defaults
    annotations = {}
    namespace = {
        'model_config': ConfigDict(
            from_attributes=True,
            arbitrary_types_allowed=True,
            validate_assignment=True
        )
    }
    
    for field_name, (field_type, default_value) in field_definitions.items():
        annotations[field_name] = field_type
        
        if default_value is not ...:
            namespace[field_name] = default_value
    
    # Add annotations to namespace
    namespace['__annotations__'] = annotations
    
    # Create the model class
    return type(model_name, (BaseModel,), namespace)


def create_minimal_schema(model, mode: str) -> type[BaseModel]:
    """Create a minimal schema for deep recursion cases"""
    model_name = f"{model.__name__}{mode.capitalize()}SchemaMinimal"
    
    # Only include primary key
    field_definitions = {}
    
    for col in model.__table__.columns:
        if col.primary_key:
            base_type = python_type_from_sqla(col.type)
            field_definitions[col.name] = (base_type, ...)
            break  # Usually just need one primary key
    
    return create_pydantic_model(model_name, field_definitions)


def python_type_from_sqla(sqla_type) -> type:
    """Enhanced type mapping from SQLAlchemy to Python types"""
    
    # Handle TypeDecorator (custom types)
    if isinstance(sqla_type, TypeDecorator):
        return python_type_from_sqla(sqla_type.impl)
    
    # Core types
    type_mapping = {
        Integer: int,
        String: str,
        Text: str,
        Boolean: bool,
        DateTime: datetime.datetime,
        Date: datetime.date,
        Time: datetime.time,
        Float: float,
        Numeric: Decimal,
    }
    
    # Check direct type mapping
    for sqla_type_class, python_type in type_mapping.items():
        if isinstance(sqla_type, sqla_type_class):
            return python_type
    
    # Handle dialect-specific types
    if hasattr(sqlalchemy.dialects, 'postgresql'):
        from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY
        if isinstance(sqla_type, PG_UUID):
            return UUID
        elif isinstance(sqla_type, JSONB):
            return Dict[str, Any]
        elif isinstance(sqla_type, ARRAY):
            item_type = python_type_from_sqla(sqla_type.item_type)
            return List[item_type]
    
    # Handle MySQL specific types
    if hasattr(sqlalchemy.dialects, 'mysql'):
        from sqlalchemy.dialects.mysql import JSON as MySQL_JSON
        if isinstance(sqla_type, MySQL_JSON):
            return Dict[str, Any]
    
    # Handle Enum types
    if hasattr(sqla_type, 'enum_class') and sqla_type.enum_class:
        return sqla_type.enum_class
    
    # Fallback to string for unknown types
    return str


def print_pydantic_model_source(model: type[BaseModel], include_config: bool = False) -> None:
    """Print clean Pydantic model source code"""
    print(f"class {model.__name__}(BaseModel):")
    
    # Print model config if requested
    if include_config and hasattr(model, 'model_config'):
        print(f"    model_config = {repr(model.model_config)}")
        print()
    
    annotations = getattr(model, '__annotations__', {})
    
    if not annotations:
        print("    pass")
        print()
        return
    
    for field_name, field_type in annotations.items():
        # Format the type annotation
        type_str = format_type_annotation(field_type)
        
        # Get the default value from the class
        default_str = ""
        if hasattr(model, field_name):
            default_value = getattr(model, field_name)
            if default_value is not None:
                if isinstance(default_value, Field):
                    # Handle Field objects
                    if hasattr(default_value, 'default'):
                        if default_value.default is not ...:
                            default_str = f" = {repr(default_value.default)}"
                    elif hasattr(default_value, 'default_factory'):
                        factory_name = getattr(default_value.default_factory, '__name__', 'factory')
                        default_str = f" = Field(default_factory={factory_name})"
                else:
                    # Regular default value
                    default_str = f" = {repr(default_value)}"
            elif 'Optional' in type_str:
                default_str = " = None"
        elif 'Optional' in type_str:
            default_str = " = None"
        
        print(f"    {field_name}: {type_str}{default_str}")
    
    print()


def format_type_annotation(type_hint) -> str:
    """Format type annotations for readable output"""
    if hasattr(type_hint, '__origin__'):
        if type_hint.__origin__ is Union:
            args = [a for a in type_hint.__args__ if a is not type(None)]
            if len(args) == 1 and type(None) in type_hint.__args__:
                return f"Optional[{format_type_annotation(args[0])}]"
            else:
                formatted_args = [format_type_annotation(arg) for arg in type_hint.__args__]
                return f"Union[{', '.join(formatted_args)}]"
        elif type_hint.__origin__ is list:
            item_type = format_type_annotation(type_hint.__args__[0])
            return f"List[{item_type}]"
        elif type_hint.__origin__ is dict:
            key_type = format_type_annotation(type_hint.__args__[0])
            value_type = format_type_annotation(type_hint.__args__[1])
            return f"Dict[{key_type}, {value_type}]"
    
    return getattr(type_hint, '__name__', str(type_hint))


# Example usage with enhanced features
if __name__ == "__main__":
    from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
    from sqlalchemy.orm import declarative_base, relationship
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
    import uuid

    Base = declarative_base()

    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        uuid = Column(PG_UUID(as_uuid=True), default=uuid.uuid4, unique=True)
        username = Column(String(50), nullable=False, unique=True)
        email = Column(String(100), nullable=False, unique=True)
        full_name = Column(String(100), nullable=True)
        bio = Column(Text, nullable=True)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
        
        posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
        comments = relationship("Comment", back_populates="author")

    class Post(Base):
        __tablename__ = "posts"
        id = Column(Integer, primary_key=True)
        title = Column(String(200), nullable=False)
        content = Column(Text, nullable=False)
        is_published = Column(Boolean, default=False)
        author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.datetime.utcnow)
        
        author = relationship("User", back_populates="posts")
        comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

    class Comment(Base):
        __tablename__ = "comments"
        id = Column(Integer, primary_key=True)
        content = Column(Text, nullable=False)
        author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
        post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        
        author = relationship("User", back_populates="comments")
        post = relationship("Post", back_populates="comments")

    # Create schemas with different configurations
    basic_config = SchemaConfig()
    
    relationship_config = SchemaConfig(
        include_relationships=True,
        max_recursion_depth=2
    )

    # Generate schemas
    schemas = {}
    
    for model_class in [User, Post, Comment]:
        for mode in ['create', 'update', 'response']:
            # Basic schema
            schema_name = f"{model_class.__name__}{mode.capitalize()}Schema"
            config = relationship_config if mode == 'response' else basic_config
            schemas[schema_name] = generate_schema(model_class, mode, config)

    # Print all schemas
    for schema_name, schema_class in schemas.items():
        print(f"=== {schema_name} ===")
        print_pydantic_model_source(schema_class, include_config=False)
        
    