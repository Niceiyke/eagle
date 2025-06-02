# config.py
"""
Database configuration and exports
Re-exports enhanced database functionality with schema generation
"""
from typing import Optional, AsyncGenerator, Type, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel as PydanticBaseModel

# Import all enhanced database components
from . import (
    # Core database
    db, Base, BaseModel, get_db,
    
    # Schema generation
    SchemaType, SchemaGenerator, inspect_model_schemas,
    
    # SQLAlchemy components
    Column, Integer, String, DateTime, Boolean, ForeignKey, relationship,
    select, update, delete, func, Mapped, mapped_column,
    
    # Pydantic
    PydanticBaseModel, Field
)

# Re-export everything for backward compatibility and convenience
__all__ = [
    # Core database classes
    'db', 'Base', 'BaseModel', 'get_db',
    
    # Schema generation
    'SchemaType', 'SchemaGenerator', 'inspect_model_schemas',
    
    # SQLAlchemy components  
    'Column', 'Integer', 'String', 'DateTime', 'Boolean',
    'ForeignKey', 'relationship', 'select', 'update', 'delete',
    'func', 'Mapped', 'mapped_column',
    
    # Pydantic
    'PydanticBaseModel', 'Field',
    
    # Type hints
    'AsyncSession'
]

# Convenience functions
def setup_database(database_url: Optional[str] = None, echo_sql: bool = False):
    """Setup database with custom configuration"""
    global db
    if database_url or echo_sql:
        from . import Database
        db = Database(database_url=database_url, echo_sql=echo_sql)
    return db

async def create_all_tables():
    """Create all database tables"""
    await db.create_tables()

async def drop_all_tables():
    """Drop all database tables (WARNING: This deletes all data!)"""
    await db.drop_tables()

def inspect_all_models():
    """Inspect schemas for all registered models"""
    print("=== All Model Schemas ===")
    
    # Get all model classes from the registry
    for model_class in Base.registry._class_registry.values():
        if (hasattr(model_class, '__tablename__') and 
            hasattr(model_class, 'get_all_schemas') and
            not getattr(model_class, '__abstract__', False)):
            inspect_model_schemas(model_class)