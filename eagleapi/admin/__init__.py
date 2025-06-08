#admin __init__.py
"""
Eagle Admin Dashboard

This module provides an automatic admin interface for Eagle API applications.
"""
from typing import Optional, Type, Dict, Any, List, TypeVar, Callable
from fastapi import Request, Depends, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy import select, inspect, func, and_, or_, not_
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel as PydanticBaseModel
from eagleapi.db import get_db
from pathlib import Path
import os
import importlib
import inspect as py_inspect
from typing import Any, Dict, List, Optional, Type, TypeVar
from functools import wraps
from enum import Enum
from datetime import datetime
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Type variable for model types
ModelType = TypeVar('ModelType')

def register_model_to_admin(cls=None, *, name: str = None):
    """
    Decorator to register a model with the admin interface.
    
    Args:
        name (str, optional): Custom name for the model in the admin interface. 
                           If not provided, the class name will be used.
                           
    Example:
        @register_model_to_admin 
        class MyModel(Base):
            __tablename__ = 'my_models'
            
        # Or with custom name:
        @register_model_to_admin (name='CustomModelName')
        class AnotherModel(Base):
            __tablename__ = 'another_models'
    """
    def wrap(cls):
        ModelRegistry.register(cls, name=name)
        return cls
    
    # Handle both @register_model_to_admin  and @register_model_to_admin () syntax
    if cls is None:
        return wrap
    return wrap(cls)


class FilterType(str, Enum):
    """Enum for different types of filters."""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTIPLE_SELECT = "multiple_select"

class FieldFilter(PydanticBaseModel):
    """Model for field filters."""
    name: str
    label: str
    type: FilterType
    options: Optional[List[str]] = None
    default_value: Optional[str] = None

class ModelInfo(PydanticBaseModel):
    """Model information with filter definitions."""
    name: str
    table_name: str
    fields: List[Dict[str, Any]]
    filters: List[FieldFilter]

    def get_filter_by_name(self, field_name: str) -> Optional[FieldFilter]:
        """Get filter for a specific field by name."""
        return next((f for f in self.filters if f.name == field_name), None)

    @classmethod
    def from_model(cls, model: Type[ModelType]) -> 'ModelInfo':
        """Create ModelInfo from a SQLAlchemy model."""
        mapper = inspect(model)
        fields = []
        filters = []
        
        for column in mapper.columns:
            field_info = {
                "name": column.name,
                "type": str(column.type.__visit_name__).upper() if hasattr(column.type, '__visit_name__') else str(column.type.python_type.__name__).upper(),
                "nullable": column.nullable,
                "primary_key": column.primary_key,
                "default": column.default.arg if column.default else None,
                "foreign_keys": [
                    {
                        "name": fk.name,
                        "referenced_table": fk.column.table.name,
                        "referenced_column": fk.column.name
                    }
                    for fk in column.foreign_keys
                ] if column.foreign_keys else None,
                "unique": column.unique,
                "index": column.index
            }
            fields.append(field_info)
            
            # Determine filter type based on column type
            filter_type = None
            options = None
            
            if column.type.python_type == str:
                filter_type = FilterType.TEXT
            elif column.type.python_type in (int, float):
                filter_type = FilterType.NUMBER
            elif column.type.python_type == bool:
                filter_type = FilterType.BOOLEAN
                options = ["True", "False"]
            elif hasattr(column.type, 'python_type') and issubclass(column.type.python_type, datetime):
                filter_type = FilterType.DATE
            elif column.foreign_keys:
                filter_type = FilterType.SELECT
                # Get distinct values for select options
                async def get_options():
                    async with get_db() as db:
                        result = await db.execute(select(column))
                        return list(set(str(r[0]) for r in result.all()))
                options = get_options()
            
            if filter_type:
                filters.append(
                    FieldFilter(
                        name=column.name,
                        label=column.info.get('label', column.name.replace('_', ' ').title()),
                        type=filter_type,
                        options=options
                    )
                )
                
        return cls(
            name=model.__name__,
            table_name=model.__tablename__,
            fields=fields,
            filters=filters
        )

    class Config:
        arbitrary_types_allowed = True

    def get_filter_by_name(self, name: str) -> Optional[FieldFilter]:
        """Get filter by field name."""
        return next((f for f in self.filters if f.name == name), None)

    def get_filters_by_type(self, filter_type: FilterType) -> List[FieldFilter]:
        """Get all filters of a specific type."""
        return [f for f in self.filters if f.type == filter_type]

class ModelRegistry:
    _instance = None
    _models: Dict[str, Type[ModelType]] = {}
    _model_info_cache: Dict[str, ModelInfo] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, model: Type[ModelType], name: str = None) -> None:
        """
        Register a model with the admin interface.
        
        Args:
            model: The SQLAlchemy model class to register
            name: Optional custom name for the model in the admin interface.
                 If not provided, the class name will be used.
        
        Raises:
            ValueError: If the model is not a valid SQLAlchemy model
            ValueError: If a model with the same name is already registered
        """
        if not hasattr(model, '__tablename__'):
            raise ValueError(f"{model.__name__} is not a valid SQLAlchemy model")
            
        model_name = name or model.__name__
        if model_name in cls._models:
            raise ValueError(f"Model name '{model_name}' is already registered")
            
        cls._models[model_name] = model
        cls._model_info_cache.clear()  # Clear cache when models change
    
    @classmethod
    def get_models(cls) -> Dict[str, Type[ModelType]]:
        """Get all registered models."""
        return cls._models.copy()  # Return a copy to prevent modification
    
    @classmethod
    def get_model(cls, model_name: str) -> Optional[Type[ModelType]]:
        """Get a model by name."""
        return cls._models.get(model_name)
    
    @classmethod
    def get_model_info(cls, model: Type[ModelType]) -> ModelInfo:
        """Get information about a model's fields and relationships."""
        if not hasattr(model, '__tablename__'):
            raise ValueError(f"{model.__name__} is not a valid SQLAlchemy model")
            
        mapper = inspect(model)
        fields = []
        filters = []
        
        # Process columns
        for column in mapper.columns:
            field_info = {
                "name": column.name,
                "type": str(column.type.__visit_name__).upper() if hasattr(column.type, '__visit_name__') else str(column.type.python_type.__name__).upper(),
                "nullable": column.nullable,
                "primary_key": column.primary_key,
                "default": column.default.arg if column.default else None,
                "foreign_keys": [
                    {
                        "name": fk.name,
                        "referenced_table": fk.column.table.name,
                        "referenced_column": fk.column.name
                    }
                    for fk in column.foreign_keys
                ] if column.foreign_keys else None,
                "unique": column.unique,
                "index": column.index,
                "is_relationship": False
            }
            fields.append(field_info)
            
            # Determine filter type based on column type
            filter_type = None
            options = None
            
            if column.type.python_type == str:
                filter_type = FilterType.TEXT
            elif column.type.python_type in (int, float):
                filter_type = FilterType.NUMBER
            elif column.type.python_type == bool:
                filter_type = FilterType.BOOLEAN
                options = ["True", "False"]
            elif hasattr(column.type, 'python_type') and issubclass(column.type.python_type, datetime):
                filter_type = FilterType.DATE
            elif column.foreign_keys:
                filter_type = FilterType.SELECT
                # Get distinct values for select options from the related table
                related_model = next(iter(column.foreign_keys)).column.table
                related_pk = next(iter(related_model.primary_key)).name
                related_columns = [col.name for col in related_model.columns]
                
                async def get_options():
                    async with get_db() as db:
                        # Get all records from the related table
                        result = await db.execute(select(related_model))
                        options = []
                        for item in result.scalars():
                            # Use the first string column as display value, fall back to PK
                            display_value = None
                            for col in related_columns:
                                if col != related_pk and isinstance(getattr(item, col, None), str):
                                    display_value = f"{getattr(item, col)}"
                                    break
                            if not display_value:
                                display_value = str(getattr(item, related_pk))
                            options.append({
                                'value': str(getattr(item, related_pk)),
                                'display': display_value
                            })
                        return options
                options = get_options()
            
            if filter_type:
                filters.append(
                    FieldFilter(
                        name=column.name,
                        label=column.info.get('label', column.name.replace('_', ' ').title()),
                        type=filter_type,
                        options=options
                    )
                )
        
        # Process relationships
        for rel in mapper.relationships:
            # Skip backrefs to avoid duplication
            if rel.back_populates or rel.backref:
                continue
                
            related_model = rel.mapper.class_
            related_pk = next(iter(inspect(related_model).primary_key)).name
            related_columns = [col.name for col in inspect(related_model).columns]
            
            # Add relationship field to fields list
            field_info = {
                "name": rel.key,
                "type": "relationship",
                "relationship_type": 'many-to-many' if rel.uselist else 'many-to-one',
                "uselist": bool(rel.uselist),
                "related_model": related_model.__name__,
                "related_table": related_model.__tablename__,
                "related_pk": related_pk,
                "nullable": rel.direction.name != 'MANYTOONE' or any(c.nullable for c in rel.local_columns),
                "is_relationship": True
            }
            fields.append(field_info)
            
            # Add filter for the relationship
            async def get_related_options():
                async with get_db() as db:
                    result = await db.execute(select(related_model))
                    options = []
                    for item in result.scalars():
                        # Use the first string column as display value, fall back to PK
                        display_value = None
                        for col in related_columns:
                            if col != related_pk and isinstance(getattr(item, col, None), str):
                                display_value = f"{getattr(item, col)}"
                                break
                        if not display_value:
                            display_value = str(getattr(item, related_pk))
                        options.append({
                            'value': str(getattr(item, related_pk)),
                            'display': display_value
                        })
                    return options
                    
            filter_type = FilterType.MULTIPLE_SELECT if rel.uselist else FilterType.SELECT
            filters.append(
                FieldFilter(
                    name=rel.key,
                    label=rel.key.replace('_', ' ').title(),
                    type=filter_type,
                    options=get_related_options()
                )
            )
            
            # Get display columns for the related model
            related_columns = [col.name for col in inspect(related_model).columns]
            
            field_info = {
                "name": rel.key,
                "type": "RELATIONSHIP",
                "relationship_type": str(rel.direction).split('.')[-1],
                "uselist": rel.uselist,
                "related_model": related_model.__name__,
                "related_table": related_model.__tablename__,
                "related_pk": related_pk,
                "is_relationship": True,
                "nullable": rel.direction.name != 'MANYTOONE'  # Many-to-one is not nullable
            }
            fields.append(field_info)
            
            # Add filter for relationship
            if not rel.uselist:  # Only add filter for to-one relationships
                async def get_related_options():
                    async with get_db() as db:
                        result = await db.execute(select(related_model))
                        options = []
                        for item in result.scalars():
                            # Use the first string column as display value, fall back to PK
                            display_value = None
                            for col in related_columns:
                                if col != related_pk and isinstance(getattr(item, col, None), str):
                                    display_value = f"{getattr(item, col)}"
                                    break
                            if not display_value:
                                display_value = str(getattr(item, related_pk))
                            options.append({
                                'value': str(getattr(item, related_pk)),
                                'display': display_value
                            })
                        return options
                
                filters.append(
                    FieldFilter(
                        name=rel.key,
                        label=rel.key.replace('_', ' ').title(),
                        type=FilterType.SELECT,
                        options=get_related_options()
                    )
                )
                
        model_info = ModelInfo(
            name=model.__name__,
            table_name=model.__tablename__,
            fields=fields,
            filters=filters
        )
        cls._model_info_cache[model.__name__] = model_info
        return model_info
    
    @classmethod
    def scan_models(cls, models_module: str) -> None:
        """
        Scan a module for SQLAlchemy models and register them.
        
        Args:
            models_module: Dot path to the module containing models
            
        Raises:
            ImportError: If the module cannot be imported
            ValueError: If no models are found in the module
        """
        try:
            module = importlib.import_module(models_module)
            models_found = False
            
            for name, obj in py_inspect.getmembers(module):
                if (
                    py_inspect.isclass(obj) 
                    and hasattr(obj, '__tablename__')
                    and obj.__module__ == models_module
                ):
                    cls.register(obj)
                    models_found = True
            
            if not models_found:
                raise ValueError(f"No SQLAlchemy models found in module '{models_module}'")
                
        except ImportError as e:
            raise ImportError(f"Error importing module '{models_module}': {str(e)}") from e
        except ValueError as e:
            raise ValueError(f"Error scanning models in '{models_module}': {str(e)}") from e
        except Exception as e:
            raise Exception(f"Unexpected error scanning models: {str(e)}") from e
    
    @classmethod
    def unregister(cls, model_name: str) -> None:
        """
        Unregister a model from the admin interface.
        
        Args:
            model_name: Name of the model to unregister
            
        Raises:
            KeyError: If the model is not registered
        """
        if model_name not in cls._models:
            raise KeyError(f"Model '{model_name}' is not registered")
            
        del cls._models[model_name]
        cls._model_info_cache.clear()  # Clear cache when models change

# Create a singleton instance
model_registry = ModelRegistry()

class Admin:
    def __init__(self, app, path: str = "/admin", models_module: str = None):
        """
        Initialize the admin interface.
        
        Args:
            app: The EagleAPI instance
            path: The URL path where the admin dashboard will be mounted
            models_module: Dot path to the module containing SQLAlchemy models (e.g., 'app.models')
        """
        self.app = app
        self.path = path
        self.models_module = models_module
        self._setup()
    
    def _setup(self):
        """Set up the admin interface."""
        # Get the directory where this file is located
        current_dir = Path(__file__).parent
        
        # Mount static files
        self.static_dir = current_dir / "static"
        if not self.static_dir.exists():
            self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure templates
        self.templates_dir = current_dir / "templates"
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.templates = Jinja2Templates(directory=str(self.templates_dir))
        
        # Auto-discover models if models_module is provided
        if self.models_module:
            model_registry.scan_models(self.models_module)
        
        # Add admin routes to the main app
        self._add_routes()
               
        # Mount static files
        self.app.mount(
            f"{self.path}/static",
            StaticFiles(directory=str(self.static_dir)),
            name="admin_static"
        )
        
        self.app.logger.info(f"Admin dashboard available at {self.path}")
    
    def _add_routes(self):
        """Add admin routes to the main application."""
        @self.app.get(self.path, include_in_schema=False)
        async def admin_dashboard(request: Request):
            models = []
            for name, model in model_registry.get_models().items():
                model_info = model_registry.get_model_info(model)
                models.append({
                    "name": name,
                    "table_name": model_info.table_name,
                    "fields": model_info.fields
                })
            return self.templates.TemplateResponse(
                "admin.html",
                {
                    "request": request,
                    "title": "Eagle Admin Dashboard",
                    "models": models,
                    "admin_path": self.path
                }
            )
            
        @self.app.get(f"{self.path}/{{model_name}}", include_in_schema=False)
        async def list_model(
            request: Request,
            model_name: str,
            db: AsyncSession = Depends(get_db)
        ):
            try:
                self.app.logger.info(f"Loading model list for {model_name}")
                model = model_registry.get_model(model_name)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                    
                # Get model info first to understand relationships
                model_info = model_registry.get_model_info(model)
                self.app.logger.info(f"Model info: {model_info}")
                
                # Build the query with joinedload for relationships
                stmt = select(model)
                
                # Add joinedload for all relationships to avoid N+1 queries
                for field in model_info.fields:
                    if field.get('is_relationship'):
                        relationship_name = field['name']
                        self.app.logger.info(f"Adding selectinload for relationship: {relationship_name}")
                        stmt = stmt.options(selectinload(getattr(model, relationship_name)))
                
                # Execute the query
                self.app.logger.info("Executing query...")
                result = await db.execute(stmt)
                items = result.scalars().all()
                self.app.logger.info(f"Found {len(items)} items")
                
                # Prepare items data for the template
                prepared_items = []
                for item in items:
                    item_data = {}
                    for field in model_info.fields:
                        field_name = field['name']
                        try:
                            if field.get('is_relationship'):
                                # Handle relationship fields
                                related = getattr(item, field_name, None)
                                self.app.logger.debug(f"Processing relationship field {field_name}: {related}")
                                if related is None:
                                    item_data[field_name] = None
                                elif field.get('uselist'):
                                    # For to-many relationships
                                    related_list = list(related) if related else []
                                    self.app.logger.debug(f"To-many relationship {field_name} has {len(related_list)} items")
                                    item_data[field_name] = related_list
                                else:
                                    # For to-one relationships
                                    self.app.logger.debug(f"To-one relationship {field_name}: {related}")
                                    item_data[field_name] = related
                            else:
                                # Regular field
                                value = getattr(item, field_name, None)
                                item_data[field_name] = value
                        except Exception as e:
                            # If there's an error accessing the field, log it and set to None
                            error_msg = f"Error accessing field {field_name}: {str(e)}"
                            self.app.logger.error(error_msg, exc_info=True)
                            item_data[field_name] = None
                    
                    # Add the item's primary key for actions
                    item_data['id'] = getattr(item, 'id', None)
                    prepared_items.append(item_data)
                
                # Get field names for the table headers
                field_names = [field['name'] for field in model_info.fields]
                
                # Log the first item for debugging
                if prepared_items:
                    self.app.logger.debug(f"First item data: {prepared_items[0]}")
                
                return self.templates.TemplateResponse(
                    "model_list.html",
                    {
                        "request": request,
                        "title": f"{model_name} List",
                        "model_name": model_name,
                        "items": prepared_items,
                        "fields": field_names,
                        "model_info": model_info,
                        "admin_path": self.path
                    }
                )
            except Exception as e:
                self.app.logger.error(f"Error in list_model: {str(e)}", exc_info=True)
                raise
            
        @self.app.get(f"{self.path}/{{model_name}}/new", include_in_schema=False)
        async def new_model_form(
            request: Request,
            model_name: str,
        ):
            model = model_registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
                
            model_info = model_registry.get_model_info(model)
            
            return self.templates.TemplateResponse(
                "model_form.html",
                {
                    "request": request,
                    "title": f"New {model_name}",
                    "model_name": model_name,
                    "fields": model_info.fields,
                    "model_info": model_info,
                    "admin_path": self.path
                }
            )
            
        @self.app.post(f"{self.path}/{{model_name}}/create", include_in_schema=False)
        async def create_model(
            request: Request,
            model_name: str,
            db: AsyncSession = Depends(get_db)
        ):
            model = model_registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
                
            form_data = await request.form()
            data = {}
            
            # Get model info to check field types
            model_info = model_registry.get_model_info(model)
            
            # Convert form data to proper types
            for field in model_info.fields:
                field_name = field['name']
                if field_name not in form_data:
                    continue
                    
                value = form_data[field_name]
                
                # Skip empty values for non-required fields
                if value == '':
                    if field.get('nullable', True):
                        data[field_name] = None
                    continue
                
                # Handle relationship fields
                if field.get('is_relationship'):
                    if field['uselist']:  # Many-to-many or one-to-many
                        # Handle multiple selections
                        if isinstance(value, str):
                            value = [v for v in value.split(',') if v.strip()]
                        if not value:  # Skip if no values provided
                            continue
                        # Get the related model and primary key
                        related_model = model_registry.get_model(field['related_model'])
                        related_pk = field['related_pk']
                        # Create a list of related objects
                        related_objects = []
                        for pk in value:
                            result = await db.execute(
                                select(related_model).where(
                                    getattr(related_model, related_pk) == pk
                                )
                            )
                            if obj := result.scalars().first():
                                related_objects.append(obj)
                        if related_objects:
                            data[field_name] = related_objects
                        continue
                    else:  # Many-to-one or one-to-one
                        # Get the related model and primary key
                        related_model = model_registry.get_model(field['related_model'])
                        related_pk = field['related_pk']
                        # Get the related object
                        result = await db.execute(
                            select(related_model).where(
                                getattr(related_model, related_pk) == value
                            )
                        )
                        if obj := result.scalars().first():
                            data[field_name] = obj
                        continue
                
                # Convert based on field type for regular fields
                field_type = field['type']
                
                # Handle boolean fields - checkboxes send 'on' when checked, nothing when unchecked
                if field_type == 'BOOLEAN':
                    value = True if value == 'on' else False
                # Convert datetime fields
                elif field_type == 'DATETIME' and value:
                    from datetime import datetime
                    try:
                        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        continue  # Skip if datetime conversion fails
                # Convert integer fields
                elif field_type in ['INTEGER', 'BIGINT', 'SMALLINT'] and value:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        continue  # Skip if conversion fails
                # Handle foreign key fields
                elif field.get('foreign_keys') and value:
                    try:
                        value = int(value) if value.isdigit() else value
                    except (ValueError, AttributeError):
                        continue  # Skip if conversion fails
                    
                    data[field_name] = value
            
            # Create new record
            if data:  # Only proceed if we have data to insert
                stmt = insert(model).values(**data)
                await db.execute(stmt)
                await db.commit()
            
            return RedirectResponse(
                url=f"{self.path}/{model_name}",
                status_code=303
            )
            
        @self.app.get(f"{self.path}/{{model_name}}/{{item_id}}/edit", include_in_schema=False)
        async def edit_model_form(
            request: Request,
            model_name: str,
            item_id: int,
            db: AsyncSession = Depends(get_db)
        ):
            model = model_registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
                
            # Get primary key column name
            pk = inspect(model).primary_key[0].name
            
            # Get the item
            result = await db.execute(select(model).where(getattr(model, pk) == item_id))
            item = result.scalars().first()
            
            if not item:
                raise HTTPException(status_code=404, detail="Item not found")
                
            model_info = model_registry.get_model_info(model)
            
            return self.templates.TemplateResponse(
                "model_form.html",
                {
                    "request": request,
                    "title": f"Edit {model_name}",
                    "model_name": model_name,
                    "item": item,
                    "fields": model_info.fields,
                    "admin_path": self.path,
                    "model_info": model_info  # Add model_info to template context
                }
            )
            
        @self.app.post(f"{self.path}/{{model_name}}/{{item_id}}/update", include_in_schema=False)
        async def update_model(
            request: Request,
            model_name: str,
            item_id: int,
            db: AsyncSession = Depends(get_db)
        ):
            model = model_registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
                
            # Get primary key column name
            pk = inspect(model).primary_key[0].name
            
            form_data = await request.form()
            data = {}
            
            # Get model info to check field types
            model_info = model_registry.get_model_info(model)
            
            # Convert form data to proper types
            for field in model_info.fields:
                field_name = field['name']
                if field_name not in form_data:
                    continue
                    
                value = form_data[field_name]
                
                # Skip empty values for non-required fields
                if value == '':
                    if field.get('nullable', True):
                        data[field_name] = None
                    continue
                
                # Handle relationship fields
                if field.get('is_relationship'):
                    if field['uselist']:  # Many-to-many or one-to-many
                        # Handle multiple selections
                        if isinstance(value, str):
                            value = [v for v in value.split(',') if v.strip()]
                        if not value:  # Skip if no values provided
                            continue
                            
                        # Get the related model and primary key
                        related_model = model_registry.get_model(field['related_model'])
                        related_pk = field['related_pk']
                        
                        # Get existing related objects
                        result = await db.execute(select(model).where(getattr(model, pk) == item_id))
                        existing_item = result.scalars().first()
                        
                        if existing_item:
                            # Clear existing relationships
                            related_attr = getattr(existing_item, field_name)
                            if hasattr(related_attr, 'clear'):
                                related_attr.clear()
                            
                            # Add new relationships
                            related_objects = []
                            for pk_val in value:
                                result = await db.execute(
                                    select(related_model).where(
                                        getattr(related_model, related_pk) == pk_val
                                    )
                                )
                                if obj := result.scalars().first():
                                    related_objects.append(obj)
                            
                            if related_objects:
                                setattr(existing_item, field_name, related_objects)
                        
                        continue  # Skip normal data processing for relationships
                    else:  # Many-to-one or one-to-one
                        # Get the related model and primary key
                        related_model = model_registry.get_model(field['related_model'])
                        related_pk = field['related_pk']
                        # Get the related object
                        if value:  # Only process if a value is provided
                            result = await db.execute(
                                select(related_model).where(
                                    getattr(related_model, related_pk) == value
                                )
                            )
                            if obj := result.scalars().first():
                                data[field_name] = obj
                        else:
                            data[field_name] = None
                        continue
                
                # Convert based on field type for regular fields
                field_type = field['type']
                
                # Handle boolean fields - checkboxes send 'on' when checked, nothing when unchecked
                if field_type == 'BOOLEAN':
                    value = True if value == 'on' else False
                # Convert datetime fields
                elif field_type == 'DATETIME' and value:
                    from datetime import datetime
                    try:
                        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        continue  # Skip if datetime conversion fails
                # Convert integer fields
                elif field_type in ['INTEGER', 'BIGINT', 'SMALLINT'] and value:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        continue  # Skip if conversion fails
                # Handle foreign key fields
                elif field.get('foreign_keys') and value:
                    try:
                        value = int(value) if value.isdigit() else value
                    except (ValueError, AttributeError):
                        continue  # Skip if conversion fails
                
                data[field_name] = value
            
            # Get the existing item
            result = await db.execute(select(model).where(getattr(model, pk) == item_id))
            existing_item = result.scalars().first()
            
            if not existing_item:
                raise HTTPException(status_code=404, detail="Item not found")
            
            # Update regular fields
            for field_name, value in data.items():
                # Skip relationship fields as they're handled separately
                field_info = next((f for f in model_info.fields if f['name'] == field_name), None)
                if field_info and field_info.get('is_relationship'):
                    continue
                setattr(existing_item, field_name, value)
            
            # Add the item to the session and commit
            db.add(existing_item)
            await db.commit()
            # Refresh to get any database defaults or triggers
            await db.refresh(existing_item)
            
            return RedirectResponse(
                url=f"{self.path}/{model_name}",
                status_code=303
            )
            
        @self.app.post(f"{self.path}/{{model_name}}/{{item_id}}/delete", include_in_schema=False)
        async def delete_model(
            request: Request,
            model_name: str,
            item_id: int,
            db: AsyncSession = Depends(get_db)
        ):
            model = model_registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
                
            # Get primary key column name
            pk = inspect(model).primary_key[0].name
            
            # Get the item with relationships
            result = await db.execute(
                select(model).where(getattr(model, pk) == item_id)
            )
            item = result.scalars().first()
            
            if not item:
                raise HTTPException(status_code=404, detail="Item not found")
            
            # Delete the item
            await db.delete(item)
            await db.commit()
            
            return RedirectResponse(
                url=f"{self.path}/{model_name}",
                status_code=303
            )

def setup_admin(app, path: str = "/admin", models_module: str = None):
    """
    Set up the admin dashboard for the Eagle application.
    
    Args:
        app: The EagleAPI instance
        path: The URL path where the admin dashboard will be mounted
        models_module: Dot path to the module containing SQLAlchemy models (e.g., 'app.models')
    """
    admin = Admin(app, path=path, models_module=models_module)
    return admin
