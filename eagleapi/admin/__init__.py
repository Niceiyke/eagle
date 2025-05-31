"""
Eagle Admin Dashboard

This module provides an automatic admin interface for Eagle API applications.
"""
from typing import Optional, Type, Dict, Any, List, TypeVar, Callable
from fastapi import Request, Depends, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy import inspect, select, update, delete, insert
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel as PydanticBaseModel
from eagleapi.db import get_db
from pathlib import Path
import os
import importlib
import inspect as py_inspect
from typing import Any, Dict, List, Optional, Type, TypeVar
from functools import wraps

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


class ModelInfo(PydanticBaseModel):
    name: str
    table_name: str
    fields: List[Dict[str, Any]]

class ModelRegistry:
    _instance = None
    _models: Dict[str, Type[ModelType]] = {}
    
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
        """
        model_name = name or model.__name__
        cls._models[model_name] = model
    
    @classmethod
    def get_models(cls) -> Dict[str, Type[ModelType]]:
        """Get all registered models."""
        return cls._models
    
    @classmethod
    def get_model(cls, model_name: str) -> Optional[Type[ModelType]]:
        """Get a model by name."""
        return cls._models.get(model_name)
    
    @classmethod
    def get_model_info(cls, model: Type[ModelType]) -> ModelInfo:
        """Get information about a model's fields."""
        mapper = inspect(model)
        fields = []
        
        for column in mapper.columns:
            field_info = {
                "name": column.name,
                "type": str(column.type.__visit_name__).upper() if hasattr(column.type, '__visit_name__') else str(column.type.python_type.__name__).upper(),
                "nullable": column.nullable,
                "primary_key": column.primary_key,
                "default": column.default.arg if column.default else None
            }
            fields.append(field_info)
            
        return ModelInfo(
            name=model.__name__,
            table_name=model.__tablename__,
            fields=fields
        )
    
    @classmethod
    def scan_models(cls, models_module: str) -> None:
        """Scan a module for SQLAlchemy models and register them."""
        try:
            module = importlib.import_module(models_module)
            for name, obj in py_inspect.getmembers(module):
                if (
                    py_inspect.isclass(obj) 
                    and hasattr(obj, '__tablename__')
                    and obj.__module__ == models_module
                ):
                    cls.register(obj)
        except ImportError as e:
            print(f"Error scanning models: {e}")

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
        
        # Import and include API routes
        from . import api
        self.app.include_router(api.router, prefix=f"{self.path}/api")
        
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
            model = model_registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
                
            # Get all records
            result = await db.execute(select(model))
            items = result.scalars().all()
            
            model_info = model_registry.get_model_info(model)
            
            return self.templates.TemplateResponse(
                "model_list.html",
                {
                    "request": request,
                    "title": f"{model_name} List",
                    "model_name": model_name,
                    "items": items,
                    "fields": [field['name'] for field in model_info.fields],
                    "admin_path": self.path
                }
            )
            
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
                if field_name in form_data:
                    value = form_data[field_name]
                    
                    # Skip empty values
                    if value == '':
                        continue
                        
                    # Convert based on field type
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
                    "admin_path": self.path
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
                if field_name in form_data:
                    value = form_data[field_name]
                    
                    # Skip empty values for non-required fields
                    if value == '':
                        continue
                        
                    # Convert based on field type
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
                    
                    data[field_name] = value
            
            # Update record
            if data:  # Only proceed if we have data to update
                stmt = (
                    update(model)
                    .where(getattr(model, pk) == item_id)
                    .values(**data)
                )
                await db.execute(stmt)
                await db.commit()
            
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
            
            # Delete record
            stmt = (
                delete(model)
                .where(getattr(model, pk) == item_id)
            )
            await db.execute(stmt)
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
