"""
EagleAPI Admin: Professional, auto-configurable admin module for SQLAlchemy + Pydantic

Features:
- Auto-register SQLAlchemy models and Pydantic schemas
- Auto-generate CRUD endpoints
- Auto-generate OpenAPI-compatible forms
- Easy configuration (list display, search, filters)
- Production-ready, extensible, FastAPI-based
"""

import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from eagleapi.db.session import get_db
import inspect
import logging

import os

# Setup Jinja2 templates for admin
TEMPLATES_PATH = os.path.join(os.path.dirname(__file__), 'templates')
templates = Jinja2Templates(directory=TEMPLATES_PATH)

class AdminModelRegistry:
    def __init__(self):
        self._models: List[Dict[str, Any]] = []
    
    def register(self, model_config: Dict[str, Any]) -> None:
        self._models.append(model_config)
    
    def get_models(self) -> List[Dict[str, Any]]:
        return self._models

# Global registry instance
_admin_model_registry = AdminModelRegistry()

def register_admin(
    *,
    model: Type[Any],
    list_display: Optional[List[str]] = None,
    search_fields: Optional[List[str]] = None,
    filters: Optional[List[str]] = None,
    permissions: Optional[Dict[str, Any]] = None,
    custom_routes: Optional[List[APIRouter]] = None,
    schema_config: Optional[dict] = None,
):
    """
    Register a model with the admin interface. All CRUD API and HTML endpoints will be auto-registered.

    Args:
        model: SQLAlchemy model class
        list_display: Fields to show in list view
        search_fields, filters, permissions, custom_routes: Optional config
        schema_config: Dict for schema generator config (exclude/include columns, etc)

    Example:
        register_admin(
            model=User,
            list_display=["id", "email", "is_active"]
        )
    """
    from eagleapi.db.schemas.generator import generate_schema, SchemaConfig
    config = SchemaConfig(**(schema_config or {}))
    create_schema = generate_schema(model, mode="create", config=config)
    read_schema = generate_schema(model, mode="response", config=config)
    update_schema = generate_schema(model, mode="update", config=config)
    # If list_display is not provided, use all fields from the response schema
    if list_display is None:
        list_display = list(read_schema.model_fields.keys())
    _admin_model_registry.register({
        "model": model,
        "create_schema": create_schema,
        "read_schema": read_schema,
        "update_schema": update_schema,
        "list_display": list_display,
        "search_fields": search_fields or [],
        "filters": filters or [],
        "permissions": permissions or {},
        "custom_routes": custom_routes or [],
    })

def get_admin_router() -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])
    logger = logging.getLogger("eagleapi.admin")
    from sqlalchemy import select
    import json
    import datetime
    
    # --- ADMIN INDEX ROUTE ---
    @router.get("/", response_class=HTMLResponse, name="admin_index")
    async def admin_index(request: Request):
        models = [
            {
                "display_name": entry["model"].__name__,
                "name": entry["model"].__tablename__
            }
            for entry in _admin_model_registry.get_models()
        ]
        return templates.TemplateResponse(
            "admin/index.html",
            {
                "request": request,
                "models": models,
            },
        )

    # Get a fresh copy of registered models for this router
    registered_models = _admin_model_registry.get_models()
    
    # Create a mapping of model names to their config for easy lookup
    model_registry = {entry["model"].__tablename__: entry for entry in registered_models}
    
    # Register routes for each model
    for model_name, entry in model_registry.items():
        # Create a local copy of the model and schemas for this iteration
        current_model = entry["model"]
        current_model_name = model_name
        current_create_schema = entry["create_schema"]
        current_read_schema = entry["read_schema"]
        current_update_schema = entry["update_schema"]
        current_list_display = entry["list_display"]
        
        # Log the model being registered
        logger.info(f"Registering admin routes for model: {current_model_name} ({current_model.__name__})")

        @router.get(f"/{current_model_name}/", response_class=HTMLResponse, name=f"admin_{current_model_name}_list")
        async def admin_list(request: Request, db: AsyncSession = Depends(get_db), model_name: str = current_model_name, model: Type[Any] = current_model, list_display: List[str] = current_list_display):
            try:
                result = await db.execute(select(model))
                orm_objs = result.scalars().all()
                objects = []
                first_obj_dict = None
                for obj in orm_objs:
                    obj_dict = _sa_to_dict(obj)
                    if first_obj_dict is None and obj_dict:
                        first_obj_dict = obj_dict
                    objects.append(obj_dict)
                first_obj_keys = list(first_obj_dict.keys()) if first_obj_dict else []
                first_obj_json = json.dumps(first_obj_dict, indent=2, ensure_ascii=False) if first_obj_dict else None
                return templates.TemplateResponse("admin/list.html", {"request": request, "objects": objects, "model_name": model_name, "model_display_name": model.__name__, "list_display": list_display, "first_obj_keys": first_obj_keys, "first_obj_json": first_obj_json, "object_count": len(objects)})

            except Exception as exc:
                logger.error(f"Error in admin list view for {model_name}", exc_info=True)
                # Return a simple error response that won't trigger template rendering
                return HTMLResponse(f"<h2>Error loading {model_name} list</h2><pre>{str(exc)}</pre>", status_code=500)
                return HTMLResponse(f"<h2>Admin List Error: {model_name}</h2><pre>{exc}</pre>", status_code=500)

        @router.get(f"/{current_model_name}/new/", response_class=HTMLResponse, name=f"admin_create_{current_model_name}")
        async def admin_create_form(
            request: Request,
            model_name: str = current_model_name,
            create_schema: Type[BaseModel] = current_create_schema
        ):
            # Create an empty form with default values
            form_data = create_schema.construct()
            fields = _get_form_fields_from_schema(create_schema)
            return templates.TemplateResponse(
                "admin/form.html",
                {
                    "request": request,
                    "model_name": model_name,
                    "model_display_name": current_model.__name__,
                    "action": f"/{model_name}/new/",
                    "fields": fields,
                    "is_edit": False,
                },
            )

        @router.post(f"/{current_model_name}/new/", response_class=HTMLResponse, name=f"admin_create_{current_model_name}_post")
        async def admin_create_post(
            request: Request,
            db: AsyncSession = Depends(get_db),
            model_name: str = current_model_name,
            model: Type[Any] = current_model,
            create_schema: Type[BaseModel] = current_create_schema,
            list_display: List[str] = current_list_display
        ):
            try:
                form_data = await request.form()
                fields = _get_form_fields_from_schema(create_schema)
                data = {}
                for field in fields:
                    value = form_data.get(field['name'])
                    if field['type'] == 'checkbox':
                        data[field['name']] = value == 'on'
                    else:
                        data[field['name']] = value
                # Remove any empty string values for fields that should be None
                for k, v in data.items():
                    if v == '':
                        data[k] = None
                # Directly create the model instance
                item = model(**data)
                db.add(item)
                await db.commit()
                await db.refresh(item)
                # Redirect to list view
                return RedirectResponse(url=f"/admin/{model_name}/", status_code=303)
            except Exception as exc:
                await db.rollback()
                logger.error(f"Admin create post error for {model_name}: {exc}", exc_info=True)
                return HTMLResponse(f"<h2>Admin Create Error: {model_name}</h2><pre>{exc}</pre>", status_code=500)

        # EDIT FORM
        @router.get(f"/{current_model_name}/edit/{{item_id}}/", response_class=HTMLResponse, name=f"admin_edit_{current_model_name}")
        async def admin_edit_form(
            request: Request, 
            item_id: int, 
            db: AsyncSession = Depends(get_db),
            model_name: str = current_model_name,
            model: Type[Any] = current_model,
            update_schema: Type[BaseModel] = current_update_schema
        ):
            try:
                # Get the item to edit
                result = await db.execute(select(model).where(model.id == item_id))
                item = result.scalar_one_or_none()
                if not item:
                    raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
                
                # Convert to Pydantic model for form
                pydantic_item = update_schema.model_validate(item) if hasattr(update_schema, 'model_validate') else update_schema.from_orm(item)
                item_dict = pydantic_item.model_dump() if hasattr(pydantic_item, 'model_dump') else pydantic_item.dict()
                
                # Get form fields with current values
                fields = _get_form_fields_from_schema(update_schema, item_dict)
                
                return templates.TemplateResponse(
                    "admin/form.html",
                    {
                        "request": request,
                        "model_name": model_name,
                        "model_display_name": model.__name__,
                        "item_id": item_id,
                        "fields": fields,
                        "is_edit": True,
                        "action": f"/admin/{model_name}/edit/{item_id}/",
                    },
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.error(f"Admin edit form error for {model_name} {item_id}: {exc}", exc_info=True)
                return HTMLResponse(f"<h2>Admin Edit Error: {model_name} {item_id}</h2><pre>{exc}</pre>", status_code=500)

        @router.post(f"/{current_model_name}/edit/{{item_id}}/", response_class=HTMLResponse, name=f"admin_edit_{current_model_name}_post")
        async def admin_edit_post(
            request: Request, 
            item_id: int, 
            db: AsyncSession = Depends(get_db),
            model_name: str = current_model_name,
            model: Type[Any] = current_model,
            update_schema: Type[BaseModel] = current_update_schema
        ):
            try:
                form_data = await request.form()
                # Get the existing item
                result = await db.execute(select(model).where(model.id == item_id))
                item = result.scalar_one_or_none()
                if not item:
                    raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
                # Convert form data to dict and handle checkboxes
                fields = _get_form_fields_from_schema(update_schema)
                data = {}
                for field in fields:
                    value = form_data.get(field['name'])
                    if field['type'] == 'checkbox':
                        data[field['name']] = value == 'on'
                    else:
                        data[field['name']] = value
                # Remove any empty string values for fields that should be None
                for k, v in data.items():
                    if v == '':
                        data[k] = None
                # Update the item directly
                for key, value in data.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                # Save changes
                await db.commit()
                await db.refresh(item)
                # Redirect to list view
                return RedirectResponse(url=f"/admin/{model_name}/", status_code=303)
            except HTTPException:
                raise
            except Exception as exc:
                await db.rollback()
                logger.error(f"Admin edit post error for {model_name} {item_id}: {exc}", exc_info=True)
                return HTMLResponse(f"<h2>Admin Edit Error: {model_name} {item_id}</h2><pre>{exc}</pre>", status_code=500)


        # DELETE CONFIRMATION
        @router.get(f"/{current_model_name}/delete/{{item_id}}/", response_class=HTMLResponse, name=f"admin_delete_{current_model_name}")
        async def admin_delete_confirm(
            request: Request, 
            item_id: int, 
            db: AsyncSession = Depends(get_db),
            model_name: str = current_model_name,
            model: Type[Any] = current_model
        ):
            try:
                result = await db.execute(select(model).where(model.id == item_id))
                item = result.scalar_one_or_none()
                if not item:
                    raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
                
                return templates.TemplateResponse(
                    "admin/confirm_delete.html",
                    {
                        "request": request,
                        "model_name": model_name,
                        "model_display_name": model.__name__,
                        "item": item,
                        "back_url": f"/admin/{model_name}/",
                    },
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.error(f"Admin delete confirm error for {model_name} {item_id}: {exc}", exc_info=True)
                return HTMLResponse(f"<h2>Admin Delete Error: {model_name} {item_id}</h2><pre>{exc}</pre>", status_code=500)

        @router.post(f"/{current_model_name}/delete/{{item_id}}/", response_class=HTMLResponse, name=f"admin_delete_{current_model_name}_post")
        async def admin_delete_post(
            request: Request, 
            item_id: int, 
            db: AsyncSession = Depends(get_db),
            model_name: str = current_model_name,
            model: Type[Any] = current_model
        ):
            try:
                result = await db.execute(select(model).where(model.id == item_id))
                item = result.scalar_one_or_none()
                if not item:
                    raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
                
                await db.delete(item)
                await db.commit()
                
                # Redirect to list view
                return RedirectResponse(url=f"/admin/{model_name}/", status_code=303)
                
            except HTTPException:
                raise
            except Exception as exc:
                await db.rollback()
                logger.error(f"Admin delete post error for {model_name} {item_id}: {exc}", exc_info=True)
                return HTMLResponse(f"<h2>Admin Delete Error: {model_name} {item_id}</h2><pre>{exc}</pre>", status_code=500)

        # --- Helper: SQLAlchemy to dict (with datetime support) ---
        def _sa_to_dict(obj):
            if obj is None:
                return None
            result = {}
            for key in obj.__table__.columns.keys():
                value = getattr(obj, key)
                # Convert datetime to ISO format string
                if isinstance(value, datetime.datetime):
                    value = value.isoformat()
                # Handle LogLevel enum by converting to string
                elif hasattr(value, 'name') and hasattr(value, 'value') and type(value).__module__ != 'builtins':
                    value = str(value)
                result[key] = value
            return result

        # --- CRUD API ROUTES (kept for programmatic/React use) ---
        @router.get(f"/api/{current_model_name}/", name=f"api_list_{current_model_name}")
        async def api_list(
            skip: int = 0,
            limit: int = 100,
            db: AsyncSession = Depends(get_db),
            model: Type[Any] = current_model
        ):
            result = await db.execute(select(model).offset(skip).limit(limit))
            items = result.scalars().all()
            return [_sa_to_dict(item) for item in items]

        @router.post(f"/api/{current_model_name}/", status_code=201, name=f"api_create_{current_model_name}")
        async def api_create(
            request: Request,
            db: AsyncSession = Depends(get_db),
            model: Type[Any] = current_model
        ):
            data = await request.json()
            item = model(**data)
            db.add(item)
            await db.commit()
            await db.refresh(item)
            return _sa_to_dict(item)

        @router.get(f"/api/{current_model_name}/{{item_id}}/", name=f"api_read_{current_model_name}")
        async def api_read(
            item_id: int,
            db: AsyncSession = Depends(get_db),
            model: Type[Any] = current_model
        ):
            result = await db.execute(select(model).where(model.id == item_id))
            item = result.scalar_one_or_none()
            if item is None:
                raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
            return _sa_to_dict(item)

        @router.put(f"/api/{current_model_name}/{{item_id}}/", name=f"api_update_{current_model_name}")
        async def api_update(
            item_id: int,
            request: Request,
            db: AsyncSession = Depends(get_db),
            model: Type[Any] = current_model
        ):
            data = await request.json()
            result = await db.execute(select(model).where(model.id == item_id))
            db_item = result.scalar_one_or_none()
            if db_item is None:
                raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
            for key, value in data.items():
                if hasattr(db_item, key):
                    setattr(db_item, key, value)
            await db.commit()
            await db.refresh(db_item)
            return _sa_to_dict(db_item)

        @router.delete(f"/api/{current_model_name}/{{item_id}}/", status_code=204, name=f"api_delete_{current_model_name}")
        async def api_delete(
            item_id: int,
            db: AsyncSession = Depends(get_db),
            model: Type[Any] = current_model
        ):
            result = await db.execute(select(model).where(model.id == item_id))
            db_item = result.scalar_one_or_none()
            if db_item is None:
                raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
            await db.delete(db_item)
            await db.commit()
            return Response(status_code=204)


        # Register any custom routes
        for custom_router in entry["custom_routes"]:
            router.include_router(custom_router)

    return router

def _get_form_fields_from_schema(schema, values=None):
    # Returns a list of dicts: [{name, label, type, value, required, error}]
    values = values or {}
    fields = []
    for name, field in schema.__fields__.items():
        field_type = 'text'
        annotation = field.annotation
        field_info = field
        
        # Determine field type from annotation
        if annotation in (int,):
            field_type = 'number'
        elif annotation in (float,):
            field_type = 'number'
        elif annotation in (bool,):
            field_type = 'checkbox'
        elif annotation in (str,):
            field_type = 'text'
            
        # Get field title and required status
        title = getattr(field_info, 'title', None) or name.capitalize()
        is_required = getattr(field_info, 'is_required', False)
        
        fields.append({
            'name': name,
            'label': title,
            'type': field_type,
            'value': values.get(name, ''),
            'required': is_required,
            'error': None,
        })
    return fields

def _validate_form_data(schema, data, fields):
    try:
        schema(**data)
        return False
    except Exception as e:
        # Attach error to fields
        for f in fields:
            if f['name'] in str(e):
                f['error'] = str(e)
        return True

def register_model_to_admin(_cls=None, **decorator_kwargs):
    """
    Decorator to register a SQLAlchemy model with the admin.
    Can be used as @register_model_to_admin or @register_model_to_admin(list_display=[...], ...)
    """
    def wrap(cls):
        # Use the same registry function as register_admin
        register_admin(model=cls, **decorator_kwargs)
        return cls

    # Handle both @register_model_to_admin and @register_model_to_admin()
    if _cls is None:
        return wrap
    return wrap(_cls)

class AdminApp:
    """
    Usage:
        from eagleapi.admin import AdminApp, register_admin, register_model_to_admin
        # register_admin(...)
        # or
        # @register_model_to_admin(list_display=[...])
        # class MyModel(...): ...
        admin_app = AdminApp()
        app.include_router(admin_app.router)
    """
    def __init__(self):
        self.router = get_admin_router()
