"""
Admin interface for Eagle Framework.

Provides an admin dashboard for managing application data.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import app
from ..db import db, BaseModel

# Type variables
ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class AdminConfig:
    """Configuration for admin interface."""
    
    def __init__(
        self,
        title: str = "Eagle Admin",
        logo_url: str = "/static/admin/logo.png",
        favicon_url: str = "/static/admin/favicon.ico",
        login_logo: str = "/static/admin/login-logo.png",
        site_name: str = "Eagle Admin",
        site_url: str = "/",
        login_url: str = "/admin/login",
        logout_url: str = "/admin/logout",
        templates_dir: str = "templates/admin",
    ):
        self.title = title
        self.logo_url = logo_url
        self.favicon_url = favicon_url
        self.login_logo = login_logo
        self.site_name = site_name
        self.site_url = site_url
        self.login_url = login_url
        self.logout_url = logout_url
        self.templates_dir = templates_dir
        self._models: Dict[str, Any] = {}
    
    def register_model(
        self,
        model: Type[BaseModel],
        name: Optional[str] = None,
        icon: str = "table",
        create_schema: Optional[Type[BaseModel]] = None,
        update_schema: Optional[Type[BaseModel]] = None,
        list_display: Optional[List[str]] = None,
        search_fields: Optional[List[str]] = None,
        list_filter: Optional[List[str]] = None,
    ) -> None:
        """Register a model with the admin interface."""
        model_name = name or model.__name__
        self._models[model_name] = {
            "model": model,
            "name": model_name,
            "icon": icon,
            "create_schema": create_schema,
            "update_schema": update_schema,
            "list_display": list_display or ["id"],
            "search_fields": search_fields or [],
            "list_filter": list_filter or [],
        }
    
    def get_models(self) -> Dict[str, Any]:
        """Get all registered models."""
        return self._models
    
    def get_model(self, model_name: str) -> Dict[str, Any]:
        """Get a registered model by name."""
        if model_name not in self._models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        return self._models[model_name]


# Default admin configuration
admin_config = AdminConfig()

# Admin router
router = APIRouter(prefix="/admin", tags=["admin"])

# Templates
templates = Jinja2Templates(directory=admin_config.templates_dir)


@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Admin dashboard."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": admin_config.title,
            "site_name": admin_config.site_name,
            "models": admin_config.get_models().values(),
        },
    )


@router.get("/{model_name}", response_class=HTMLResponse)
async def model_list(
    request: Request,
    model_name: str,
    page: int = 1,
    per_page: int = 20,
    db: AsyncSession = Depends(db.get_session),
):
    """List model instances."""
    model_info = admin_config.get_model(model_name)
    model = model_info["model"]
    
    # Get total count
    result = await db.execute(select(model).count())
    total = result.scalar()
    
    # Get paginated results
    offset = (page - 1) * per_page
    result = await db.execute(
        select(model).offset(offset).limit(per_page)
    )
    items = result.scalars().all()
    
    return templates.TemplateResponse(
        "model_list.html",
        {
            "request": request,
            "title": f"{model_name} List | {admin_config.title}",
            "model_info": model_info,
            "items": items,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": (total + per_page - 1) // per_page,
        },
    )


@router.get("/{model_name}/create", response_class=HTMLResponse)
async def model_create_form(request: Request, model_name: str):
    """Show create form for a model."""
    model_info = admin_config.get_model(model_name)
    
    if not model_info["create_schema"]:
        raise HTTPException(status_code=400, detail="Create not supported for this model")
    
    return templates.TemplateResponse(
        "model_form.html",
        {
            "request": request,
            "title": f"Create {model_name} | {admin_config.title}",
            "model_info": model_info,
            "action": f"/admin/{model_name}/create",
            "method": "post",
            "item": {},
        },
    )


@router.get("/{model_name}/{id}", response_class=HTMLResponse)
async def model_detail(
    request: Request,
    model_name: str,
    id: int,
    db: AsyncSession = Depends(db.get_session),
):
    """View a model instance."""
    model_info = admin_config.get_model(model_name)
    model = model_info["model"]
    
    result = await db.execute(select(model).where(model.id == id))
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail=f"{model_name} not found")
    
    return templates.TemplateResponse(
        "model_detail.html",
        {
            "request": request,
            "title": f"{model_name} Detail | {admin_config.title}",
            "model_info": model_info,
            "item": item,
        },
    )


# Include the admin router in the app
app.include_router(router)

# Export public API
__all__ = [
    'AdminConfig', 'admin_config', 'ModelType', 'CreateSchemaType', 'UpdateSchemaType'
]
