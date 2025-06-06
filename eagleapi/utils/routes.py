"""
Utility endpoints for the Eagle API.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from ..db import get_db
from .typescript import generate_all_typescript_interfaces

router = APIRouter(
    prefix="/utils",
    tags=["utils"],
    responses={404: {"description": "Not found"}},
)

@router.get(
    "/typescript-models",
    response_class=PlainTextResponse,
    summary="Generate TypeScript interfaces from SQLAlchemy models",
    description="""
    Generates TypeScript interfaces for all SQLAlchemy models in the application.
    The output can be saved directly to a .ts file in your frontend project.
    """,
    responses={
        200: {
            "content": {"text/plain": {"example": "export interface User {\n  id: number;\n  name: string;\n  email: string;\n}"}},
            "description": "TypeScript interfaces for all models",
        }
    }
)
async def get_typescript_models():
    """
    Generate TypeScript interfaces for all SQLAlchemy models.
    
    This endpoint returns TypeScript interfaces that match your database models,
    which can be used in your frontend code for type safety.
    """
    try:
        typescript_code = generate_all_typescript_interfaces()
        return PlainTextResponse(
            content=typescript_code,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=models.generated.ts"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate TypeScript interfaces: {str(e)}"
        )
