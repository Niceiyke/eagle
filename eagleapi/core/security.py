"""
Security utilities for authentication and authorization.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable, Awaitable

from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status
from starlette.status import HTTP_401_UNAUTHORIZED

from ..core.config import settings
from ..core.password import verify_password, get_password_hash
from ..db.models.models import AdminUser
from ..schemas.token import TokenData

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

# This will be set by the app startup event
_get_db_session: Optional[Callable[[], Awaitable[AsyncSession]]] = None

def set_db_session_getter(getter: Callable[[], Awaitable[AsyncSession]]) -> None:
    """Set the database session getter function."""
    global _get_db_session
    _get_db_session = getter

async def get_current_user(token: str = Depends(oauth2_scheme)) -> AdminUser:
    """Get the current user from a JWT token."""
    if _get_db_session is None:
        raise RuntimeError("Database session getter not set. Call set_db_session_getter first.")
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    # Get a new database session
    db = await _get_db_session()
    try:
        from sqlalchemy import select
        result = await db.execute(select(AdminUser).where(AdminUser.email == token_data.email))
        user = result.scalar_one_or_none()
        
        if user is None:
            raise credentials_exception
        return user
    finally:
        await db.close()

async def get_current_active_user(
    current_user: AdminUser = Depends(get_current_user),
) -> AdminUser:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user
