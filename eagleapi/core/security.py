"""
Security utilities for authentication and authorization.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable, Awaitable

from eagle import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_401_UNAUTHORIZED

from ..core.config import settings
from ..db.user_model import User
from ..schemas.token import TokenData

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)

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

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
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
        result = await db.execute(select(User).where(User.email == token_data.email))
        user = result.scalar_one_or_none()
        
        if user is None:
            raise credentials_exception
        return user
    finally:
        await db.close()

async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user
