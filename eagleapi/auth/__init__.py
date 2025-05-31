"""
Authentication and authorization module for Eagle Framework.

Provides JWT authentication, OAuth2 integration, and role-based access control.
"""
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import db,get_db
from .. import app
from ..services.auth import get_user_by_email,get_user_by_username,authenticate_user,create_user
from ..schemas.user import UserCreate
from ..core import security
from ..core.config import settings
# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


from pydantic import BaseModel as PydanticBaseModel

class Token(PydanticBaseModel):
    """Token response model."""
    access_token: str
    token_type: str

    class Config:
        orm_mode = True
        from_attributes = True


class TokenData(PydanticBaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: List[str] = []

    class Config:
        orm_mode = True
        from_attributes = True


class User(PydanticBaseModel):
    """Base user model."""
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: bool = False
    is_superuser: bool = False
    scopes: List[str] = []
    
    class Config:
        orm_mode = True
        from_attributes = True


class UserInDB(User):
    """User model with password hash."""
    hashed_password: str
    
    class Config:
        orm_mode = True
        from_attributes = True


class UserCreate(PydanticBaseModel):
    """User creation model."""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    
    class Config:
        orm_mode = True
        from_attributes = True


class UserUpdate(PydanticBaseModel):
    """User update model."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    is_superuser: Optional[bool] = None
    
    class Config:
        orm_mode = True
        from_attributes = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create a new access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(db.get_session)
) -> User:
    """Get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # In a real app, you would fetch the user from the database here
    # user = await get_user(db, username=token_data.username)
    # if user is None:
    #     raise credentials_exception
    # return user
    
    # For now, return a mock user
    return User(
        id=1,
        username=token_data.username,
        email="user@example.com",
        is_superuser=False
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Authentication routers
from eagleapi import APIRouter

router = APIRouter(tags=["auth"])


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """OAuth2 compatible token login."""
    # In a real app, you would validate the username and password against your database
    # user = await authenticate_user(db, form_data.username, form_data.password)
    # if not user:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Incorrect username or password",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )
    
    # For now, accept any username/password
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username, "scopes": form_data.scopes},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}



@router.post(
    "/register",
    response_model=UserInDB,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    response_description="The created user"
)
async def register_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Register a new user.
    
    - **username**: must be unique
    - **email**: must be a valid email and unique
    - **password**: at least 8 characters
    - **full_name**: user's full name
    """
    # Check if user already exists by email or username
    db_user = await get_user_by_email(db, email=user_in.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    db_user = await get_user_by_username(db, username=user_in.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    user = await create_user(db=db, user_in=user_in)
    return user

@router.post(
    "/token",
    response_model=Token,
    summary="OAuth2 token endpoint",
    response_description="Access token for authentication"
)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    """
    OAuth2 compatible token login, get an access token for future requests.
    
    - **username**: your username or email
    - **password**: your password
    
    Returns an access token that can be used in the Authorization header.
    """
    user = await authenticate_user(
        db, form_data.username, form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get(
    "/me",
    response_model=UserInDB,
    summary="Get current user information",
    response_description="The current user's information"
)
async def read_users_me(
    current_user: User = Depends(security.get_current_active_user)
) -> Any:
    """
    Get current user information.
    
    Requires authentication with a valid access token.
    """
    return current_user









# Export public API
__all__ = [
    'Token', 'User', 'UserInDB', 'UserCreate', 'UserUpdate',
    'get_current_user', 'get_current_active_user', 'create_access_token',
    'verify_password', 'get_password_hash', 'router'
]
