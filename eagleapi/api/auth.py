"""
Authentication routes for the Eagle Framework.
"""
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from ..core import security
from ..core.config import settings
from ..db import get_db
from ..schemas.token import Token
from ..schemas.user import User, UserCreate, UserInDB
from ..services import auth as auth_service

# Create router
router = APIRouter()

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
    db_user = await auth_service.get_user_by_email(db, email=user_in.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    db_user = await auth_service.get_user_by_username(db, username=user_in.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    user = await auth_service.create_user(db=db, user_in=user_in)
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
    user = await auth_service.authenticate_user(
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
