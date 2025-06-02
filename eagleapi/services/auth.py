"""
Authentication service for user registration, login, and token management.
"""
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core import security
from ..core.config import settings
from ..db.models.models import AdminUser
from ..schemas.user import UserCreate

async def get_user(db: AsyncSession, user_id: int) -> Optional[AdminUser]:
    """Get a user by ID."""
    result = await db.execute(select(AdminUser).filter(AdminUser.id == user_id))
    return result.scalar_one_or_none()

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[AdminUser]:
    """Get a user by email."""
    result = await db.execute(select(AdminUser).filter(AdminUser.email == email))
    return result.scalar_one_or_none()

async def get_user_by_username(db: AsyncSession, username: str) -> Optional[AdminUser]:
    """Get a user by username."""
    result = await db.execute(select(AdminUser).filter(AdminUser.username == username))
    return result.scalar_one_or_none()

async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[AdminUser]:
    """Authenticate a user by email and password."""
    user = await get_user_by_email(db, email)
    if not user:
        return None
    if not security.verify_password(password, user.hashed_password):
        return None
    return user

async def create_user(db: AsyncSession, user_in: UserCreate) -> AdminUser:
    """Create a new user with hashed password."""
    hashed_password = security.get_password_hash(user_in.password)
    db_user = AdminUser(
        email=user_in.email,
        username=user_in.username,
        full_name=user_in.full_name,
        hashed_password=hashed_password,
        is_active=True,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user
