"""
User-related Pydantic models for request/response validation.
"""
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class UserBase(BaseModel):
    """Base user model with common fields."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None

class UserCreate(UserBase):
    """Model for creating a new user (includes password)."""
    password: str = Field(..., min_length=8)

class UserInDBBase(UserBase):
    """Base model for user stored in DB (includes ID and disabled status)."""
    id: int
    is_active: bool = True
    is_superuser: bool = False

    class Config:
        from_attributes = True

class UserInDB(UserInDBBase):
    """User model with hashed password for storage in DB."""
    hashed_password: str

class User(UserInDBBase):
    """User model for returning to clients (excludes hashed password)."""
    pass
