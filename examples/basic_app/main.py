"""
Basic Eagle Framework Example

This example demonstrates the core features of the Eagle framework.
"""
import os
from typing import List, Optional

from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from eagleapi import Eagle, Request, Response
from eagleapi.db import BaseModel as Base, db, Column, Integer, String, Boolean, create_async_engine
from eagleapi.auth import get_current_user, User, get_password_hash
from eagleapi.admin import admin_config

# Create the application
app = Eagle(
    title="Eagle Example App",
    description="A basic example of the Eagle framework",
    version="0.1.0",
)

# Database Models
class DBUser(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)

# Pydantic Models
class UserBase(BaseModel):
    username: str
    email: EmailStr
    is_active: Optional[bool] = True
    is_superuser: bool = False

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    
    class Config:
        orm_mode = True

# Register models with admin
admin_config.register_model(
    DBUser,
    name="Users",
    icon="users",
    create_schema=UserCreate,
    update_schema=UserBase,
    list_display=["id", "username", "email", "is_active"],
    search_fields=["username", "email"],
    list_filter=["is_active", "is_superuser"]
)

# Routes
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Eagle Framework!"}

@app.get("/users/me", response_model=UserInDB)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return current_user

@app.post("/users/", response_model=UserInDB, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: AsyncSession = Depends(db.get_session)):
    """Create a new user."""
    # Check if user already exists
    result = await db.execute(select(DBUser).where(DBUser.email == user.email))
    if result.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = DBUser(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
    )
    
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    
    return db_user

# Database initialization
@app.on_event("startup")
async def startup():
    """Initialize database and create admin user."""
    # Create database tables
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create admin user if it doesn't exist
    async with db.get_session() as session:
        result = await session.execute(select(DBUser).where(DBUser.email == "admin@example.com"))
        admin_user = result.scalars().first()
        
        if not admin_user:
            hashed_password = get_password_hash("admin")
            admin_user = DBUser(
                username="admin",
                email="admin@example.com",
                hashed_password=hashed_password,
                is_active=True,
                is_superuser=True,
            )
            session.add(admin_user)
            await session.commit()
            await session.refresh(admin_user)
            print("Created admin user:", admin_user.username)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
