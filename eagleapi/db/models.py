"""
Database models for the Eagle Framework.
"""
from sqlalchemy import Column, String, Boolean, DateTime, select
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from . import Base, BaseModel
from ..services.auth import verify_password, get_password_hash

class User(BaseModel, Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    email = Column(String(100), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self) -> str:
        return f"<User {self.username}>"
    
    @classmethod
    async def get_by_username(cls, db: 'AsyncSession', username: str) -> Optional['User']:
        """Get a user by username."""
        result = await db.execute(select(cls).where(cls.username == username))
        return result.scalars().first()
    
    @classmethod
    async def get_by_email(cls, db: 'AsyncSession', email: str) -> Optional['User']:
        """Get a user by email."""
        result = await db.execute(select(cls).where(cls.email == email))
        return result.scalars().first()
    
    def set_password(self, password: str) -> None:
        """Set the user's password."""
        self.hashed_password = get_password_hash(password)
    
    def verify_password(self, password: str) -> bool:
        """Verify the user's password."""
        if not self.hashed_password:
            return False
        return verify_password(password, self.hashed_password)
    
    async def update_last_login(self, db: 'AsyncSession') -> None:
        """Update the user's last login timestamp."""
        self.last_login = datetime.utcnow()
        await self.save(db)
