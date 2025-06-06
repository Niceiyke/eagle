# """
# Database models for the Eagle Framework.
# """
# from sqlalchemy import Column, String, Boolean, DateTime, select, func
# from datetime import datetime
# from typing import Optional, TYPE_CHECKING, List, Dict, Any
# from pydantic import BaseModel as PydanticBaseModel

# from eagleapi.db import Base
# from eagleapi.core.password import get_password_hash, verify_password
# from eagleapi.admin import register_model_to_admin


# if TYPE_CHECKING:
#     from sqlalchemy.ext.asyncio import AsyncSession


# class ModelBaseSchema(PydanticBaseModel):
#     """Base schema for all models."""
#     id: int
#     created_at: datetime
#     updated_at: datetime

#     class Config:
#         from_attributes = True

# @register_model_to_admin(name="AdminUser")
# class AdminUser(Base):
#     """User model for authentication and authorization.
    
#     Attributes:
#         email (str): User's email address, unique and required
#         username (str): User's username, unique and required
#         full_name (str): User's full name, optional
#         hashed_password (str): Hashed password
#         is_active (bool): Whether the user is active
#         is_superuser (bool): Whether the user has superuser privileges
#         last_login (datetime): Timestamp of last login
#     """
#     __tablename__ = "admin_users"
    
#     email = Column(String(100), unique=True, index=True, nullable=False)
#     username = Column(String(50), unique=True, index=True, nullable=False)
#     full_name = Column(String(100), nullable=True)
#     hashed_password = Column(String(255), nullable=False)
#     is_active = Column(Boolean(), default=True)
#     is_superuser = Column(Boolean(), default=False)
#     last_login = Column(DateTime, nullable=True)
    
#     __table_args__ = (
#         dict(
#             comment="Admin users table",
#             sqlite_autoincrement=True
#         ),
#     )
    
#     def __repr__(self) -> str:
#         return f"<AdminUser {self.username}>"
    
#     @classmethod
#     async def get_by_username(cls, db: 'AsyncSession', username: str) -> Optional['AdminUser']:
#         """Get a user by username.
        
#         Args:
#             db: Async database session
#             username: Username to search for
            
#         Returns:
#             AdminUser instance or None if not found
#         """
#         result = await db.execute(select(cls).where(cls.username == username))
#         return result.scalars().first()
    
#     @classmethod
#     async def get_by_email(cls, db: 'AsyncSession', email: str) -> Optional['AdminUser']:
#         """Get a user by email.
        
#         Args:
#             db: Async database session
#             email: Email address to search for
            
#         Returns:
#             AdminUser instance or None if not found
#         """
#         result = await db.execute(select(cls).where(cls.email == email))
#         return result.scalars().first()
    
#     @classmethod
#     async def count(cls, db: 'AsyncSession') -> int:
#         """Get the total count of users.
        
#         Args:
#             db: Async database session
            
#         Returns:
#             Total number of users
#         """
#         result = await db.execute(select(func.count()).select_from(cls))
#         return result.scalar()
    
#     @classmethod
#     async def get_all(
#         cls,
#         db: 'AsyncSession',
#         skip: int = 0,
#         limit: int = 100
#     ) -> List['AdminUser']:
#         """Get all users with pagination.
        
#         Args:
#             db: Async database session
#             skip: Number of records to skip
#             limit: Maximum number of records to return
            
#         Returns:
#             List of AdminUser instances
#         """
#         result = await db.execute(
#             select(cls).offset(skip).limit(limit)
#         )
#         return result.scalars().all()
    
#     def set_password(self, password: str) -> None:
#         """Set the user's password.
        
#         Args:
#             password: Plain text password to hash and store
#         """
#         self.hashed_password = get_password_hash(password)
    
#     def verify_password(self, password: str) -> bool:
#         """Verify the user's password.
        
#         Args:
#             password: Plain text password to verify
            
#         Returns:
#             True if password matches, False otherwise
#         """
#         if not self.hashed_password:
#             return False
#         return verify_password(password, self.hashed_password)
    
#     async def update_last_login(self, db: 'AsyncSession') -> None:
#         """Update the user's last login timestamp.
        
#         Args:
#             db: Async database session
#         """
#         self.last_login = datetime.now()
#         await self.save(db)
    
#     async def deactivate(self, db: 'AsyncSession') -> None:
#         """Deactivate the user account.
        
#         Args:
#             db: Async database session
#         """
#         self.is_active = False
#         await self.save(db)
    
#     async def activate(self, db: 'AsyncSession') -> None:
#         """Activate the user account.
        
#         Args:
#             db: Async database session
#         """
#         self.is_active = True
#         await self.save(db)
    
#     async def promote_to_superuser(self, db: 'AsyncSession') -> None:
#         """Promote user to superuser.
        
#         Args:
#             db: Async database session
#         """
#         self.is_superuser = True
#         await self.save(db)
    
#     async def demote_from_superuser(self, db: 'AsyncSession') -> None:
#         """Demote user from superuser.
        
#         Args:
#             db: Async database session
#         """
#         self.is_superuser = False
#         await self.save(db)
    
#     async def update_profile(
#         self,
#         db: 'AsyncSession',
#         **kwargs: Dict[str, Any]
#     ) -> None:
#         """Update user profile information.
        
#         Args:
#             db: Async database session
#             kwargs: Dictionary of fields to update
#         """
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#         await self.save(db)
