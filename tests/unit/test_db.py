"""
Unit tests for Eagle Framework database functionality.
"""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from eagleapi.db.user_model import User

class TestDatabase:
    """Test cases for database operations."""

    @pytest.mark.asyncio
    async def test_database_connection(self, session: AsyncSession):
        """Test that the database connection works."""
        # Simple query to test the connection
        result = await session.execute(select(1))
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_user_model_crud(self, session: AsyncSession):
        """Test the User model CRUD operations."""
        # Test create
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password_123"
        )
        
        # Add to database
        session.add(user)
        await session.commit()
        
        # Test read
        result = await session.execute(select(User).where(User.username == "testuser"))
        db_user = result.scalar_one_or_none()
        
        # Assertions
        assert db_user is not None
        assert db_user.username == "testuser"
        assert db_user.email == "test@example.com"
        assert db_user.hashed_password == "hashed_password_123"
        assert db_user.is_active is True
        assert db_user.is_superuser is False
        
        # Test update
        db_user.email = "updated@example.com"
        await session.commit()
        
        # Verify update
        result = await session.execute(select(User).where(User.username == "testuser"))
        updated_user = result.scalar_one()
        assert updated_user.email == "updated@example.com"
        
        # Test password hashing
        await updated_user.set_password("new_secure_password")
        assert updated_user.verify_password("new_secure_password") is True
        assert updated_user.verify_password("wrong_password") is False
        
        # Test last login update
        await updated_user.update_last_login(session)
        assert updated_user.last_login is not None
        
        # Test delete
        await session.delete(updated_user)
        await session.commit()
        
        # Verify deletion
        result = await session.execute(select(User).where(User.username == "testuser"))
        assert result.scalar_one_or_none() is None
