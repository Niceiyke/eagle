"""
Database tests for Eagle Framework.
"""
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from eagleapi.db import Base
from eagleapi.db.user_model import User

@pytest.fixture
async def test_db():
    """Create a test database session."""
    # Create an in-memory SQLite database for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create a session factory
    async_session = sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    
    # Create a session
    session = async_session()
    
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()

@pytest.mark.asyncio
async def test_db_connection(test_db):
    """Test that the database connection works."""
    # Simple query to test the connection
    result = await test_db.execute(text("SELECT 1"))
    assert result.scalar() == 1

@pytest.mark.asyncio
async def test_create_user(test_db):
    """Test creating a user in the database."""
    # Create a test user
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password_123"
    )
    
    # Add to database
    test_db.add(user)
    await test_db.commit()
    
    # Retrieve the user using SQLAlchemy ORM instead of raw SQL
    result = await test_db.execute(
        text("SELECT * FROM users WHERE username = :username"),
        {"username": "testuser"}
    )
    db_user = result.first()
    
    # Assertions
    assert db_user is not None
    assert db_user.username == "testuser"
    assert db_user.email == "test@example.com"
