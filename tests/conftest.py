"""
Pytest configuration and fixtures for Eagle Framework tests.
"""
import asyncio
import os
import pytest
from pathlib import Path
from typing import AsyncGenerator, Generator, Any, Dict

import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from eagleapi import create_app
from eagleapi.db import Base, get_db
from eagleapi.core.config import settings

# Set test environment variables
os.environ["ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

# Create a test app with overrides
@pytest.fixture(scope="function")
async def app() -> AsyncGenerator[Any, None]:
    """Create a test FastAPI application."""
    # Create test database
    test_app = create_app()
    
    # Create a test database engine
    test_engine = create_async_engine(
        settings.DATABASE_URL,
        echo=True,
        future=True
    )
    
    # Create a test session factory
    TestingSessionLocal = async_sessionmaker(
        bind=test_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )
    
    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    # Override get_db dependency
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with TestingSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    test_app.dependency_overrides[get_db] = override_get_db
    
    # Store the engine in the app state for cleanup
    test_app.state.engine = test_engine
    
    yield test_app
    
    # Clean up
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await test_engine.dispose()

# Test client fixture
@pytest.fixture(scope="function")
def client(app) -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)

# Database session fixture
@pytest_asyncio.fixture(scope="function")
async def db_session(app) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    async with app.state.engine.connect() as conn:
        # Begin a non-ORM transaction
        await conn.begin()
        
        # Bind an individual Session to the connection
        TestingSessionLocal = async_sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=conn,
        )
        
        # Create a new session for the test
        session = TestingSessionLocal()
        
        try:
            yield session
            await session.rollback()
        finally:
            await session.close()
            await conn.rollback()

# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Test user data
@pytest.fixture
def test_user() -> Dict[str, str]:
    """Return test user data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "testpassword123"
    }

# Create a test user in the database
@pytest_asyncio.fixture
async def create_test_user(db_session: AsyncSession, test_user: Dict[str, str]) -> Dict[str, str]:
    """Create a test user in the database."""
    from eagleapi.db.user_model import User
    from eagleapi.core import security
    
    user = User(
        username=test_user["username"],
        email=test_user["email"],
        full_name=test_user["full_name"],
        hashed_password=security.get_password_hash(test_user["password"]),
        is_active=True,
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    # Return the test user data with the ID from the database
    return {
        **test_user,
        "id": user.id,
        "is_active": user.is_active
    }
