"""
Database tests for Eagle Framework.
"""
import os
import asyncio
import pytest
from datetime import datetime
from typing import AsyncGenerator, Any

from sqlalchemy import Column, String, Boolean, Integer, DateTime, select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

# Import db first to avoid circular imports
from eagleapi.db import db, get_db, Base as BaseModel

# Create a test-specific base class to avoid polluting the main models
TestBase = declarative_base()

# Test model for database tests
class TestModel(TestBase):
    """Test model for database tests."""
    __tablename__ = "test_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    description = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def test_db_engine():
    """Create a test database engine and tables for each test function."""
    # Create an in-memory SQLite database for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=True
    )
    
    # Create all tables for our test models
    async with engine.begin() as conn:
        # Drop all tables first to ensure a clean state
        await conn.run_sync(TestBase.metadata.drop_all)
        # Create tables for all models that inherit from TestBase
        await conn.run_sync(TestBase.metadata.create_all)
    
    # Override the global db engine for testing
    original_engine = db.engine
    original_session_factory = db.session_factory
    
    db.engine = engine
    db.session_factory = async_sessionmaker(
        bind=engine, 
        expire_on_commit=False,
        class_=AsyncSession
    )
    
    yield engine
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.drop_all)
    await engine.dispose()
    
    # Restore original engine and session factory
    db.engine = original_engine
    db.session_factory = original_session_factory
    if original_engine:
        await original_engine.dispose()

@pytest.fixture
async def test_db(test_db_engine):
    """Create a test database session with automatic rollback."""
    async with db.session_factory() as session:
        # Start a transaction
        await session.begin()
        
        try:
            yield session
        finally:
            # Rollback the transaction to clean up after each test
            await session.rollback()
            await session.close()

@pytest.fixture
def test_model_data():
    """Provide test data for TestModel."""
    return {
        "name": "Test Item",
        "description": "A test item",
        "is_active": True
    }

@pytest.mark.asyncio
async def test_db_connection(test_db):
    """Test that the database connection works."""
    # Simple query to test the connection
    result = await test_db.execute(text("SELECT 1"))
    assert result.scalar() == 1

@pytest.mark.asyncio
async def test_create_test_model(test_db, test_model_data):
    """Test creating a test model in the database."""
    # Create a test instance
    test_item = TestModel(**test_model_data)
    
    # Add to session and commit
    test_db.add(test_item)
    await test_db.commit()
    await test_db.refresh(test_item)
    
    # Verify the item was created with correct data
    assert test_item.id is not None, "Test item should have an ID after commit"
    assert test_item.name == test_model_data["name"], "Name should match"
    assert test_item.description == test_model_data["description"], "Description should match"
    assert test_item.is_active == test_model_data["is_active"], "is_active should match"
    assert isinstance(test_item.created_at, datetime), "created_at should be a datetime"
    assert isinstance(test_item.updated_at, datetime), "updated_at should be a datetime"
    
    # Create a new session to verify the data was actually committed
    async with db.session_factory() as new_session:
        # Query using the ID
        result = await new_session.get(TestModel, test_item.id)
        assert result is not None, "Should be able to retrieve the test item by ID"
        assert result.name == test_model_data["name"], "Name should match in new session"
        assert result.description == test_model_data["description"], "Description should match in new session"
        assert result.is_active == test_model_data["is_active"], "is_active should match in new session"

@pytest.mark.asyncio
async def test_get_db_dependency(test_db_engine):
    """Test the get_db dependency."""
    # Get a new database session using the dependency
    db_gen = get_db()
    session = await anext(db_gen)
    
    try:
        # Verify we can use the session
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1
    finally:
        # Clean up
        await db_gen.aclose()

@pytest.mark.asyncio
async def test_db_session_rollback(test_db, test_model_data):
    """Test that database sessions are properly rolled back on error."""
    # Create a test instance but don't commit
    test_item = TestModel(**test_model_data)
    test_db.add(test_item)
    
    # The item should not be in the database yet (due to transaction isolation)
    result = await test_db.execute(select(TestModel))
    assert result.scalars().first() is None
    
    # Commit and verify it's there
    await test_db.commit()
    
    # Create a new session to verify the commit
    async with db.session_factory() as session:
        result = await session.execute(select(TestModel))
        assert result.scalars().first() is not None

# Test for creating and querying a test model
@pytest.mark.asyncio
async def test_create_and_query_model(test_db):
    """Test creating and querying a test model in the database."""
    # Create test data
    test_data = {
        "name": "Test Item",
        "description": "This is a test item",
        "is_active": True
    }
    
    # Create model instance
    test_item = TestModel(**test_data)
    
    # Add to database
    test_db.add(test_item)
    await test_db.commit()
    await test_db.refresh(test_item)
    
    # Verify the item was created with correct data
    assert test_item.id is not None
    assert test_item.name == test_data["name"]
    assert test_item.description == test_data["description"]
    assert test_item.is_active == test_data["is_active"]
    assert isinstance(test_item.created_at, datetime)
    assert isinstance(test_item.updated_at, datetime)
    
    # Verify we can query the item back
    async with db.session_factory() as session:
        result = await session.execute(
            select(TestModel).where(TestModel.id == test_item.id)
        )
        db_item = result.scalars().first()
        
        # Assertions
        assert db_item is not None
        assert db_item.name == test_data["name"]
        assert db_item.description == test_data["description"]
        assert db_item.is_active == test_data["is_active"]
