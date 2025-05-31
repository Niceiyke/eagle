"""
Tests for database setup and initialization in the Eagle framework.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from eagleapi import Depends
from eagleapi.db import Base, db, get_db
from eagleapi.core.config import settings

# Use the existing test database URL from settings
TEST_DATABASE_URL = settings.DATABASE_URL


@pytest.mark.asyncio
async def test_database_initialization(app):
    """Test that the database is properly initialized when the app starts."""
    # The app fixture already triggers app startup
    
    # Verify the database engine is available
    assert hasattr(db, 'engine'), "Database engine should be initialized"
    assert db.engine is not None, "Database engine should not be None"
    
    # Verify we can connect to the database
    async with db.engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1, "Should be able to execute a simple query"

@pytest.mark.asyncio
async def test_tables_created(app, db_session: AsyncSession):
    """Test that all expected tables are created in the database."""
    # Get the list of tables that should exist
    expected_tables = set(Base.metadata.tables.keys())

    # Get the actual tables in the database
    result = await db_session.execute(
        text("SELECT name FROM sqlite_master WHERE type='table'")
    )
    actual_tables = {row[0] for row in result}
    
    # Skip internal SQLite tables
    actual_tables = {t for t in actual_tables if not t.startswith('sqlite_')}
    
    # Verify all expected tables exist
    assert expected_tables.issubset(actual_tables), \
        f"Expected tables {expected_tables} not found in database. Found: {actual_tables}"

@pytest.mark.asyncio
async def test_health_check(app, client: TestClient, db_session):
    """Test the health check endpoint reports database status."""
    # The db_session fixture ensures the database is initialized
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # The database should be connected since we're using the db_session fixture
    assert data["database"] == "connected"

@pytest.mark.asyncio
async def test_database_session_in_request(app, client: TestClient):
    """Test that database sessions work within request handlers."""
    # Add a test endpoint
    @app.get("/test-session")
    async def test_session(session: AsyncSession = Depends(get_db)):
        # Test that we can get a session and execute a query
        result = await session.execute(text("SELECT 1"))
        return {"result": result.scalar()}
    
    # Test the endpoint
    response = client.get("/test-session")
    assert response.status_code == 200
    assert response.json()["result"] == 1
