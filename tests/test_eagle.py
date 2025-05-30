"""
Core tests for the Eagle Framework.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from eagleapi import EagleAPI, __version__
from eagleapi.db import Base

@pytest.fixture
def app():
    """Create a test Eagle application."""
    app = EagleAPI(
        title="Eagle Test App",
        description="Test application for Eagle Framework",
        version=__version__,
    )
    return app

@pytest.fixture
def client(app):
    """Create a test client for the Eagle application."""
    return TestClient(app)

def test_app_creation(app):
    """Test that the Eagle application can be created."""
    assert app is not None
    assert app.title == "Eagle Test App"
    assert app.version == __version__

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Welcome to Eagle Framework" in response.json()["message"]

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    # The endpoint exists but might return different content
    assert "status" in response.json()

def test_docs_endpoints(client):
    """Test that API documentation endpoints are available."""
    response = client.get("/docs", allow_redirects=False)
    # Should either return 200 or redirect (308)
    assert response.status_code in [200, 308]
    
    response = client.get("/redoc", allow_redirects=False)
    # Should either return 200 or redirect (308)
    assert response.status_code in [200, 308]
    
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "openapi" in response.json()

@pytest.mark.asyncio
async def test_database_connection():
    """Test database connection and table creation."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import text
    
    # Create an in-memory SQLite database
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Verify tables were created
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result]
        assert any("users" in table.lower() for table in tables)
    
    await engine.dispose()
