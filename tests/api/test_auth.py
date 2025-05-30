"""
API tests for authentication endpoints.
"""
import os
import sys
import logging
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
import pytest
from eagleapi import EagleAPI, Depends, HTTPException, status, APIRouter
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, configure_mappers
from sqlalchemy.sql import text
from pydantic import BaseModel, EmailStr

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Import models to ensure they are registered with SQLAlchemy
from eagleapi.db.base import Base
from eagleapi.db.user_model import User
from eagleapi.db import db

# Create a test FastAPI app
app = EagleAPI()

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_auth.db"

# Create test engine and session factory
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=True,
    future=True
)
TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

# Override the get_db dependency for testing
async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Override get_db dependency for testing."""
    async with TestingSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Override the database dependency in the app
app.dependency_overrides[db.get_session] = override_get_db

# Add a simple test endpoint
@app.get("/test")
async def test_endpoint():
    return {"message": "Test endpoint is working!"}

# Fixture to initialize the test database
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

# Fixture to initialize the test database
@pytest.fixture(scope="session", autouse=True)
async def init_db():
    """Initialize the test database with tables."""
    logger.info("\n" + "="*80)
    logger.info("INITIALIZING TEST DATABASE")
    logger.info("="*80)
    
    # Make sure all models are imported and registered with Base.metadata
    from eagleapi.db.user_model import User  # noqa: F401
    
    # Explicitly configure mappers
    logger.info("Configuring SQLAlchemy mappers...")
    configure_mappers()
    
    # Log all registered tables
    logger.info("\nRegistered tables in metadata:")
    for table_name, table in Base.metadata.tables.items():
        logger.info(f"- {table_name} ({table})")
    
    # Create all tables
    async with test_engine.begin() as conn:
        logger.info("\nDropping all tables...")
        await conn.run_sync(Base.metadata.drop_all)
        
        logger.info("\nCreating all tables...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created successfully")
    
    # Verify tables were created
    async with test_engine.connect() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result]
        logger.info(f"\nTables in database: {tables}")
        
        # Log table schemas for debugging
        for table in tables:
            logger.info(f"\nSchema for table '{table}':")
            result = await conn.execute(text(f"PRAGMA table_info({table})"))
            columns = [dict(row) for row in result.mappings()]
            for col in columns:
                logger.info(f"  - {col['name']}: {col['type']} {'PRIMARY KEY' if col['pk'] else ''}")
    
    # Override the FastAPI app's database dependency
    from eagleapi.main import app
    from eagleapi.db import get_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield  # This is where the testing happens
    
    # Clean up
    logger.info("\n" + "="*80)
    logger.info("CLEANING UP TEST DATABASE")
    logger.info("="*80)
    
    # Clear overrides
    app.dependency_overrides.clear()
    
    # Drop all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    # Dispose the engine
    await test_engine.dispose()
    
    # Remove the test database file
    if os.path.exists("test_auth.db"):
        os.remove("test_auth.db")
    
    logger.info("✅ Test database cleanup complete")

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    # Use the test app we created at the top of the file
    with TestClient(app) as test_client:
        yield test_client


def test_test_endpoint(client: TestClient):
    """Test that the test endpoint works."""
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "Test endpoint is working!"}

class TestAuthAPI:
    """Test cases for authentication endpoints."""

    @pytest.mark.asyncio
    async def test_register_user(self, client: TestClient, test_user: dict):
        """Test user registration."""
        print("\n=== Starting test_register_user ===")
        user_data = {
            "username": test_user["username"],
            "email": test_user["email"],
            "full_name": test_user["full_name"],
            "password": test_user["password"]
        }
        print(f"Sending registration request with data: {user_data}")
        response = client.post("/api/v1/auth/register", json=user_data)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        print(f"Response JSON: {data}")
        
        assert "id" in data
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert data["full_name"] == user_data["full_name"]
        assert data["is_active"] is True
        assert "password" not in data
        assert "hashed_password" not in data
        print("=== test_register_user passed ===\n")

    @pytest.mark.asyncio
    async def test_register_existing_user(self, client: TestClient, test_user: dict):
        """Test registering with an existing username or email."""
        # First register the test user
        await test_register_user(self, client, test_user)
        
        # Try to register with the same email
        user_data = {
            "username": "newuser",
            "email": test_user["email"],
            "full_name": "New User",
            "password": "newpassword123"
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "email already registered" in response.text.lower()
        
        # Try to register with the same username
        user_data = {
            "username": test_user["username"],
            "email": "newemail@example.com",
            "full_name": "New User",
            "password": "newpassword123"
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "username already registered" in response.text.lower()

    @pytest.mark.asyncio
    async def test_login(self, client: TestClient, test_user: dict):
        """Test user login and token generation."""
        # First register the test user
        await test_register_user(self, client, test_user)
        
        # Test login with correct credentials
        response = client.post(
            "/api/v1/auth/token",
            data={
                "username": test_user["username"],
                "password": test_user["password"]
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        
        # Test login with incorrect password
        response = client.post(
            "/api/v1/auth/token",
            data={
                "username": test_user["username"],
                "password": "wrongpassword"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # Test login with non-existent user
        response = client.post(
            "/api/v1/auth/token",
            data={
                "username": "nonexistent",
                "password": "password"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_protected_route(self, client: TestClient, test_user: dict):
        """Test accessing a protected route with and without authentication."""
        # First register the test user
        await test_register_user(self, client, test_user)
        
        # Get a valid token
        token_response = client.post(
            "/api/v1/auth/token",
            data={
                "username": test_user["username"],
                "password": test_user["password"]
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        token = token_response.json()["access_token"]
        
        # Test accessing protected route with token
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]
        
        # Test accessing protected route with invalid token
        headers = {"Authorization": "Bearer invalidtoken"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # Test accessing protected route without token
        response = client.get("/api/v1/auth/me")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

async def test_register_user(test_case, client: TestClient, user_data: dict):
    """Helper function to register a user."""
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert "id" in data
    assert data["username"] == user_data["username"]
    assert data["email"] == user_data["email"]
    assert data["full_name"] == user_data["full_name"]
    assert data["is_active"] is True
    assert "password" not in data
    assert "hashed_password" not in data
    return data
