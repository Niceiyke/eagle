"""
Simple authentication tests for Eagle Framework.
"""
import pytest
from eagleapi import EagleAPI, Depends, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import Optional

# Create a test FastAPI app
app = EagleAPI()

# Mock user database
fake_users_db = {}

# Models
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

class UserCreate(User):
    password: str

# Helper functions
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def fake_hash_password(password: str):
    return "hashed_" + password

def fake_verify_password(plain_password: str, hashed_password: str):
    return fake_hash_password(plain_password) == hashed_password

# Routes
@app.post("/register", response_model=User)
async def register(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = fake_hash_password(user.password)
    user_dict = user.dict()
    del user_dict["password"]
    user_dict["hashed_password"] = hashed_password
    fake_users_db[user.username] = user_dict
    return user_dict

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_user)):
    return current_user

# Test client fixture
@pytest.fixture
def client():
    return TestClient(app)

def test_register_user(client):
    """Test user registration."""
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "testpass123"
    }
    
    # Test successful registration
    response = client.post("/register", json=user_data)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == user_data["username"]
    assert "hashed_password" not in data
    
    # Test duplicate username
    response = client.post("/register", json=user_data)
    assert response.status_code == 400

def test_protected_route(client):
    """Test accessing a protected route."""
    # First register a user
    user_data = {
        "username": "testuser2",
        "password": "testpass123"
    }
    client.post("/register", json=user_data)
    
    # Test accessing protected route (should fail without auth)
    response = client.get("/users/me")
    assert response.status_code == 422  # Missing auth
    
    # Note: In a real test, you would test with proper authentication
    # This is a simplified example without actual auth for testing purposes
