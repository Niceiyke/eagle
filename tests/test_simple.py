"""
Simple test file to verify basic Eagle Framework functionality.
"""
from fastapi.testclient import TestClient
from eagleapi import EagleAPI

app = EagleAPI()

def test_root_endpoint():
    """Test the root endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Eagle" in response.json()["message"]

def test_health_check():
    """Test the health check endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_docs_endpoints():
    """Test that API documentation endpoints are available."""
    client = TestClient(app)
    
    # Test Swagger UI
    response = client.get("/docs")
    assert response.status_code == 200
    
    # Test ReDoc
    response = client.get("/redoc")
    assert response.status_code == 200
    
    # Test OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "openapi" in response.json()

def test_custom_route():
    """Test adding and using a custom route."""
    # Create a new app for this test
    test_app = Eagle()
    
    @test_app.get("/custom")
    def custom_route():
        return {"custom": "success"}
    
    client = TestClient(test_app)
    response = client.get("/custom")
    assert response.status_code == 200
    assert response.json() == {"custom": "success"}
