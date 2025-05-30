"""
Unit tests for Eagle Framework core functionality.
"""
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from eagleapi import Eagle

class TestEagleCore:
    """Test cases for Eagle core functionality."""

    def test_app_creation(self):
        """Test creating a basic Eagle application."""
        app = Eagle()
        assert app is not None
        assert hasattr(app, "router")
        assert len(app.routes) > 0

    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns expected response."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        assert "message" in response.json()

    def test_health_check(self, client: TestClient):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ok"}

    def test_404_response(self, client: TestClient):
        """Test 404 response for non-existent routes."""
        response = client.get("/non-existent-route")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "detail" in response.json()

    def test_docs_endpoints(self, client: TestClient):
        """Test that API documentation endpoints are available."""
        response = client.get("/docs")
        assert response.status_code == status.HTTP_200_OK
        
        response = client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK
        
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        assert "openapi" in response.json()
