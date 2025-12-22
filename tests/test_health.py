"""
Tests for health check endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test suite for health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test basic health check returns 200 and correct structure."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data

    def test_detailed_health_check(self, client: TestClient) -> None:
        """Test detailed health check returns component information."""
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "components" in data
        assert "database" in data["components"]
        assert "azure_openai" in data["components"]
        assert "langfuse" in data["components"]
        assert "gcp" in data["components"]

    def test_readiness_check(self, client: TestClient) -> None:
        """Test readiness probe returns ready status."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ready" in data
        assert "checks" in data
        assert isinstance(data["ready"], bool)

    def test_liveness_check(self, client: TestClient) -> None:
        """Test liveness probe returns alive status."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"


class TestRootEndpoint:
    """Test suite for root endpoint."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "disclaimer" in data
        assert "health" in data


