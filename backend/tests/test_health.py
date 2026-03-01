import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import patch, MagicMock

from app.main import app
from app.core.database import get_db


@pytest.fixture
def client(db_session):
    """
    Provides a TestClient that uses the test database session.
    """
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestHealthCheck:
    """Test suite for health check endpoints."""
    
    def test_health_check_success_with_database_connection(self, client):
        """
        Test that the health check endpoint returns 200 status with healthy status
        when the database connection is successful.
        
        This test verifies:
        - Response status code is 200 OK
        - Response contains 'healthy' status
        - Database connection is reported as 'connected'
        - Environment information is included
        """
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert "environment" in data
    
    def test_health_check_database_connection_failure(self, client, db_session):
        """
        Test that the health check endpoint returns 503 status when database
        connection fails.
        
        This test verifies:
        - Response status code is 503 Service Unavailable
        - Appropriate error detail is returned
        - The error message mentions database connection failure
        """
        # Mock the database execute method to raise an exception
        with patch.object(db_session, 'execute', side_effect=Exception("Connection refused")):
            response = client.get("/api/v1/health")
            
            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
            assert "Database connection failed" in data["detail"]
    
    def test_health_check_returns_json_format(self, client):
        """
        Test that the health check endpoint returns valid JSON with expected fields.
        
        This test verifies:
        - Response content-type is application/json
        - Response contains all required fields (status, database, environment)
        - All fields have appropriate data types
        """
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data.get("status"), str)
        assert isinstance(data.get("database"), str)
        assert isinstance(data.get("environment"), str)
        assert len(data) == 3  # Exactly three fields
