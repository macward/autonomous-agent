"""Tests for API endpoints."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.schemas import AgentRunStatus
from src.llm.base import LLMResponse, ResponseType


@pytest.fixture(autouse=True)
def set_env_vars():
    """Set required environment variables for tests."""
    os.environ["AGENT_API_KEY"] = "test-api-key"
    os.environ["LLM_API_KEY"] = "test-llm-key"
    yield
    # Cleanup
    if "AGENT_API_KEY" in os.environ:
        del os.environ["AGENT_API_KEY"]
    if "LLM_API_KEY" in os.environ:
        del os.environ["LLM_API_KEY"]


@pytest.fixture
def app():
    """Create test app."""
    # Reset database and registry for each test
    from src.storage.database import reset_database
    from src.tools.registry import reset_registry
    reset_database()
    reset_registry()

    from src.main import app
    yield app

    reset_database()
    reset_registry()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"

    def test_health_no_auth_required(self, client):
        """Test health check doesn't require authentication."""
        response = client.get("/health")
        assert response.status_code == 200


class TestAgentRunEndpoint:
    """Tests for POST /agent/run."""

    def test_missing_api_key(self, client):
        """Test that missing API key returns 401."""
        response = client.post("/agent/run", json={"input": "test"})

        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_invalid_api_key(self, client):
        """Test that invalid API key returns 401."""
        response = client.post(
            "/agent/run",
            json={"input": "test"},
            headers={"X-API-Key": "wrong-key"},
        )

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_empty_input(self, client):
        """Test that empty input returns 422."""
        response = client.post(
            "/agent/run",
            json={"input": ""},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 422

    def test_input_too_long(self, client):
        """Test that too long input returns 422."""
        response = client.post(
            "/agent/run",
            json={"input": "x" * 10001},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 422

    def test_successful_run(self, client):
        """Test successful agent run with mocked LLM."""
        mock_response = LLMResponse.text("Here is the result")

        with patch("src.api.routes.AnthropicConnector") as MockConnector:
            mock_llm = AsyncMock()
            mock_llm.complete.return_value = mock_response
            MockConnector.return_value = mock_llm

            response = client.post(
                "/agent/run",
                json={"input": "Hello agent"},
                headers={"X-API-Key": "test-api-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["output"] == "Here is the result"
        assert "request_id" in data
        assert "duration_ms" in data

    def test_run_with_tool_call(self, client):
        """Test agent run that uses a tool."""
        # First response requests tool, second gives final answer
        mock_tool_response = LLMResponse.tool(
            name="health_check",
            arguments={},
        )
        mock_final_response = LLMResponse.text("The system is healthy")

        with patch("src.api.routes.AnthropicConnector") as MockConnector:
            mock_llm = AsyncMock()
            mock_llm.complete.side_effect = [mock_tool_response, mock_final_response]
            MockConnector.return_value = mock_llm

            response = client.post(
                "/agent/run",
                json={"input": "Check system health"},
                headers={"X-API-Key": "test-api-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["tool_executions"]) >= 1
        assert data["tool_executions"][0]["tool_name"] == "health_check"

    def test_run_with_llm_error(self, client):
        """Test agent run when LLM returns error."""
        mock_response = LLMResponse.error("LLM service unavailable")

        with patch("src.api.routes.AnthropicConnector") as MockConnector:
            mock_llm = AsyncMock()
            mock_llm.complete.return_value = mock_response
            MockConnector.return_value = mock_llm

            response = client.post(
                "/agent/run",
                json={"input": "Test request"},
                headers={"X-API-Key": "test-api-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "LLM service unavailable" in data["error"]


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_constant_time_comparison(self, client):
        """Test that API key comparison is constant-time."""
        import secrets
        from src.api.auth import verify_api_key

        # This is a basic test - in production you'd want timing analysis
        # For now, we just verify the code path uses secrets.compare_digest
        import inspect
        source = inspect.getsource(verify_api_key)
        assert "compare_digest" in source
