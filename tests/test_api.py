"""Tests for API endpoints."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.llm.base import LLMResponse


@pytest.fixture(autouse=True)
def set_env_vars():
    """Set required environment variables for tests."""
    os.environ["AGENT_API_KEY"] = "test-api-key"
    os.environ["LLM_API_KEY"] = "test-llm-key"
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    yield
    # Cleanup
    for key in ["AGENT_API_KEY", "LLM_API_KEY", "RATE_LIMIT_ENABLED"]:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture
async def configured_app():
    """Create a fully configured app with mocked LLM."""
    from src.storage.database import reset_database
    from src.tools.registry import reset_registry

    reset_database()
    reset_registry()

    # Import after resetting
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from src.api.agent import AgentOrchestrator
    from src.api.routes import router
    from src.core.config import get_settings
    from src.storage.database import AsyncDatabase
    from src.storage.repository import AuditRepository
    from src.tools.builtin import register_builtin_tools
    from src.tools.executor import ToolExecutor
    from src.tools.registry import ToolRegistry

    # Create temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = get_settings()

        app = FastAPI()

        # Initialize components
        db = AsyncDatabase(Path(tmpdir) / "test.db")
        await db.initialize()

        registry = ToolRegistry()
        repository = AuditRepository(db=db)

        # Mock settings with permissive permissions for tests
        mock_settings = MagicMock()
        mock_settings.max_tool_timeout = 30
        mock_settings.max_output_size = 10000
        mock_settings.workspace_root = Path(tmpdir) / "workspace"
        mock_settings.workspace_root.mkdir(parents=True, exist_ok=True)
        mock_settings.blocked_permissions = set()

        executor = ToolExecutor(registry=registry, repository=repository, settings=mock_settings)
        register_builtin_tools(registry, executor)

        # Create mock LLM
        mock_llm = AsyncMock()

        orchestrator = AgentOrchestrator(
            llm=mock_llm,
            registry=registry,
            executor=executor,
            repository=repository,
        )

        # Set up app state
        app.state.registry = registry
        app.state.executor = executor
        app.state.repository = repository
        app.state.llm = mock_llm
        app.state.orchestrator = orchestrator
        app.state.settings = settings

        # Add CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

        app.include_router(router)

        yield app

    reset_database()
    reset_registry()


@pytest.fixture
async def client(configured_app):
    """Create async test client."""
    transport = ASGITransport(app=configured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Tests for GET /health."""

    async def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"

    async def test_health_no_auth_required(self, client):
        """Test health check doesn't require authentication."""
        response = await client.get("/health")
        assert response.status_code == 200


class TestAgentRunEndpoint:
    """Tests for POST /agent/run."""

    async def test_missing_api_key(self, client):
        """Test that missing API key returns 401."""
        response = await client.post("/agent/run", json={"input": "test"})

        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    async def test_invalid_api_key(self, client):
        """Test that invalid API key returns 401."""
        response = await client.post(
            "/agent/run",
            json={"input": "test"},
            headers={"X-API-Key": "wrong-key"},
        )

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    async def test_empty_input(self, client):
        """Test that empty input returns 422."""
        response = await client.post(
            "/agent/run",
            json={"input": ""},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 422

    async def test_input_too_long(self, client):
        """Test that too long input returns 422."""
        response = await client.post(
            "/agent/run",
            json={"input": "x" * 10001},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 422

    async def test_successful_run(self, configured_app, client):
        """Test successful agent run with mocked LLM."""
        mock_response = LLMResponse.text("Here is the result")
        configured_app.state.llm.complete.return_value = mock_response

        response = await client.post(
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

    async def test_run_with_tool_call(self, configured_app, client):
        """Test agent run that uses a tool."""
        # First response requests tool, second gives final answer
        mock_tool_response = LLMResponse.tool(
            name="health_check",
            arguments={},
            tool_use_id="test_tool_use_123",
        )
        mock_final_response = LLMResponse.text("The system is healthy")

        configured_app.state.llm.complete.side_effect = [
            mock_tool_response, mock_final_response
        ]

        response = await client.post(
            "/agent/run",
            json={"input": "Check system health"},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["tool_executions"]) >= 1
        assert data["tool_executions"][0]["tool_name"] == "health_check"

    async def test_run_with_llm_error(self, configured_app, client):
        """Test agent run when LLM returns error."""
        mock_response = LLMResponse.from_error("LLM service unavailable")
        configured_app.state.llm.complete.return_value = mock_response

        response = await client.post(
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

    def test_constant_time_comparison(self):
        """Test that API key comparison is constant-time."""
        # This is a basic test - in production you'd want timing analysis
        # For now, we just verify the code path uses secrets.compare_digest
        import inspect

        from src.api.auth import verify_api_key
        source = inspect.getsource(verify_api_key)
        assert "compare_digest" in source
