"""Integration tests for the complete agent flow."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.agent import AgentOrchestrator, AgentResult
from src.api.schemas import AgentRunStatus
from src.llm.base import LLMResponse, Message
from src.storage.database import Database, reset_database
from src.storage.models import ExecutionStatus, RequestStatus
from src.storage.repository import AuditRepository
from src.tools.builtin import register_builtin_tools
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry, reset_registry


@pytest.fixture(autouse=True)
def set_env_vars():
    """Set required environment variables for tests."""
    os.environ["AGENT_API_KEY"] = "test-api-key"
    os.environ["LLM_API_KEY"] = "test-llm-key"
    yield
    if "AGENT_API_KEY" in os.environ:
        del os.environ["AGENT_API_KEY"]
    if "LLM_API_KEY" in os.environ:
        del os.environ["LLM_API_KEY"]


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create test files
        (workspace / "readme.txt").write_text("Welcome to the workspace")
        (workspace / "config.json").write_text('{"key": "value"}')
        (workspace / "data").mkdir()
        (workspace / "data" / "sample.txt").write_text("Sample data file")

        yield workspace


@pytest.fixture
def temp_db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        db.initialize()
        yield db


@pytest.fixture
def orchestrator(temp_workspace, temp_db):
    """Create a fully configured orchestrator for testing."""
    from unittest.mock import MagicMock

    # Create components
    registry = ToolRegistry()
    repository = AuditRepository(db=temp_db)

    settings = MagicMock()
    settings.max_tool_timeout = 30
    settings.max_output_size = 10000
    settings.workspace_root = temp_workspace

    executor = ToolExecutor(registry=registry, repository=repository, settings=settings)
    register_builtin_tools(registry, executor)

    # Mock LLM
    mock_llm = AsyncMock()

    return AgentOrchestrator(
        llm=mock_llm,
        registry=registry,
        executor=executor,
        repository=repository,
    )


class TestE2EAgentFlow:
    """End-to-end tests for agent flow."""

    async def test_complete_flow_with_tool(self, orchestrator, temp_workspace):
        """Test: instruction → reasoning → tool execution → response."""
        # Setup: LLM first requests list_dir, then gives final answer
        orchestrator.llm.complete.side_effect = [
            LLMResponse.tool(name="list_dir", arguments={"path": "."}),
            LLMResponse.text("I found 3 items: readme.txt, config.json, and data folder."),
        ]

        # Create request
        request = orchestrator.repository.create_request("What files are in the workspace?")

        # Run agent
        result = await orchestrator.run(
            input_text="What files are in the workspace?",
            request_id=request.id,
        )

        # Verify result
        assert result.status == AgentRunStatus.SUCCESS
        assert "found" in result.output.lower() or "items" in result.output.lower()
        assert len(result.tool_executions) == 1
        assert result.tool_executions[0].tool_name == "list_dir"
        assert result.tool_executions[0].status == "success"

    async def test_flow_with_multiple_tools(self, orchestrator, temp_workspace):
        """Test flow with multiple sequential tool calls."""
        orchestrator.llm.complete.side_effect = [
            LLMResponse.tool(name="list_dir", arguments={"path": "."}),
            LLMResponse.tool(name="read_file", arguments={"path": "readme.txt"}),
            LLMResponse.text("The readme says: Welcome to the workspace"),
        ]

        request = orchestrator.repository.create_request("Read the readme file")
        result = await orchestrator.run("Read the readme file", request.id)

        assert result.status == AgentRunStatus.SUCCESS
        assert len(result.tool_executions) == 2

    async def test_flow_health_check_only(self, orchestrator):
        """Test flow with just health check."""
        orchestrator.llm.complete.side_effect = [
            LLMResponse.tool(name="health_check", arguments={}),
            LLMResponse.text("System is healthy and running."),
        ]

        request = orchestrator.repository.create_request("Check system status")
        result = await orchestrator.run("Check system status", request.id)

        assert result.status == AgentRunStatus.SUCCESS
        assert len(result.tool_executions) == 1
        assert result.tool_executions[0].tool_name == "health_check"

    async def test_flow_no_tool_needed(self, orchestrator):
        """Test flow where LLM responds directly without tools."""
        orchestrator.llm.complete.return_value = LLMResponse.text(
            "I'm an AI assistant. How can I help you?"
        )

        request = orchestrator.repository.create_request("Hello")
        result = await orchestrator.run("Hello", request.id)

        assert result.status == AgentRunStatus.SUCCESS
        assert result.output == "I'm an AI assistant. How can I help you?"
        assert len(result.tool_executions) == 0


class TestSecurityConstraints:
    """Tests for security constraints."""

    async def test_path_traversal_blocked(self, orchestrator, temp_workspace):
        """Test that path traversal attempts are blocked."""
        orchestrator.llm.complete.side_effect = [
            LLMResponse.tool(name="read_file", arguments={"path": "../../../etc/passwd"}),
            LLMResponse.text("I couldn't access that file due to security restrictions."),
        ]

        request = orchestrator.repository.create_request("Read /etc/passwd")
        result = await orchestrator.run("Read /etc/passwd", request.id)

        # Tool should have failed with path traversal error
        assert len(result.tool_executions) == 1
        assert result.tool_executions[0].status in ("validation_error", "execution_error")
        assert "escapes workspace" in result.tool_executions[0].error

    async def test_invalid_tool_name(self, orchestrator):
        """Test handling of non-existent tool request."""
        orchestrator.llm.complete.side_effect = [
            LLMResponse.tool(name="nonexistent_tool", arguments={}),
            LLMResponse.text("That tool doesn't exist."),
        ]

        request = orchestrator.repository.create_request("Use fake tool")
        result = await orchestrator.run("Use fake tool", request.id)

        assert len(result.tool_executions) == 1
        assert "not found" in result.tool_executions[0].error

    async def test_invalid_tool_input(self, orchestrator):
        """Test handling of invalid tool input."""
        orchestrator.llm.complete.side_effect = [
            LLMResponse.tool(name="read_file", arguments={}),  # Missing required 'path'
            LLMResponse.text("I need to specify a file path."),
        ]

        request = orchestrator.repository.create_request("Read a file")
        result = await orchestrator.run("Read a file", request.id)

        assert len(result.tool_executions) == 1
        assert result.tool_executions[0].status == "validation_error"

    async def test_max_iterations_limit(self, orchestrator):
        """Test that agent stops after max iterations."""
        # Return tool calls forever
        orchestrator.llm.complete.return_value = LLMResponse.tool(
            name="health_check", arguments={}
        )

        request = orchestrator.repository.create_request("Keep checking")
        result = await orchestrator.run("Keep checking", request.id)

        assert result.status == AgentRunStatus.ERROR
        assert "Maximum" in result.error or "iterations" in result.error
        # Should have exactly MAX_ITERATIONS tool calls
        assert len(result.tool_executions) == 5


class TestAuditLogging:
    """Tests for audit logging completeness."""

    async def test_request_logged(self, orchestrator):
        """Test that requests are logged."""
        orchestrator.llm.complete.return_value = LLMResponse.text("Response")

        request = orchestrator.repository.create_request("Test input")
        await orchestrator.run("Test input", request.id)

        # Verify request was updated
        saved_request = orchestrator.repository.get_request(request.id)
        assert saved_request is not None
        assert saved_request.status == RequestStatus.COMPLETED
        assert saved_request.final_output == "Response"

    async def test_decision_logged(self, orchestrator):
        """Test that decisions are logged."""
        orchestrator.llm.complete.side_effect = [
            LLMResponse.tool(name="health_check", arguments={}),
            LLMResponse.text("Done"),
        ]

        request = orchestrator.repository.create_request("Check health")
        await orchestrator.run("Check health", request.id)

        decisions = orchestrator.repository.get_decisions_for_request(request.id)
        assert len(decisions) == 1
        assert decisions[0].selected_tool == "health_check"

    async def test_execution_logged(self, orchestrator):
        """Test that tool executions are logged."""
        orchestrator.llm.complete.side_effect = [
            LLMResponse.tool(name="health_check", arguments={}),
            LLMResponse.text("Done"),
        ]

        request = orchestrator.repository.create_request("Check health")
        await orchestrator.run("Check health", request.id)

        executions = orchestrator.repository.get_executions_for_request(request.id)
        assert len(executions) == 1
        assert executions[0].tool_name == "health_check"
        assert executions[0].status == ExecutionStatus.SUCCESS

    async def test_error_logged(self, orchestrator):
        """Test that errors are logged."""
        orchestrator.llm.complete.return_value = LLMResponse.error("LLM failure")

        request = orchestrator.repository.create_request("Fail")
        await orchestrator.run("Fail", request.id)

        saved_request = orchestrator.repository.get_request(request.id)
        assert saved_request.status == RequestStatus.FAILED
        assert "LLM failure" in saved_request.error


class TestAPIIntegration:
    """Integration tests through the HTTP API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        reset_database()
        reset_registry()
        from src.main import app
        yield TestClient(app)
        reset_database()
        reset_registry()

    def test_full_api_flow(self, client):
        """Test complete flow through HTTP API."""
        with patch("src.api.routes.AnthropicConnector") as MockConnector:
            mock_llm = AsyncMock()
            mock_llm.complete.side_effect = [
                LLMResponse.tool(name="health_check", arguments={}),
                LLMResponse.text("System health verified."),
            ]
            MockConnector.return_value = mock_llm

            response = client.post(
                "/agent/run",
                json={"input": "Verify system health"},
                headers={"X-API-Key": "test-api-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["tool_executions"]) == 1
        assert data["request_id"] is not None

    def test_api_error_handling(self, client):
        """Test API error responses."""
        with patch("src.api.routes.AnthropicConnector") as MockConnector:
            mock_llm = AsyncMock()
            mock_llm.complete.side_effect = Exception("Unexpected error")
            MockConnector.return_value = mock_llm

            response = client.post(
                "/agent/run",
                json={"input": "Cause error"},
                headers={"X-API-Key": "test-api-key"},
            )

        # The API returns 200 with error status in body, not 500
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "error" in data["error"].lower() or "unexpected" in data["error"].lower()
