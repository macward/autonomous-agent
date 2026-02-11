"""Tests for tool executor (async version)."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.storage.database import AsyncDatabase
from src.storage.repository import AuditRepository
from src.tools.base import Tool, ToolDefinition, ToolExecutionError, ToolPermission
from src.tools.executor import ExecutionResultStatus, ToolExecutor
from src.tools.registry import ToolRegistry


class SuccessTool(Tool):
    """Tool that succeeds."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="success_tool",
            description="A tool that succeeds",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"result": kwargs["value"]}


class SlowTool(Tool):
    """Tool that takes time."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="slow_tool",
            description="A slow tool",
            input_schema={"type": "object", "properties": {}},
            timeout_seconds=1,
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(5)
        return {"result": "done"}


class FailingTool(Tool):
    """Tool that fails."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="failing_tool",
            description="A tool that fails",
            input_schema={"type": "object", "properties": {}},
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise ToolExecutionError("Intentional failure", "failing_tool")


class LargeOutputTool(Tool):
    """Tool that returns large output."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="large_output_tool",
            description="A tool with large output",
            input_schema={"type": "object", "properties": {}},
            max_output_size=100,
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"result": "x" * 1000}


class NetworkTool(Tool):
    """Tool that requires network permission."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="network_tool",
            description="A tool that needs network",
            input_schema={"type": "object", "properties": {}},
            permissions=[ToolPermission.NETWORK],
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"result": "network access"}


class ExecuteTool(Tool):
    """Tool that requires execute permission."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="execute_tool",
            description="A tool that executes commands",
            input_schema={"type": "object", "properties": {}},
            permissions=[ToolPermission.EXECUTE],
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"result": "command executed"}


@pytest.fixture
def registry():
    """Create registry with test tools."""
    reg = ToolRegistry()
    reg.register(SuccessTool())
    reg.register(SlowTool())
    reg.register(FailingTool())
    reg.register(LargeOutputTool())
    reg.register(NetworkTool())
    reg.register(ExecuteTool())
    return reg


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.max_tool_timeout = 30
    settings.max_output_size = 10000
    settings.workspace_root = Path(tempfile.mkdtemp())
    # Default: block NETWORK and EXECUTE permissions
    settings.blocked_permissions = {ToolPermission.NETWORK, ToolPermission.EXECUTE}
    return settings


@pytest.fixture
def mock_settings_permissive():
    """Create mock settings with no blocked permissions."""
    settings = MagicMock()
    settings.max_tool_timeout = 30
    settings.max_output_size = 10000
    settings.workspace_root = Path(tempfile.mkdtemp())
    settings.blocked_permissions = set()
    return settings


@pytest.fixture
def executor(registry, mock_settings):
    """Create executor for testing."""
    return ToolExecutor(registry=registry, settings=mock_settings)


@pytest.fixture
def executor_permissive(registry, mock_settings_permissive):
    """Create executor with no permission restrictions."""
    return ToolExecutor(registry=registry, settings=mock_settings_permissive)


@pytest.fixture
async def executor_with_repo(registry, mock_settings):
    """Create executor with audit repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = AsyncDatabase(Path(tmpdir) / "test.db")
        await db.initialize()
        repo = AuditRepository(db=db)
        yield ToolExecutor(registry=registry, repository=repo, settings=mock_settings)


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    async def test_execute_success(self, executor):
        """Test successful execution."""
        result = await executor.execute("success_tool", {"value": "hello"})

        assert result.status == ExecutionResultStatus.SUCCESS
        assert result.output == {"result": "hello"}
        assert result.error is None
        assert result.duration_ms >= 0  # May be 0 for very fast executions

    async def test_execute_tool_not_found(self, executor):
        """Test execution of non-existent tool."""
        result = await executor.execute("nonexistent", {})

        assert result.status == ExecutionResultStatus.VALIDATION_ERROR
        assert "not found" in result.error

    async def test_execute_validation_error(self, executor):
        """Test execution with invalid input."""
        result = await executor.execute("success_tool", {})  # Missing required field

        assert result.status == ExecutionResultStatus.VALIDATION_ERROR
        assert "Validation failed" in result.error

    async def test_execute_timeout(self, executor):
        """Test execution timeout."""
        result = await executor.execute("slow_tool", {})

        assert result.status == ExecutionResultStatus.TIMEOUT
        assert "timed out" in result.error

    async def test_execute_tool_error(self, executor):
        """Test execution with tool error."""
        result = await executor.execute("failing_tool", {})

        assert result.status == ExecutionResultStatus.EXECUTION_ERROR
        assert "Intentional failure" in result.error

    async def test_execute_output_truncation(self, executor):
        """Test output truncation for large outputs."""
        result = await executor.execute("large_output_tool", {})

        assert result.status == ExecutionResultStatus.OUTPUT_TRUNCATED
        assert result.truncated is True
        assert result.output["_truncated"] is True

    async def test_execute_with_audit(self, executor_with_repo):
        """Test execution logs to audit repository."""
        # Create a request and decision first (required for foreign key)
        request = await executor_with_repo.repository.create_request("test input")
        decision = await executor_with_repo.repository.create_decision(
            request_id=request.id,
            reasoning="Test reasoning",
            selected_tool="success_tool",
        )

        result = await executor_with_repo.execute(
            "success_tool",
            {"value": "test"},
            request_id=request.id,
            decision_id=decision.id,
        )

        assert result.status == ExecutionResultStatus.SUCCESS

        # Verify execution was logged
        executions = await executor_with_repo.repository.get_executions_for_request(request.id)
        assert len(executions) == 1
        assert executions[0].tool_name == "success_tool"

    def test_result_to_dict(self, executor):
        """Test ExecutionResult serialization."""
        from src.tools.executor import ExecutionResult

        result = ExecutionResult(
            status=ExecutionResultStatus.SUCCESS,
            output={"key": "value"},
            duration_ms=100,
        )

        d = result.to_dict()
        assert d["status"] == "success"
        assert d["output"] == {"key": "value"}
        assert d["duration_ms"] == 100


class TestPermissionEnforcement:
    """Tests for tool permission enforcement."""

    async def test_network_permission_blocked(self, executor):
        """Test that NETWORK permission is blocked by default."""
        result = await executor.execute("network_tool", {})

        assert result.status == ExecutionResultStatus.PERMISSION_DENIED
        assert "network" in result.error.lower()

    async def test_execute_permission_blocked(self, executor):
        """Test that EXECUTE permission is blocked by default."""
        result = await executor.execute("execute_tool", {})

        assert result.status == ExecutionResultStatus.PERMISSION_DENIED
        assert "execute" in result.error.lower()

    async def test_network_permission_allowed_when_not_blocked(self, executor_permissive):
        """Test that NETWORK works when not blocked."""
        result = await executor_permissive.execute("network_tool", {})

        assert result.status == ExecutionResultStatus.SUCCESS
        assert result.output == {"result": "network access"}

    async def test_execute_permission_allowed_when_not_blocked(self, executor_permissive):
        """Test that EXECUTE works when not blocked."""
        result = await executor_permissive.execute("execute_tool", {})

        assert result.status == ExecutionResultStatus.SUCCESS
        assert result.output == {"result": "command executed"}


class TestPathValidation:
    """Tests for workspace path validation."""

    def test_valid_path(self, executor):
        """Test valid path within workspace."""
        path = executor.validate_path("subdir/file.txt")
        assert str(executor.workspace_root) in str(path)

    def test_path_traversal_blocked(self, executor):
        """Test path traversal is blocked."""
        with pytest.raises(ToolExecutionError) as exc_info:
            executor.validate_path("../../../etc/passwd")

        assert "escapes workspace" in exc_info.value.message

    def test_absolute_path_outside_workspace(self, executor):
        """Test absolute path outside workspace is blocked."""
        with pytest.raises(ToolExecutionError):
            executor.validate_path("/etc/passwd")
