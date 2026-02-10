"""Tests for storage module."""

import tempfile
from pathlib import Path

import pytest

from src.storage.database import Database
from src.storage.models import ExecutionStatus, RequestStatus
from src.storage.repository import AuditRepository


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        db.initialize()
        yield db


@pytest.fixture
def repo(temp_db):
    """Create a repository with temp database."""
    return AuditRepository(db=temp_db)


class TestDatabase:
    """Tests for Database class."""

    def test_initialize_creates_tables(self, temp_db):
        """Test that initialization creates all tables."""
        with temp_db.connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t["name"] for t in tables]

        assert "requests" in table_names
        assert "decisions" in table_names
        assert "executions" in table_names
        assert "audit_logs" in table_names


class TestAuditRepository:
    """Tests for AuditRepository class."""

    def test_create_request(self, repo):
        """Test creating a request."""
        request = repo.create_request("Test input")

        assert request.id is not None
        assert request.input_text == "Test input"
        assert request.status == RequestStatus.PENDING

    def test_update_request_status(self, repo):
        """Test updating request status."""
        request = repo.create_request("Test input")
        repo.update_request_status(
            request.id,
            RequestStatus.COMPLETED,
            final_output="Done",
        )

        updated = repo.get_request(request.id)
        assert updated.status == RequestStatus.COMPLETED
        assert updated.final_output == "Done"
        assert updated.completed_at is not None

    def test_create_decision(self, repo):
        """Test creating a decision."""
        request = repo.create_request("Test input")
        decision = repo.create_decision(
            request_id=request.id,
            reasoning="I should use the health_check tool",
            selected_tool="health_check",
            tool_input={"param": "value"},
        )

        assert decision.id is not None
        assert decision.request_id == request.id
        assert decision.selected_tool == "health_check"
        assert decision.tool_input == {"param": "value"}

    def test_get_decisions_for_request(self, repo):
        """Test getting decisions for a request."""
        request = repo.create_request("Test input")
        repo.create_decision(request.id, "First reasoning", "tool1")
        repo.create_decision(request.id, "Second reasoning", "tool2")

        decisions = repo.get_decisions_for_request(request.id)
        assert len(decisions) == 2
        assert decisions[0].reasoning == "First reasoning"
        assert decisions[1].reasoning == "Second reasoning"

    def test_create_execution(self, repo):
        """Test creating an execution."""
        request = repo.create_request("Test input")
        decision = repo.create_decision(request.id, "Test reasoning", "test_tool")

        execution = repo.create_execution(
            decision_id=decision.id,
            request_id=request.id,
            tool_name="test_tool",
            tool_input={"key": "value"},
            status=ExecutionStatus.SUCCESS,
            output="Result",
            duration_ms=100,
        )

        assert execution.id is not None
        assert execution.tool_name == "test_tool"
        assert execution.status == ExecutionStatus.SUCCESS
        assert execution.duration_ms == 100

    def test_get_executions_for_request(self, repo):
        """Test getting executions for a request."""
        request = repo.create_request("Test input")
        decision = repo.create_decision(request.id, "Reasoning", "tool")

        repo.create_execution(
            decision.id, request.id, "tool1", {}, ExecutionStatus.SUCCESS
        )
        repo.create_execution(
            decision.id, request.id, "tool2", {}, ExecutionStatus.FAILED, error="Error"
        )

        executions = repo.get_executions_for_request(request.id)
        assert len(executions) == 2

    def test_audit_log(self, repo):
        """Test audit logging."""
        request = repo.create_request("Test input")

        log = repo.log(
            level="INFO",
            component="test",
            message="Test message",
            context={"key": "value"},
            request_id=request.id,
        )

        assert log.id is not None
        assert log.level == "INFO"
        assert log.component == "test"
        assert log.context == {"key": "value"}

    def test_get_audit_logs(self, repo):
        """Test querying audit logs."""
        request = repo.create_request("Test input")
        repo.log("INFO", "test", "Info message", request_id=request.id)
        repo.log("ERROR", "test", "Error message", request_id=request.id)
        repo.log("INFO", "other", "Other message")

        # Filter by request
        logs = repo.get_audit_logs(request_id=request.id)
        assert len(logs) == 2

        # Filter by level
        logs = repo.get_audit_logs(level="ERROR")
        assert len(logs) == 1
        assert logs[0].message == "Error message"

    def test_get_recent_requests(self, repo):
        """Test getting recent requests."""
        repo.create_request("Request 1")
        repo.create_request("Request 2")
        repo.create_request("Request 3")

        recent = repo.get_recent_requests(limit=2)
        assert len(recent) == 2
        assert recent[0].input_text == "Request 3"  # Most recent first

    def test_get_execution_stats(self, repo):
        """Test execution statistics."""
        request = repo.create_request("Test")
        decision = repo.create_decision(request.id, "Reasoning", "tool")

        repo.create_execution(
            decision.id, request.id, "tool1", {}, ExecutionStatus.SUCCESS, duration_ms=100
        )
        repo.create_execution(
            decision.id, request.id, "tool1", {}, ExecutionStatus.SUCCESS, duration_ms=200
        )
        repo.create_execution(
            decision.id, request.id, "tool2", {}, ExecutionStatus.FAILED
        )

        stats = repo.get_execution_stats()
        assert stats["total_executions"] == 3
        assert stats["by_status"]["success"] == 2
        assert stats["by_status"]["failed"] == 1
        assert stats["by_tool"]["tool1"] == 2
        assert stats["by_tool"]["tool2"] == 1
        assert stats["avg_duration_ms"] == 150
