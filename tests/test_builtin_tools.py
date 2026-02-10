"""Tests for built-in MVP tools."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.tools.base import ToolExecutionError
from src.tools.builtin import HealthCheckTool, ListDirTool, ReadFileTool
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


@pytest.fixture
def workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir)

        # Create test files and directories
        (workspace_path / "file1.txt").write_text("Hello World")
        (workspace_path / "file2.txt").write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        (workspace_path / "subdir").mkdir()
        (workspace_path / "subdir" / "nested.txt").write_text("Nested content")
        (workspace_path / "large.txt").write_text("x" * 1000)
        (workspace_path / "binary.exe").write_bytes(b"\x00\x01\x02")

        yield workspace_path


@pytest.fixture
def executor(workspace):
    """Create executor with workspace."""
    registry = ToolRegistry()
    settings = MagicMock()
    settings.max_tool_timeout = 30
    settings.max_output_size = 10000
    settings.workspace_root = workspace
    return ToolExecutor(registry=registry, settings=settings)


class TestHealthCheckTool:
    """Tests for HealthCheckTool."""

    async def test_execute(self):
        """Test health check execution."""
        tool = HealthCheckTool()
        result = await tool.execute()

        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "system" in result
        assert "platform" in result["system"]
        assert "python_version" in result["system"]

    def test_definition(self):
        """Test tool definition."""
        tool = HealthCheckTool()
        assert tool.name == "health_check"
        assert tool.permissions == []


class TestListDirTool:
    """Tests for ListDirTool."""

    async def test_list_root(self, executor, workspace):
        """Test listing workspace root."""
        tool = ListDirTool(executor)
        result = await tool.execute(path=".")

        assert result["count"] >= 4  # At least our test files
        names = [e["name"] for e in result["entries"]]
        assert "file1.txt" in names
        assert "subdir" in names

    async def test_list_subdir(self, executor, workspace):
        """Test listing subdirectory."""
        tool = ListDirTool(executor)
        result = await tool.execute(path="subdir")

        assert result["count"] == 1
        assert result["entries"][0]["name"] == "nested.txt"
        assert result["entries"][0]["type"] == "file"

    async def test_list_nonexistent(self, executor, workspace):
        """Test listing non-existent directory."""
        tool = ListDirTool(executor)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path="nonexistent")

        assert "does not exist" in exc_info.value.message

    async def test_list_file_as_dir(self, executor, workspace):
        """Test listing a file as directory."""
        tool = ListDirTool(executor)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path="file1.txt")

        assert "not a directory" in exc_info.value.message

    async def test_path_traversal_blocked(self, executor, workspace):
        """Test that path traversal is blocked."""
        tool = ListDirTool(executor)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path="../../../etc")

        assert "escapes workspace" in exc_info.value.message

    def test_definition(self, executor):
        """Test tool definition."""
        tool = ListDirTool(executor)
        assert tool.name == "list_dir"
        assert len(tool.permissions) == 1


class TestReadFileTool:
    """Tests for ReadFileTool."""

    async def test_read_file(self, executor, workspace):
        """Test reading a file."""
        tool = ReadFileTool(executor)
        result = await tool.execute(path="file1.txt")

        assert result["content"] == "Hello World"
        assert result["size"] == 11
        assert result["truncated"] is False

    async def test_read_with_line_limit(self, executor, workspace):
        """Test reading file with line limit."""
        tool = ReadFileTool(executor)
        result = await tool.execute(path="file2.txt", max_lines=2)

        assert result["lines"] == 2
        assert result["truncated"] is True
        assert result["content"] == "Line 1\nLine 2"

    async def test_read_nested_file(self, executor, workspace):
        """Test reading nested file."""
        tool = ReadFileTool(executor)
        result = await tool.execute(path="subdir/nested.txt")

        assert result["content"] == "Nested content"

    async def test_read_nonexistent(self, executor, workspace):
        """Test reading non-existent file."""
        tool = ReadFileTool(executor)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path="nonexistent.txt")

        assert "does not exist" in exc_info.value.message

    async def test_read_directory(self, executor, workspace):
        """Test reading a directory."""
        tool = ReadFileTool(executor)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path="subdir")

        assert "not a file" in exc_info.value.message

    async def test_read_binary_file(self, executor, workspace):
        """Test that binary files are rejected."""
        tool = ReadFileTool(executor)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path="binary.exe")

        assert "binary file" in exc_info.value.message

    async def test_path_traversal_blocked(self, executor, workspace):
        """Test that path traversal is blocked."""
        tool = ReadFileTool(executor)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path="../../../etc/passwd")

        assert "escapes workspace" in exc_info.value.message

    def test_definition(self, executor):
        """Test tool definition."""
        tool = ReadFileTool(executor)
        assert tool.name == "read_file"
        assert len(tool.permissions) == 1


class TestToolRegistration:
    """Tests for tool registration."""

    def test_register_builtin_tools(self, executor):
        """Test registering all built-in tools."""
        from src.tools.builtin import register_builtin_tools

        registry = ToolRegistry()
        register_builtin_tools(registry, executor)

        assert "health_check" in registry
        assert "list_dir" in registry
        assert "read_file" in registry
        assert len(registry) == 3

    def test_export_for_llm(self, executor):
        """Test exporting tools in LLM format."""
        from src.tools.builtin import register_builtin_tools

        registry = ToolRegistry()
        register_builtin_tools(registry, executor)

        exported = registry.export_for_llm()
        assert len(exported) == 3

        names = [t["function"]["name"] for t in exported]
        assert "health_check" in names
        assert "list_dir" in names
        assert "read_file" in names
