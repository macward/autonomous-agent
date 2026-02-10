"""Tests for tool registry."""

from typing import Any

import pytest

from src.tools.base import Tool, ToolDefinition, ToolPermission, ToolValidationError
from src.tools.registry import ToolRegistry


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool",
        schema: dict[str, Any] | None = None,
        permissions: list[ToolPermission] | None = None,
    ):
        self._definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=schema
            or {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            permissions=permissions or [],
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"result": kwargs.get("value", "default")}


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return ToolRegistry()


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self, registry):
        """Test registering a tool."""
        tool = MockTool()
        registry.register(tool)

        assert "mock_tool" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self, registry):
        """Test that registering duplicate tool raises error."""
        tool1 = MockTool()
        tool2 = MockTool()

        registry.register(tool1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)

    def test_register_invalid_schema_raises(self, registry):
        """Test that invalid JSON Schema raises error."""
        tool = MockTool(
            schema={"type": "invalid_type"}  # Invalid type
        )

        with pytest.raises(ValueError, match="invalid input schema"):
            registry.register(tool)

    def test_unregister_tool(self, registry):
        """Test unregistering a tool."""
        tool = MockTool()
        registry.register(tool)
        registry.unregister("mock_tool")

        assert "mock_tool" not in registry
        assert len(registry) == 0

    def test_unregister_nonexistent_raises(self, registry):
        """Test unregistering non-existent tool raises error."""
        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_get_tool(self, registry):
        """Test getting a tool by name."""
        tool = MockTool()
        registry.register(tool)

        result = registry.get("mock_tool")
        assert result is tool

    def test_get_nonexistent_returns_none(self, registry):
        """Test getting non-existent tool returns None."""
        assert registry.get("nonexistent") is None

    def test_get_or_raise(self, registry):
        """Test get_or_raise returns tool."""
        tool = MockTool()
        registry.register(tool)

        result = registry.get_or_raise("mock_tool")
        assert result is tool

    def test_get_or_raise_nonexistent(self, registry):
        """Test get_or_raise raises for non-existent tool."""
        with pytest.raises(KeyError):
            registry.get_or_raise("nonexistent")

    def test_validate_input_valid(self, registry):
        """Test validating valid input."""
        tool = MockTool()
        registry.register(tool)

        # Should not raise
        registry.validate_input("mock_tool", {"value": "test"})

    def test_validate_input_missing_required(self, registry):
        """Test validating input with missing required field."""
        tool = MockTool()
        registry.register(tool)

        with pytest.raises(ToolValidationError) as exc_info:
            registry.validate_input("mock_tool", {})

        assert "value" in exc_info.value.errors[0]

    def test_validate_input_wrong_type(self, registry):
        """Test validating input with wrong type."""
        tool = MockTool()
        registry.register(tool)

        with pytest.raises(ToolValidationError) as exc_info:
            registry.validate_input("mock_tool", {"value": 123})

        assert len(exc_info.value.errors) == 1

    def test_validate_input_nonexistent_tool(self, registry):
        """Test validating input for non-existent tool."""
        with pytest.raises(KeyError):
            registry.validate_input("nonexistent", {})

    def test_list_tools(self, registry):
        """Test listing tool names."""
        registry.register(MockTool(name="tool1"))
        registry.register(MockTool(name="tool2"))

        tools = registry.list_tools()
        assert sorted(tools) == ["tool1", "tool2"]

    def test_get_all_tools(self, registry):
        """Test getting all tools."""
        tool1 = MockTool(name="tool1")
        tool2 = MockTool(name="tool2")
        registry.register(tool1)
        registry.register(tool2)

        tools = registry.get_all_tools()
        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools

    def test_export_for_llm(self, registry):
        """Test exporting tools in LLM format."""
        registry.register(MockTool(name="test_tool", description="Test description"))

        exported = registry.export_for_llm()
        assert len(exported) == 1
        assert exported[0]["type"] == "function"
        assert exported[0]["function"]["name"] == "test_tool"
        assert exported[0]["function"]["description"] == "Test description"
        assert "parameters" in exported[0]["function"]

    def test_export_descriptions(self, registry):
        """Test exporting tool descriptions."""
        registry.register(
            MockTool(
                name="file_reader",
                description="Reads files",
                permissions=[ToolPermission.READ],
            )
        )

        text = registry.export_descriptions()
        assert "file_reader" in text
        assert "Reads files" in text
        assert "read" in text


class TestTool:
    """Tests for Tool base class."""

    def test_tool_properties(self):
        """Test tool property accessors."""
        tool = MockTool(
            name="my_tool",
            description="My description",
            permissions=[ToolPermission.READ, ToolPermission.WRITE],
        )

        assert tool.name == "my_tool"
        assert tool.description == "My description"
        assert tool.permissions == [ToolPermission.READ, ToolPermission.WRITE]

    def test_to_llm_format(self):
        """Test LLM format export."""
        tool = MockTool(name="test", description="Test tool")
        fmt = tool.to_llm_format()

        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "test"
        assert fmt["function"]["description"] == "Test tool"

    async def test_execute(self):
        """Test tool execution."""
        tool = MockTool()
        result = await tool.execute(value="hello")

        assert result == {"result": "hello"}
