"""Base classes for tool definitions."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolPermission(str, Enum):
    """Permission levels for tools."""

    READ = "read"  # Can read data
    WRITE = "write"  # Can write/modify data
    EXECUTE = "execute"  # Can execute commands
    NETWORK = "network"  # Can access network


class ToolDefinition(BaseModel):
    """Definition of a tool for registration."""

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Human-readable description")
    input_schema: dict[str, Any] = Field(..., description="JSON Schema for input validation")
    permissions: list[ToolPermission] = Field(
        default_factory=list, description="Required permissions"
    )
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Execution timeout")
    max_output_size: int = Field(default=10000, ge=100, description="Max output size in bytes")


class Tool(ABC):
    """Abstract base class for tools.

    All tools must inherit from this class and implement the execute method.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool definition.

        Returns:
            ToolDefinition with name, description, schema, permissions
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with validated inputs.

        Args:
            **kwargs: Tool-specific inputs (already validated)

        Returns:
            Tool output as a dictionary

        Raises:
            ToolExecutionError: If execution fails
        """
        ...

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.definition.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self.definition.description

    @property
    def input_schema(self) -> dict[str, Any]:
        """Get input JSON Schema."""
        return self.definition.input_schema

    @property
    def permissions(self) -> list[ToolPermission]:
        """Get required permissions."""
        return self.definition.permissions

    def to_llm_format(self) -> dict[str, Any]:
        """Export tool in LLM-friendly format.

        Returns format compatible with OpenAI/Anthropic function calling.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolExecutionError(Exception):
    """Error during tool execution."""

    def __init__(self, message: str, tool_name: str, details: dict[str, Any] | None = None):
        self.message = message
        self.tool_name = tool_name
        self.details = details or {}
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolValidationError(Exception):
    """Error validating tool input."""

    def __init__(self, message: str, tool_name: str, errors: list[str]):
        self.message = message
        self.tool_name = tool_name
        self.errors = errors
        super().__init__(f"Tool '{tool_name}' validation failed: {message}")
