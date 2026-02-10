"""Tool registry for managing available tools."""

from typing import Any

import jsonschema
from jsonschema import Draft7Validator, ValidationError

from src.core.logging import get_logger
from src.tools.base import Tool, ToolValidationError

logger = get_logger("tools.registry")


class ToolRegistry:
    """Registry for managing available tools.

    Provides registration, lookup, validation, and export functionality.
    Only tools registered here can be executed by the agent.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, Draft7Validator] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already registered
        """
        name = tool.name

        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        # Validate the schema itself is valid JSON Schema
        try:
            Draft7Validator.check_schema(tool.input_schema)
        except jsonschema.SchemaError as e:
            raise ValueError(f"Tool '{name}' has invalid input schema: {e.message}")

        # Create and cache validator
        self._validators[name] = Draft7Validator(tool.input_schema)
        self._tools[name] = tool

        logger.info(f"Registered tool: {name}")

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Tool name to remove

        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")

        del self._tools[name]
        del self._validators[name]
        logger.info(f"Unregistered tool: {name}")

    def get(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def get_or_raise(self, name: str) -> Tool:
        """Get a tool by name, raising if not found.

        Args:
            name: Tool name

        Returns:
            Tool instance

        Raises:
            KeyError: If tool not found
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry")
        return tool

    def validate_input(self, name: str, input_data: dict[str, Any]) -> None:
        """Validate input against tool's JSON Schema.

        Args:
            name: Tool name
            input_data: Input to validate

        Raises:
            KeyError: If tool not found
            ToolValidationError: If validation fails
        """
        validator = self._validators.get(name)
        if validator is None:
            raise KeyError(f"Tool '{name}' not found in registry")

        errors: list[str] = []
        for error in validator.iter_errors(input_data):
            path = ".".join(str(p) for p in error.absolute_path) or "(root)"
            errors.append(f"{path}: {error.message}")

        if errors:
            raise ToolValidationError(
                message=f"Input validation failed with {len(errors)} error(s)",
                tool_name=name,
                errors=errors,
            )

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools.

        Returns:
            List of tool instances
        """
        return list(self._tools.values())

    def export_for_llm(self) -> list[dict[str, Any]]:
        """Export all tools in LLM-friendly format.

        Returns format compatible with OpenAI/Anthropic function calling.

        Returns:
            List of tool definitions in LLM format
        """
        return [tool.to_llm_format() for tool in self._tools.values()]

    def export_descriptions(self) -> str:
        """Export tool descriptions as formatted text.

        Useful for including in system prompts.

        Returns:
            Formatted string with tool descriptions
        """
        lines = ["Available tools:", ""]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.permissions:
                perms = ", ".join(p.value for p in tool.permissions)
                lines.append(f"  Permissions: {perms}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools


# Global registry instance
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None
