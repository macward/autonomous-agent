"""Base classes for LLM connectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseType(str, Enum):
    """Type of LLM response."""

    TEXT = "text"  # Plain text response
    TOOL_CALL = "tool_call"  # Request to call a tool
    ERROR = "error"  # Error response


@dataclass
class ToolCall:
    """Represents a tool call request from the LLM."""

    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM."""

    response_type: ResponseType
    content: str | None = None
    tool_call: ToolCall | None = None
    error: str | None = None
    raw_response: Any = None
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def text(cls, content: str, raw: Any = None, usage: dict[str, int] | None = None) -> "LLMResponse":
        """Create a text response."""
        return cls(
            response_type=ResponseType.TEXT,
            content=content,
            raw_response=raw,
            usage=usage or {},
        )

    @classmethod
    def tool(cls, name: str, arguments: dict[str, Any], raw: Any = None, usage: dict[str, int] | None = None) -> "LLMResponse":
        """Create a tool call response."""
        return cls(
            response_type=ResponseType.TOOL_CALL,
            tool_call=ToolCall(name=name, arguments=arguments),
            raw_response=raw,
            usage=usage or {},
        )

    @classmethod
    def error(cls, message: str) -> "LLMResponse":
        """Create an error response."""
        return cls(
            response_type=ResponseType.ERROR,
            error=message,
        )


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "user", "assistant", "system"
    content: str


class LLMConnector(ABC):
    """Abstract base class for LLM connectors.

    All LLM integrations must implement this interface.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and get a response.

        Args:
            messages: Conversation history
            tools: Available tools in LLM format
            system_prompt: Optional system prompt

        Returns:
            LLM response (text, tool call, or error)
        """
        ...

    @abstractmethod
    async def complete_with_tool_result(
        self,
        messages: list[Message],
        tool_name: str,
        tool_result: str,
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Continue conversation after a tool execution.

        Args:
            messages: Conversation history
            tool_name: Name of the executed tool
            tool_result: Result from tool execution
            tools: Available tools in LLM format
            system_prompt: Optional system prompt

        Returns:
            LLM response
        """
        ...


class LLMError(Exception):
    """Error from LLM operations."""

    def __init__(self, message: str, retryable: bool = False):
        self.message = message
        self.retryable = retryable
        super().__init__(message)
