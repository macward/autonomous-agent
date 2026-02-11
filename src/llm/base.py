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
    tool_use_id: str | None = None  # Anthropic tool_use block ID
    error: str | None = None
    raw_response: Any = None
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def text(
        cls, content: str, raw: Any = None, usage: dict[str, int] | None = None
    ) -> "LLMResponse":
        """Create a text response."""
        return cls(
            response_type=ResponseType.TEXT,
            content=content,
            raw_response=raw,
            usage=usage or {},
        )

    @classmethod
    def tool(
        cls,
        name: str,
        arguments: dict[str, Any],
        tool_use_id: str | None = None,
        raw: Any = None,
        usage: dict[str, int] | None = None,
    ) -> "LLMResponse":
        """Create a tool call response."""
        return cls(
            response_type=ResponseType.TOOL_CALL,
            tool_call=ToolCall(name=name, arguments=arguments),
            tool_use_id=tool_use_id,
            raw_response=raw,
            usage=usage or {},
        )

    @classmethod
    def from_error(cls, message: str) -> "LLMResponse":
        """Create an error response.

        Note: Named 'from_error' to avoid shadowing the 'error' attribute.
        """
        return cls(
            response_type=ResponseType.ERROR,
            error=message,
        )


@dataclass
class Message:
    """A message in the conversation.

    Supports both simple text messages and structured tool use messages.
    """

    role: str  # "user", "assistant"
    content: str

    # Tool use fields (for assistant messages with tool_use blocks)
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None

    # Tool result fields (for user messages with tool_result blocks)
    is_tool_result: bool = False
    is_error: bool = False


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


class LLMError(Exception):
    """Error from LLM operations."""

    def __init__(self, message: str, retryable: bool = False):
        self.message = message
        self.retryable = retryable
        super().__init__(message)
