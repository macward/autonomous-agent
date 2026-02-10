"""Tests for LLM connector."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.anthropic_connector import AnthropicConnector
from src.llm.base import LLMError, LLMResponse, Message, ResponseType


class MockContentBlock:
    """Mock content block."""

    def __init__(self, block_type: str, text: str = None, name: str = None, input: dict = None):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input


class MockUsage:
    """Mock usage info."""

    def __init__(self, input_tokens: int = 100, output_tokens: int = 50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockResponse:
    """Mock Anthropic response."""

    def __init__(self, content: list, usage: MockUsage = None):
        self.content = content
        self.usage = usage or MockUsage()


class TestLLMResponse:
    """Tests for LLMResponse class."""

    def test_text_response(self):
        """Test creating text response."""
        response = LLMResponse.text("Hello world", usage={"input_tokens": 10})

        assert response.response_type == ResponseType.TEXT
        assert response.content == "Hello world"
        assert response.tool_call is None
        assert response.usage["input_tokens"] == 10

    def test_tool_response(self):
        """Test creating tool call response."""
        response = LLMResponse.tool(
            name="read_file",
            arguments={"path": "test.txt"},
        )

        assert response.response_type == ResponseType.TOOL_CALL
        assert response.tool_call.name == "read_file"
        assert response.tool_call.arguments == {"path": "test.txt"}

    def test_error_response(self):
        """Test creating error response."""
        response = LLMResponse.error("Something went wrong")

        assert response.response_type == ResponseType.ERROR
        assert response.error == "Something went wrong"


class TestAnthropicConnector:
    """Tests for AnthropicConnector class."""

    @pytest.fixture
    def connector(self):
        """Create connector with mock client."""
        with patch("src.llm.anthropic_connector.anthropic.AsyncAnthropic"):
            return AnthropicConnector(api_key="test-key")

    def test_convert_tools(self, connector):
        """Test converting tools to Anthropic format."""
        tools = [{
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

        result = connector._convert_tools(tools)

        assert len(result) == 1
        assert result[0]["name"] == "test_tool"
        assert result[0]["description"] == "A test tool"
        assert "input_schema" in result[0]

    def test_convert_tools_none(self, connector):
        """Test converting None tools."""
        assert connector._convert_tools(None) is None

    def test_convert_messages(self, connector):
        """Test converting messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]

        result = connector._convert_messages(messages)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_parse_response_text(self, connector):
        """Test parsing text response."""
        mock_response = MockResponse(
            content=[MockContentBlock("text", text="Hello world")],
        )

        result = connector._parse_response(mock_response)

        assert result.response_type == ResponseType.TEXT
        assert result.content == "Hello world"

    def test_parse_response_tool(self, connector):
        """Test parsing tool use response."""
        mock_response = MockResponse(
            content=[MockContentBlock("tool_use", name="read_file", input={"path": "test.txt"})],
        )

        result = connector._parse_response(mock_response)

        assert result.response_type == ResponseType.TOOL_CALL
        assert result.tool_call.name == "read_file"
        assert result.tool_call.arguments == {"path": "test.txt"}

    def test_parse_response_mixed(self, connector):
        """Test parsing response with text and tool use."""
        mock_response = MockResponse(
            content=[
                MockContentBlock("text", text="Let me help"),
                MockContentBlock("tool_use", name="list_dir", input={"path": "."}),
            ],
        )

        result = connector._parse_response(mock_response)

        # Tool use takes precedence
        assert result.response_type == ResponseType.TOOL_CALL
        assert result.tool_call.name == "list_dir"

    async def test_complete_success(self, connector):
        """Test successful completion."""
        mock_response = MockResponse(
            content=[MockContentBlock("text", text="Response text")],
        )
        connector.client.messages.create = AsyncMock(return_value=mock_response)

        result = await connector.complete([Message(role="user", content="Hello")])

        assert result.response_type == ResponseType.TEXT
        assert result.content == "Response text"

    async def test_complete_with_tools(self, connector):
        """Test completion with tools."""
        mock_response = MockResponse(
            content=[MockContentBlock("tool_use", name="health_check", input={})],
        )
        connector.client.messages.create = AsyncMock(return_value=mock_response)

        tools = [{
            "type": "function",
            "function": {
                "name": "health_check",
                "description": "Check health",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

        result = await connector.complete(
            [Message(role="user", content="Check status")],
            tools=tools,
        )

        assert result.response_type == ResponseType.TOOL_CALL
        assert result.tool_call.name == "health_check"

    async def test_complete_api_error(self, connector):
        """Test handling API errors."""
        from anthropic import APIError

        connector.client.messages.create = AsyncMock(
            side_effect=APIError(message="API error", request=MagicMock(), body=None)
        )

        with pytest.raises(LLMError) as exc_info:
            await connector.complete([Message(role="user", content="Hello")])

        assert not exc_info.value.retryable


class TestMessage:
    """Tests for Message class."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestLLMError:
    """Tests for LLMError class."""

    def test_retryable_error(self):
        """Test retryable error."""
        error = LLMError("Rate limited", retryable=True)
        assert error.retryable is True
        assert error.message == "Rate limited"

    def test_non_retryable_error(self):
        """Test non-retryable error."""
        error = LLMError("Invalid API key", retryable=False)
        assert error.retryable is False
