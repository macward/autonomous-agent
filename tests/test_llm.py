"""Tests for LLM connector."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.anthropic_connector import AnthropicConnector
from src.llm.base import LLMError, LLMResponse, Message, ResponseType
from src.llm.groq_connector import GroqConnector


class MockContentBlock:
    """Mock content block."""

    def __init__(
        self,
        block_type: str,
        text: str = None,
        name: str = None,
        input: dict = None,
        id: str = None,
    ):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input
        self.id = id or "mock_tool_use_id"


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
            tool_use_id="test_id_123",
        )

        assert response.response_type == ResponseType.TOOL_CALL
        assert response.tool_call.name == "read_file"
        assert response.tool_call.arguments == {"path": "test.txt"}
        assert response.tool_use_id == "test_id_123"

    def test_error_response(self):
        """Test creating error response."""
        response = LLMResponse.from_error("Something went wrong")

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

    def test_convert_messages_simple(self, connector):
        """Test converting simple text messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]

        result = connector._convert_messages(messages)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_convert_messages_with_tool_use(self, connector):
        """Test converting messages with tool_use blocks."""
        messages = [
            Message(role="user", content="Check health"),
            Message(
                role="assistant",
                content="",
                tool_use_id="toolu_123",
                tool_name="health_check",
                tool_input={},
            ),
            Message(
                role="user",
                content='{"status": "ok"}',
                is_tool_result=True,
                tool_use_id="toolu_123",
                is_error=False,
            ),
        ]

        result = connector._convert_messages(messages)

        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Check health"}

        # Tool use block
        assert result[1]["role"] == "assistant"
        assert result[1]["content"][0]["type"] == "tool_use"
        assert result[1]["content"][0]["id"] == "toolu_123"
        assert result[1]["content"][0]["name"] == "health_check"

        # Tool result block
        assert result[2]["role"] == "user"
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[2]["content"][0]["tool_use_id"] == "toolu_123"
        assert result[2]["content"][0]["is_error"] is False

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
            content=[MockContentBlock(
                "tool_use",
                name="read_file",
                input={"path": "test.txt"},
                id="toolu_456",
            )],
        )

        result = connector._parse_response(mock_response)

        assert result.response_type == ResponseType.TOOL_CALL
        assert result.tool_call.name == "read_file"
        assert result.tool_call.arguments == {"path": "test.txt"}
        assert result.tool_use_id == "toolu_456"

    def test_parse_response_mixed(self, connector):
        """Test parsing response with text and tool use."""
        mock_response = MockResponse(
            content=[
                MockContentBlock("text", text="Let me help"),
                MockContentBlock("tool_use", name="list_dir", input={"path": "."}, id="toolu_789"),
            ],
        )

        result = connector._parse_response(mock_response)

        # Tool use takes precedence
        assert result.response_type == ResponseType.TOOL_CALL
        assert result.tool_call.name == "list_dir"
        assert result.tool_use_id == "toolu_789"

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
            content=[MockContentBlock("tool_use", name="health_check", input={}, id="toolu_abc")],
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
        assert result.tool_use_id == "toolu_abc"

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

    def test_create_tool_use_message(self):
        """Test creating a tool use message."""
        msg = Message(
            role="assistant",
            content="",
            tool_use_id="toolu_123",
            tool_name="test_tool",
            tool_input={"key": "value"},
        )
        assert msg.tool_use_id == "toolu_123"
        assert msg.tool_name == "test_tool"
        assert msg.tool_input == {"key": "value"}

    def test_create_tool_result_message(self):
        """Test creating a tool result message."""
        msg = Message(
            role="user",
            content="result data",
            is_tool_result=True,
            tool_use_id="toolu_123",
            is_error=False,
        )
        assert msg.is_tool_result is True
        assert msg.tool_use_id == "toolu_123"
        assert msg.is_error is False


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


# ============ Groq Connector Tests ============


class MockGroqUsage:
    """Mock Groq usage info."""

    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockGroqFunction:
    """Mock Groq function call."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class MockGroqToolCall:
    """Mock Groq tool call."""

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = MockGroqFunction(name, arguments)


class MockGroqMessage:
    """Mock Groq message."""

    def __init__(self, content: str | None = None, tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls


class MockGroqChoice:
    """Mock Groq choice."""

    def __init__(self, message: MockGroqMessage):
        self.message = message


class MockGroqResponse:
    """Mock Groq response."""

    def __init__(self, message: MockGroqMessage, usage: MockGroqUsage = None):
        self.choices = [MockGroqChoice(message)]
        self.usage = usage or MockGroqUsage()


class TestGroqConnector:
    """Tests for GroqConnector class."""

    @pytest.fixture
    def connector(self):
        """Create connector with mock client."""
        with patch("src.llm.groq_connector.AsyncGroq"):
            return GroqConnector(api_key="test-key")

    def test_convert_tools(self, connector):
        """Test that tools pass through unchanged (OpenAI format)."""
        tools = [{
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

        result = connector._convert_tools(tools)

        assert result == tools

    def test_convert_tools_none(self, connector):
        """Test converting None tools."""
        assert connector._convert_tools(None) is None

    def test_convert_messages_simple(self, connector):
        """Test converting simple text messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]

        result = connector._convert_messages(messages)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_convert_messages_with_tool_use(self, connector):
        """Test converting messages with tool calls."""
        messages = [
            Message(role="user", content="Check health"),
            Message(
                role="assistant",
                content="",
                tool_use_id="call_123",
                tool_name="health_check",
                tool_input={},
            ),
            Message(
                role="user",
                content='{"status": "ok"}',
                is_tool_result=True,
                tool_use_id="call_123",
                tool_name="health_check",
            ),
        ]

        result = connector._convert_messages(messages)

        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Check health"}

        # Tool call (assistant)
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] is None
        assert result[1]["tool_calls"][0]["id"] == "call_123"
        assert result[1]["tool_calls"][0]["function"]["name"] == "health_check"

        # Tool result
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_123"

    def test_parse_response_text(self, connector):
        """Test parsing text response."""
        mock_response = MockGroqResponse(
            message=MockGroqMessage(content="Hello world"),
        )

        result = connector._parse_response(mock_response)

        assert result.response_type == ResponseType.TEXT
        assert result.content == "Hello world"

    def test_parse_response_tool(self, connector):
        """Test parsing tool use response."""
        mock_response = MockGroqResponse(
            message=MockGroqMessage(
                content=None,
                tool_calls=[MockGroqToolCall(
                    id="call_456",
                    name="read_file",
                    arguments='{"path": "test.txt"}',
                )],
            ),
        )

        result = connector._parse_response(mock_response)

        assert result.response_type == ResponseType.TOOL_CALL
        assert result.tool_call.name == "read_file"
        assert result.tool_call.arguments == {"path": "test.txt"}
        assert result.tool_use_id == "call_456"

    async def test_complete_success(self, connector):
        """Test successful completion."""
        mock_response = MockGroqResponse(
            message=MockGroqMessage(content="Response text"),
        )
        connector.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await connector.complete([Message(role="user", content="Hello")])

        assert result.response_type == ResponseType.TEXT
        assert result.content == "Response text"

    async def test_complete_with_tools(self, connector):
        """Test completion with tools."""
        mock_response = MockGroqResponse(
            message=MockGroqMessage(
                content=None,
                tool_calls=[MockGroqToolCall(
                    id="call_abc",
                    name="health_check",
                    arguments="{}",
                )],
            ),
        )
        connector.client.chat.completions.create = AsyncMock(return_value=mock_response)

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
        assert result.tool_use_id == "call_abc"

    async def test_complete_api_error(self, connector):
        """Test handling API errors."""
        from groq import APIError

        connector.client.chat.completions.create = AsyncMock(
            side_effect=APIError(message="API error", request=MagicMock(), body=None)
        )

        with pytest.raises(LLMError) as exc_info:
            await connector.complete([Message(role="user", content="Hello")])

        assert not exc_info.value.retryable
