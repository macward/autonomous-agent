"""Anthropic Claude connector implementation."""

import asyncio
import json
from typing import Any

import anthropic
from anthropic import APIError, APITimeoutError, RateLimitError

from src.core.logging import get_logger
from src.llm.base import LLMConnector, LLMError, LLMResponse, Message

logger = get_logger("llm.anthropic")

# Default system prompt for the agent
DEFAULT_SYSTEM_PROMPT = """You are an autonomous agent that can execute tools to help users.

When given a task:
1. Analyze what needs to be done
2. Select the appropriate tool from the available tools
3. Provide the required parameters for the tool

Always use tools when they can help accomplish the task. Be precise with tool parameters.
If you cannot accomplish a task with the available tools, explain why clearly.

IMPORTANT: Only use tools that are explicitly provided. Never fabricate tool names or parameters."""


class AnthropicConnector(LLMConnector):
    """Connector for Anthropic Claude API.

    Features:
    - Async API calls
    - Tool use support
    - Automatic retry with exponential backoff
    - Token usage tracking
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int = 4096,
    ):
        """Initialize connector.

        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-sonnet-4-20250514)
            max_tokens: Maximum tokens in response
        """
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert tools from generic format to Anthropic format.

        Args:
            tools: Tools in generic LLM format

        Returns:
            Tools in Anthropic format
        """
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"],
                })
            else:
                # Already in Anthropic format
                anthropic_tools.append(tool)

        return anthropic_tools

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format.

        Args:
            messages: Messages in generic format

        Returns:
            Messages in Anthropic format
        """
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response into LLMResponse.

        Args:
            response: Raw Anthropic API response

        Returns:
            Parsed LLMResponse
        """
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        # Check for tool use
        for block in response.content:
            if block.type == "tool_use":
                return LLMResponse.tool(
                    name=block.name,
                    arguments=block.input,
                    raw=response,
                    usage=usage,
                )

        # Extract text content
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        content = "\n".join(text_parts) if text_parts else ""
        return LLMResponse.text(content, raw=response, usage=usage)

    async def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        system_prompt: str,
    ) -> Any:
        """Make API call with retry logic.

        Args:
            messages: Messages in Anthropic format
            tools: Tools in Anthropic format
            system_prompt: System prompt

        Returns:
            Raw API response

        Raises:
            LLMError: If all retries fail
        """
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": system_prompt,
                    "messages": messages,
                }

                if tools:
                    kwargs["tools"] = tools

                response = await self.client.messages.create(**kwargs)
                return response

            except RateLimitError as e:
                last_error = e
                delay = self.BASE_DELAY * (2 ** attempt)
                logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)

            except APITimeoutError as e:
                last_error = e
                delay = self.BASE_DELAY * (2 ** attempt)
                logger.warning(f"Timeout, retrying in {delay}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)

            except APIError as e:
                # Non-retryable errors
                logger.error(f"API error: {e}")
                raise LLMError(str(e), retryable=False)

        raise LLMError(f"Max retries exceeded: {last_error}", retryable=True)

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Send messages to Claude and get a response.

        Args:
            messages: Conversation history
            tools: Available tools in LLM format
            system_prompt: Optional system prompt

        Returns:
            LLM response
        """
        try:
            anthropic_messages = self._convert_messages(messages)
            anthropic_tools = self._convert_tools(tools)
            system = system_prompt or DEFAULT_SYSTEM_PROMPT

            response = await self._call_with_retry(
                messages=anthropic_messages,
                tools=anthropic_tools,
                system_prompt=system,
            )

            return self._parse_response(response)

        except LLMError:
            raise
        except Exception as e:
            logger.exception("Unexpected error in complete")
            return LLMResponse.error(f"Unexpected error: {type(e).__name__}: {str(e)}")

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
            messages: Conversation history (should include the assistant's tool_use)
            tool_name: Name of the executed tool
            tool_result: Result from tool execution
            tools: Available tools in LLM format
            system_prompt: Optional system prompt

        Returns:
            LLM response
        """
        try:
            anthropic_messages = self._convert_messages(messages)
            anthropic_tools = self._convert_tools(tools)
            system = system_prompt or DEFAULT_SYSTEM_PROMPT

            # Get the tool_use_id from the last assistant message
            # For simplicity, we'll construct the tool result message
            tool_result_message = {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool_use_placeholder",  # Would need to track this properly
                    "content": tool_result,
                }],
            }

            # Add the tool result to messages
            anthropic_messages.append(tool_result_message)

            response = await self._call_with_retry(
                messages=anthropic_messages,
                tools=anthropic_tools,
                system_prompt=system,
            )

            return self._parse_response(response)

        except LLMError:
            raise
        except Exception as e:
            logger.exception("Unexpected error in complete_with_tool_result")
            return LLMResponse.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
