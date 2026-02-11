"""Groq connector implementation with tool use support."""

import asyncio
import json
from typing import Any

from groq import APIError, APITimeoutError, AsyncGroq, RateLimitError

from src.core.logging import get_logger
from src.llm.base import LLMConnector, LLMError, LLMResponse, Message

logger = get_logger("llm.groq")

# Default system prompt for the agent
DEFAULT_SYSTEM_PROMPT = """You are an autonomous agent that can execute tools to help users.

When given a task:
1. Analyze what needs to be done
2. Select the appropriate tool from the available tools
3. Provide the required parameters for the tool

Always use tools when they can help accomplish the task. Be precise with tool parameters.
If you cannot accomplish a task with the available tools, explain why clearly.

IMPORTANT: Only use tools that are explicitly provided. Never fabricate tool names or parameters."""


class GroqConnector(LLMConnector):
    """Connector for Groq API with tool use support.

    Features:
    - Async API calls
    - OpenAI-compatible tool use format
    - Automatic retry with exponential backoff
    - Token usage tracking
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
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
            api_key: Groq API key
            model: Model to use (default: llama-3.3-70b-versatile)
            max_tokens: Maximum tokens in response
        """
        self.client = AsyncGroq(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert tools to Groq format (OpenAI-compatible).

        Args:
            tools: Tools in generic LLM format

        Returns:
            Tools in Groq format
        """
        if not tools:
            return None

        # Groq uses OpenAI format which matches our generic format
        return tools

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to Groq format with tool use support.

        Args:
            messages: Messages in generic format

        Returns:
            Messages in Groq/OpenAI format
        """
        groq_messages = []

        for msg in messages:
            if msg.is_tool_result:
                # Tool result message
                groq_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_use_id,
                    "name": msg.tool_name,
                    "content": msg.content,
                })
            elif msg.tool_use_id and msg.tool_name:
                # Assistant message with tool call
                groq_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": msg.tool_use_id,
                        "type": "function",
                        "function": {
                            "name": msg.tool_name,
                            "arguments": json.dumps(msg.tool_input or {}),
                        },
                    }],
                })
            else:
                # Simple text message
                groq_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        return groq_messages

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Groq response into LLMResponse.

        Args:
            response: Raw Groq API response

        Returns:
            Parsed LLMResponse
        """
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        message = response.choices[0].message

        # Check for tool calls
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            return LLMResponse.tool(
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
                tool_use_id=tool_call.id,
                raw=response,
                usage=usage,
            )

        # Text response
        content = message.content or ""
        return LLMResponse.text(content, raw=response, usage=usage)

    async def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        system_prompt: str,
    ) -> Any:
        """Make API call with retry logic.

        Args:
            messages: Messages in Groq format
            tools: Tools in Groq format
            system_prompt: System prompt

        Returns:
            Raw API response

        Raises:
            LLMError: If all retries fail
        """
        last_error = None

        # Prepend system message
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        for attempt in range(self.MAX_RETRIES):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": full_messages,
                }

                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = await self.client.chat.completions.create(**kwargs)
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
                raise LLMError(str(e), retryable=False) from e

        raise LLMError(f"Max retries exceeded: {last_error}", retryable=True)

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Send messages to Groq and get a response.

        Args:
            messages: Conversation history
            tools: Available tools in LLM format
            system_prompt: Optional system prompt

        Returns:
            LLM response
        """
        try:
            groq_messages = self._convert_messages(messages)
            groq_tools = self._convert_tools(tools)
            system = system_prompt or DEFAULT_SYSTEM_PROMPT

            response = await self._call_with_retry(
                messages=groq_messages,
                tools=groq_tools,
                system_prompt=system,
            )

            return self._parse_response(response)

        except LLMError:
            raise
        except Exception as e:
            logger.exception("Unexpected error in complete")
            return LLMResponse.from_error(f"Unexpected error: {type(e).__name__}: {str(e)}")
