"""Agent orchestration logic."""

import json
from dataclasses import dataclass, field

from src.api.schemas import AgentRunStatus, ToolExecutionInfo
from src.core.logging import get_logger
from src.llm.base import LLMConnector, Message, ResponseType
from src.storage.models import RequestStatus
from src.storage.repository import AuditRepository
from src.tools.executor import ExecutionResultStatus, ToolExecutor
from src.tools.registry import ToolRegistry

logger = get_logger("api.agent")

MAX_ITERATIONS = 5  # Maximum tool calls per request


@dataclass
class AgentResult:
    """Result of an agent run."""

    status: AgentRunStatus
    output: str | None = None
    error: str | None = None
    tool_executions: list[ToolExecutionInfo] = field(default_factory=list)


class AgentOrchestrator:
    """Orchestrates the agent loop: LLM reasoning -> tool execution -> response.

    The agent loop:
    1. Send user input to LLM with available tools
    2. If LLM requests a tool, execute it and send result back
    3. Repeat until LLM returns a text response or max iterations reached
    """

    def __init__(
        self,
        llm: LLMConnector,
        registry: ToolRegistry,
        executor: ToolExecutor,
        repository: AuditRepository,
    ):
        """Initialize orchestrator.

        Args:
            llm: LLM connector for reasoning
            registry: Tool registry
            executor: Tool executor
            repository: Audit repository
        """
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.repository = repository

    async def run(self, input_text: str, request_id: str) -> AgentResult:
        """Run the agent loop for a user request.

        Args:
            input_text: User's input text
            request_id: Request ID for audit

        Returns:
            AgentResult with output or error
        """
        tool_executions: list[ToolExecutionInfo] = []
        messages: list[Message] = [Message(role="user", content=input_text)]

        # Get tools in LLM format
        tools = self.registry.export_for_llm()

        # Mark request as processing
        await self.repository.update_request_status(request_id, RequestStatus.PROCESSING)

        try:
            for iteration in range(MAX_ITERATIONS):
                logger.info(f"Agent iteration {iteration + 1}/{MAX_ITERATIONS}")

                # Get LLM response
                response = await self.llm.complete(
                    messages=messages,
                    tools=tools if tools else None,
                )

                # Handle error response
                if response.response_type == ResponseType.ERROR:
                    logger.error(f"LLM error: {response.error}")
                    await self.repository.update_request_status(
                        request_id,
                        RequestStatus.FAILED,
                        error=response.error,
                    )
                    return AgentResult(
                        status=AgentRunStatus.ERROR,
                        error=response.error,
                        tool_executions=tool_executions,
                    )

                # Handle text response (final answer)
                if response.response_type == ResponseType.TEXT:
                    logger.info("Agent completed with text response")
                    await self.repository.update_request_status(
                        request_id,
                        RequestStatus.COMPLETED,
                        final_output=response.content,
                    )
                    return AgentResult(
                        status=AgentRunStatus.SUCCESS,
                        output=response.content,
                        tool_executions=tool_executions,
                    )

                # Handle tool call
                if response.response_type == ResponseType.TOOL_CALL:
                    tool_call = response.tool_call
                    tool_use_id = response.tool_use_id
                    logger.info(f"Agent requested tool: {tool_call.name}")

                    # Record decision
                    decision = await self.repository.create_decision(
                        request_id=request_id,
                        reasoning=f"LLM decided to use tool: {tool_call.name}",
                        selected_tool=tool_call.name,
                        tool_input=tool_call.arguments,
                    )

                    # Execute tool
                    exec_result = await self.executor.execute(
                        tool_name=tool_call.name,
                        input_data=tool_call.arguments,
                        request_id=request_id,
                        decision_id=decision.id,
                    )

                    # Record tool execution info
                    tool_info = ToolExecutionInfo(
                        tool_name=tool_call.name,
                        status=exec_result.status.value,
                        duration_ms=exec_result.duration_ms,
                        output=exec_result.output,
                        error=exec_result.error,
                    )
                    tool_executions.append(tool_info)

                    # Handle tool execution errors
                    if exec_result.status in (
                        ExecutionResultStatus.VALIDATION_ERROR,
                        ExecutionResultStatus.EXECUTION_ERROR,
                        ExecutionResultStatus.TIMEOUT,
                        ExecutionResultStatus.PERMISSION_DENIED,
                    ):
                        tool_result = json.dumps({
                            "error": exec_result.error,
                            "status": exec_result.status.value,
                        })
                        is_error = True
                    else:
                        tool_result = json.dumps(exec_result.output)
                        is_error = False

                    # Build proper message structure for tool use flow
                    # Add assistant message with tool_use block
                    messages.append(Message(
                        role="assistant",
                        content="",  # Content is in tool_use block
                        tool_use_id=tool_use_id,
                        tool_name=tool_call.name,
                        tool_input=tool_call.arguments,
                    ))

                    # Add user message with tool_result block
                    messages.append(Message(
                        role="user",
                        content=tool_result,
                        is_tool_result=True,
                        tool_use_id=tool_use_id,
                        is_error=is_error,
                    ))

            # Max iterations reached
            logger.warning("Max iterations reached")
            await self.repository.update_request_status(
                request_id,
                RequestStatus.FAILED,
                error="Max iterations reached",
            )
            return AgentResult(
                status=AgentRunStatus.ERROR,
                error="Maximum tool iterations reached without final response",
                tool_executions=tool_executions,
            )

        except Exception as e:
            logger.exception("Unexpected error in agent run")
            await self.repository.update_request_status(
                request_id,
                RequestStatus.FAILED,
                error=str(e),
            )
            return AgentResult(
                status=AgentRunStatus.ERROR,
                error=f"Agent error: {type(e).__name__}: {str(e)}",
                tool_executions=tool_executions,
            )
