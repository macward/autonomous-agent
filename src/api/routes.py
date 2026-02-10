"""API routes for the agent."""

import time
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status

from src.api.agent import AgentOrchestrator
from src.api.auth import RequireAPIKey
from src.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStatus,
    ErrorResponse,
    HealthResponse,
)
from src.core.config import get_settings
from src.core.logging import get_logger
from src.llm.anthropic_connector import AnthropicConnector
from src.storage.database import get_database
from src.storage.repository import AuditRepository
from src.tools.builtin import register_builtin_tools
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry

logger = get_logger("api.routes")

router = APIRouter()


def _get_orchestrator() -> AgentOrchestrator:
    """Create and configure the agent orchestrator."""
    settings = get_settings()

    # Initialize components
    registry = ToolRegistry()
    db = get_database()
    repository = AuditRepository(db=db)
    executor = ToolExecutor(registry=registry, repository=repository, settings=settings)

    # Register built-in tools
    register_builtin_tools(registry, executor)

    # Initialize LLM connector
    llm = AnthropicConnector(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    return AgentOrchestrator(
        llm=llm,
        registry=registry,
        executor=executor,
        repository=repository,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns basic health status. Does not require authentication.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now(UTC),
    )


@router.post(
    "/agent/run",
    response_model=AgentRunResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def agent_run(
    request: AgentRunRequest,
    api_key: RequireAPIKey,
) -> AgentRunResponse:
    """Execute the agent with the given input.

    The agent will:
    1. Analyze the input
    2. Select and execute appropriate tools
    3. Return a response

    Requires X-API-Key header for authentication.
    """
    start_time = time.perf_counter()
    created_at = datetime.now(UTC)

    logger.info(f"Agent run request: {len(request.input)} chars")

    try:
        # Get orchestrator
        orchestrator = _get_orchestrator()

        # Create audit request
        audit_request = orchestrator.repository.create_request(request.input)
        request_id = audit_request.id

        # Run agent
        result = await orchestrator.run(
            input_text=request.input,
            request_id=request_id,
        )

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return AgentRunResponse(
            request_id=request_id,
            status=result.status,
            output=result.output,
            error=result.error,
            tool_executions=result.tool_executions,
            created_at=created_at,
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.exception("Unexpected error in agent_run")
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {type(e).__name__}",
        )
