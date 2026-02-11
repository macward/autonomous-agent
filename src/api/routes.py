"""API routes for the agent."""

import time
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.agent import AgentOrchestrator
from src.api.auth import RequireAPIKey
from src.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    ErrorResponse,
    HealthResponse,
)
from src.core.logging import get_logger

logger = get_logger("api.routes")

router = APIRouter()

# Rate limiter instance - will use app.state.limiter if available
limiter = Limiter(key_func=get_remote_address)


def get_orchestrator(request: Request) -> AgentOrchestrator:
    """Get the orchestrator from app state.

    Args:
        request: FastAPI request with app state

    Returns:
        Configured AgentOrchestrator
    """
    return request.app.state.orchestrator


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
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
@limiter.limit("10/minute")
async def agent_run(
    request: Request,
    body: AgentRunRequest,
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

    logger.info(f"Agent run request: {len(body.input)} chars")

    try:
        # Get orchestrator from app state
        orchestrator = get_orchestrator(request)

        # Create audit request
        audit_request = await orchestrator.repository.create_request(body.input)
        request_id = audit_request.id

        # Run agent
        result = await orchestrator.run(
            input_text=body.input,
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

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {type(e).__name__}",
        ) from e
