"""Autonomous Agent - Main entry point with startup initialization."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.agent import AgentOrchestrator
from src.api.routes import limiter, router
from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.llm.anthropic_connector import AnthropicConnector
from src.llm.base import LLMConnector
from src.llm.groq_connector import GroqConnector
from src.storage.database import close_database, get_database
from src.storage.repository import AuditRepository
from src.tools.builtin import register_builtin_tools
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown.

    Initializes all singleton dependencies at startup:
    - Database connection
    - Tool registry with built-in tools
    - Tool executor
    - LLM connector
    - Agent orchestrator
    """
    # Get settings from app state (set in create_app)
    settings = app.state.settings
    setup_logging(debug=settings.debug)
    logger.info("Starting autonomous agent...")

    # Initialize async database
    db = await get_database()
    logger.info("Database initialized")

    # Create singleton components
    registry = ToolRegistry()
    repository = AuditRepository(db=db)
    executor = ToolExecutor(registry=registry, repository=repository, settings=settings)

    # Register built-in tools once at startup
    register_builtin_tools(registry, executor)
    logger.info(f"Registered {len(registry.list_tools())} built-in tools")

    # Initialize LLM connector based on provider
    llm: LLMConnector
    provider = settings.llm_provider.lower()
    model = settings.effective_llm_model

    if provider == "groq":
        llm = GroqConnector(
            api_key=settings.llm_api_key,
            model=model,
        )
    elif provider == "anthropic":
        llm = AnthropicConnector(
            api_key=settings.llm_api_key,
            model=model,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'anthropic' or 'groq'.")

    logger.info(f"LLM connector initialized: {provider}/{model}")

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        llm=llm,
        registry=registry,
        executor=executor,
        repository=repository,
    )

    # Store in app state for access in routes
    app.state.registry = registry
    app.state.executor = executor
    app.state.repository = repository
    app.state.llm = llm
    app.state.orchestrator = orchestrator
    app.state.db = db

    logger.info("Agent startup complete")
    yield

    # Cleanup on shutdown
    logger.info("Shutting down autonomous agent...")
    await close_database()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Autonomous Agent",
        description="Secure autonomous agent with controlled tool execution",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Store settings in app.state before lifespan runs
    app.state.settings = settings

    # Add rate limiting
    if settings.rate_limit_enabled:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Add request size validation middleware
    @app.middleware("http")
    async def validate_content_length(request: Request, call_next):
        """Validate Content-Length header to prevent oversized requests."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > settings.max_request_size:
                    max_size = settings.max_request_size
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": f"Request too large. Max: {max_size} bytes"},
                    )
            except ValueError:
                pass  # Invalid content-length, let FastAPI handle it

        return await call_next(request)

    # Include API routes
    app.include_router(router)

    return app


# Create app instance
app = create_app()


def main():
    """Start the agent server."""
    settings = get_settings()

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
