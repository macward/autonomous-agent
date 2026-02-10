"""Autonomous Agent - Main entry point."""

import uvicorn
from fastapi import FastAPI

from src.core.config import get_settings
from src.core.logging import setup_logging

app = FastAPI(
    title="Autonomous Agent",
    description="Secure autonomous agent with controlled tool execution",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Start the agent server."""
    settings = get_settings()
    setup_logging(debug=settings.debug)

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
