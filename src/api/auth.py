"""API Key authentication middleware."""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger("api.auth")

# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Annotated[str | None, Security(api_key_header)],
) -> str:
    """Verify the API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        The verified API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if api_key is None:
        logger.warning("Missing API key in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    settings = get_settings()
    expected_key = settings.agent_api_key

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, expected_key):
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key


# Dependency for protected routes
RequireAPIKey = Annotated[str, Depends(verify_api_key)]
