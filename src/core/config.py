"""Application configuration with security defaults."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from src.tools.base import ToolPermission

logger = logging.getLogger("core.config")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Security
    agent_api_key: str = Field(..., description="API key for authentication")

    # Server
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # LLM Configuration
    llm_provider: str = Field(
        default="anthropic",
        description="LLM provider: 'anthropic' or 'groq'",
    )
    llm_model: str | None = Field(
        default=None,
        description="LLM model (defaults based on provider)",
    )
    llm_api_key: str = Field(..., description="LLM API key")

    @property
    def effective_llm_model(self) -> str:
        """Get the effective LLM model based on provider."""
        if self.llm_model:
            return self.llm_model

        # Default models per provider
        defaults = {
            "anthropic": "claude-sonnet-4-20250514",
            "groq": "llama-3.3-70b-versatile",
        }
        return defaults.get(self.llm_provider, defaults["anthropic"])

    # Storage
    database_url: str = Field(
        default="sqlite:///./data/agent.db", description="Database connection URL"
    )

    # Security constraints
    workspace_root: Path = Field(
        default=Path("./workspace"), description="Isolated workspace directory"
    )
    max_tool_timeout: int = Field(
        default=30, ge=1, le=300, description="Maximum tool execution timeout in seconds"
    )
    max_output_size: int = Field(
        default=10000, ge=100, le=100000, description="Maximum output size in bytes"
    )

    # Permission enforcement - comma-separated list of blocked permissions
    # Default: block NETWORK and EXECUTE for security
    blocked_permissions_str: str = Field(
        default="network,execute",
        alias="BLOCKED_PERMISSIONS",
        description="Comma-separated list of blocked permissions (read, write, execute, network)",
    )

    # Rate limiting
    rate_limit_requests: int = Field(
        default=10, ge=1, le=1000, description="Max requests per minute"
    )
    rate_limit_enabled: bool = Field(
        default=True, description="Enable rate limiting"
    )

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed CORS origins",
    )

    # Request limits
    max_request_size: int = Field(
        default=1048576,
        ge=1024,
        le=10485760,
        description="Maximum request body size in bytes (1MB default)",
    )

    @property
    def blocked_permissions(self) -> set["ToolPermission"]:
        """Get blocked permissions as a set of ToolPermission enums."""
        from src.tools.base import ToolPermission

        if not self.blocked_permissions_str:
            return set()

        valid_values = {p.value for p in ToolPermission}
        perms = set()
        for perm_str in self.blocked_permissions_str.split(","):
            perm_str = perm_str.strip().lower()
            if perm_str:
                try:
                    perms.add(ToolPermission(perm_str))
                except ValueError:
                    logger.warning(
                        f"Invalid permission '{perm_str}' in BLOCKED_PERMISSIONS. "
                        f"Valid values: {', '.join(sorted(valid_values))}"
                    )
        return perms

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
