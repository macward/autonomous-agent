"""Application configuration with security defaults."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    llm_provider: str = Field(default="anthropic", description="LLM provider")
    llm_model: str = Field(default="claude-sonnet-4-20250514", description="LLM model")
    llm_api_key: str = Field(..., description="LLM API key")

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


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
