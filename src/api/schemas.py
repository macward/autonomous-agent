"""API request and response schemas."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    """Request body for POST /agent/run."""

    input: str = Field(..., min_length=1, max_length=10000, description="User input text")

    model_config = {"json_schema_extra": {"examples": [{"input": "List the files in the workspace"}]}}


class AgentRunStatus(str, Enum):
    """Status of an agent run."""

    SUCCESS = "success"
    ERROR = "error"
    TOOL_ERROR = "tool_error"


class ToolExecutionInfo(BaseModel):
    """Information about a tool execution."""

    tool_name: str
    status: str
    duration_ms: int | None = None
    output: dict[str, Any] | None = None
    error: str | None = None


class AgentRunResponse(BaseModel):
    """Response body for POST /agent/run."""

    request_id: str = Field(..., description="Unique request ID for audit")
    status: AgentRunStatus = Field(..., description="Overall status of the run")
    output: str | None = Field(default=None, description="Agent's final response")
    error: str | None = Field(default=None, description="Error message if failed")
    tool_executions: list[ToolExecutionInfo] = Field(
        default_factory=list, description="Tools executed during this run"
    )
    created_at: datetime = Field(..., description="Request timestamp")
    duration_ms: int = Field(..., description="Total duration in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "success",
                    "output": "The workspace contains 3 files: file1.txt, file2.txt, and config.json",
                    "tool_executions": [
                        {"tool_name": "list_dir", "status": "success", "duration_ms": 5}
                    ],
                    "created_at": "2024-01-15T10:30:00Z",
                    "duration_ms": 1234,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(default="healthy")
    version: str = Field(default="0.1.0")
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Additional details")
    request_id: str | None = Field(default=None, description="Request ID if available")
