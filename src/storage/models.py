"""Database models for audit and persistence."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RequestStatus(str, Enum):
    """Status of an agent request."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DENIED = "denied"


class AgentRequest(BaseModel):
    """Record of an incoming agent request."""

    id: str = Field(..., description="Unique request ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    input_text: str = Field(..., description="Original user input")
    status: RequestStatus = Field(default=RequestStatus.PENDING)
    completed_at: datetime | None = Field(default=None)
    final_output: str | None = Field(default=None)
    error: str | None = Field(default=None)


class AgentDecision(BaseModel):
    """Record of an LLM decision within a request."""

    id: str = Field(..., description="Unique decision ID")
    request_id: str = Field(..., description="Parent request ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reasoning: str = Field(..., description="LLM reasoning/thought process")
    selected_tool: str | None = Field(default=None, description="Tool chosen by LLM")
    tool_input: dict[str, Any] | None = Field(default=None, description="Input for the tool")


class ToolExecution(BaseModel):
    """Record of a tool execution."""

    id: str = Field(..., description="Unique execution ID")
    decision_id: str = Field(..., description="Parent decision ID")
    request_id: str = Field(..., description="Root request ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_name: str = Field(..., description="Name of executed tool")
    tool_input: dict[str, Any] = Field(..., description="Input provided to tool")
    status: ExecutionStatus = Field(..., description="Execution result status")
    output: str | None = Field(default=None, description="Tool output")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_ms: int | None = Field(default=None, description="Execution duration in ms")


class AuditLog(BaseModel):
    """General audit log entry."""

    id: str = Field(..., description="Unique log ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    level: str = Field(..., description="Log level (INFO, WARNING, ERROR)")
    component: str = Field(..., description="Component that generated the log")
    message: str = Field(..., description="Log message")
    context: dict[str, Any] | None = Field(default=None, description="Additional context")
    request_id: str | None = Field(default=None, description="Associated request ID")
