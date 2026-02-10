"""Secure tool executor with validation, timeouts, and audit logging."""

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.storage.models import ExecutionStatus
from src.storage.repository import AuditRepository
from src.tools.base import Tool, ToolExecutionError, ToolValidationError
from src.tools.registry import ToolRegistry

logger = get_logger("tools.executor")


class ExecutionResultStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    OUTPUT_TRUNCATED = "output_truncated"


@dataclass
class ExecutionResult:
    """Result of a tool execution."""

    status: ExecutionResultStatus
    output: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
        }


class ToolExecutor:
    """Secure executor for tools with validation, timeouts, and logging.

    Security features:
    - Input validation against JSON Schema
    - Execution timeouts
    - Output size limits
    - Workspace directory isolation
    - Complete audit logging
    """

    def __init__(
        self,
        registry: ToolRegistry,
        repository: AuditRepository | None = None,
        settings: Settings | None = None,
    ):
        """Initialize executor.

        Args:
            registry: Tool registry for lookup
            repository: Audit repository for logging (optional)
            settings: Application settings (uses global if not provided)
        """
        self.registry = registry
        self.repository = repository
        self._settings = settings

    @property
    def settings(self) -> Settings:
        """Get settings instance."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def workspace_root(self) -> Path:
        """Get the isolated workspace root directory."""
        return self.settings.workspace_root.resolve()

    def validate_path(self, path: str | Path) -> Path:
        """Validate and resolve a path to ensure it's within workspace.

        Args:
            path: Path to validate

        Returns:
            Resolved absolute path within workspace

        Raises:
            ToolExecutionError: If path escapes workspace
        """
        workspace = self.workspace_root
        workspace.mkdir(parents=True, exist_ok=True)

        resolved = (workspace / path).resolve()

        if not str(resolved).startswith(str(workspace)):
            raise ToolExecutionError(
                message=f"Path '{path}' escapes workspace directory",
                tool_name="path_validation",
                details={"path": str(path), "workspace": str(workspace)},
            )

        return resolved

    def _truncate_output(self, output: dict[str, Any], max_size: int) -> tuple[dict[str, Any], bool]:
        """Truncate output if it exceeds max size.

        Args:
            output: Output dictionary
            max_size: Maximum size in bytes

        Returns:
            Tuple of (possibly truncated output, was_truncated)
        """
        output_str = json.dumps(output)
        if len(output_str) <= max_size:
            return output, False

        # Truncate by converting to string representation
        truncated = {
            "_truncated": True,
            "_original_size": len(output_str),
            "_max_size": max_size,
            "message": f"Output truncated from {len(output_str)} to {max_size} bytes",
        }

        # Try to include some of the original output
        if "result" in output:
            result_str = str(output["result"])
            if len(result_str) > max_size - 200:
                result_str = result_str[: max_size - 200] + "..."
            truncated["result"] = result_str

        return truncated, True

    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        request_id: str | None = None,
        decision_id: str | None = None,
    ) -> ExecutionResult:
        """Execute a tool with full security constraints.

        Args:
            tool_name: Name of tool to execute
            input_data: Input data for the tool
            request_id: Associated request ID for audit (optional)
            decision_id: Associated decision ID for audit (optional)

        Returns:
            ExecutionResult with status, output, and timing
        """
        start_time = time.perf_counter()

        # Get tool from registry
        tool = self.registry.get(tool_name)
        if tool is None:
            error_msg = f"Tool '{tool_name}' not found in registry"
            logger.warning(error_msg)
            self._log_execution(
                tool_name, input_data, ExecutionStatus.FAILED, error=error_msg,
                request_id=request_id, decision_id=decision_id,
            )
            return ExecutionResult(
                status=ExecutionResultStatus.VALIDATION_ERROR,
                error=error_msg,
            )

        # Validate input
        try:
            self.registry.validate_input(tool_name, input_data)
        except ToolValidationError as e:
            logger.warning(f"Validation failed for {tool_name}: {e.errors}")
            self._log_execution(
                tool_name, input_data, ExecutionStatus.DENIED,
                error=str(e), request_id=request_id, decision_id=decision_id,
            )
            return ExecutionResult(
                status=ExecutionResultStatus.VALIDATION_ERROR,
                error=f"Validation failed: {'; '.join(e.errors)}",
            )

        # Execute with timeout
        timeout = min(tool.definition.timeout_seconds, self.settings.max_tool_timeout)
        max_output = min(tool.definition.max_output_size, self.settings.max_output_size)

        try:
            logger.info(f"Executing tool: {tool_name}")
            output = await asyncio.wait_for(
                tool.execute(**input_data),
                timeout=timeout,
            )
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Truncate if necessary
            output, truncated = self._truncate_output(output, max_output)

            status = (
                ExecutionResultStatus.OUTPUT_TRUNCATED
                if truncated
                else ExecutionResultStatus.SUCCESS
            )

            self._log_execution(
                tool_name, input_data, ExecutionStatus.SUCCESS,
                output=json.dumps(output), duration_ms=duration_ms,
                request_id=request_id, decision_id=decision_id,
            )

            logger.info(f"Tool {tool_name} completed in {duration_ms}ms")
            return ExecutionResult(
                status=status,
                output=output,
                duration_ms=duration_ms,
                truncated=truncated,
            )

        except asyncio.TimeoutError:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"Tool execution timed out after {timeout}s"
            logger.error(f"Tool {tool_name} timed out")

            self._log_execution(
                tool_name, input_data, ExecutionStatus.TIMEOUT,
                error=error_msg, duration_ms=duration_ms,
                request_id=request_id, decision_id=decision_id,
            )

            return ExecutionResult(
                status=ExecutionResultStatus.TIMEOUT,
                error=error_msg,
                duration_ms=duration_ms,
            )

        except ToolExecutionError as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Tool {tool_name} execution error: {e.message}")

            self._log_execution(
                tool_name, input_data, ExecutionStatus.FAILED,
                error=str(e), duration_ms=duration_ms,
                request_id=request_id, decision_id=decision_id,
            )

            return ExecutionResult(
                status=ExecutionResultStatus.EXECUTION_ERROR,
                error=e.message,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            logger.exception(f"Unexpected error in tool {tool_name}")

            self._log_execution(
                tool_name, input_data, ExecutionStatus.FAILED,
                error=error_msg, duration_ms=duration_ms,
                request_id=request_id, decision_id=decision_id,
            )

            return ExecutionResult(
                status=ExecutionResultStatus.EXECUTION_ERROR,
                error=error_msg,
                duration_ms=duration_ms,
            )

    def _log_execution(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        status: ExecutionStatus,
        output: str | None = None,
        error: str | None = None,
        duration_ms: int = 0,
        request_id: str | None = None,
        decision_id: str | None = None,
    ) -> None:
        """Log execution to audit repository.

        Args:
            tool_name: Name of executed tool
            input_data: Input provided
            status: Execution status
            output: Tool output (if success)
            error: Error message (if failed)
            duration_ms: Execution duration
            request_id: Associated request ID
            decision_id: Associated decision ID
        """
        if self.repository is None:
            return

        try:
            if decision_id and request_id:
                self.repository.create_execution(
                    decision_id=decision_id,
                    request_id=request_id,
                    tool_name=tool_name,
                    tool_input=input_data,
                    status=status,
                    output=output,
                    error=error,
                    duration_ms=duration_ms,
                )
            else:
                # Log as audit entry if no decision context
                self.repository.log(
                    level="INFO" if status == ExecutionStatus.SUCCESS else "ERROR",
                    component="executor",
                    message=f"Tool execution: {tool_name} -> {status.value}",
                    context={
                        "tool_name": tool_name,
                        "input": input_data,
                        "status": status.value,
                        "duration_ms": duration_ms,
                        "error": error,
                    },
                    request_id=request_id,
                )
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
