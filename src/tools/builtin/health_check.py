"""Health check tool - returns system status."""

import platform
from datetime import UTC, datetime
from typing import Any

from src.tools.base import Tool, ToolDefinition


class HealthCheckTool(Tool):
    """Tool that returns the agent's health status.

    This tool requires no permissions and is safe to execute at any time.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="health_check",
            description="Check the agent's health status and system information",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            permissions=[],  # No special permissions needed
            timeout_seconds=5,
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute health check.

        Returns:
            Dictionary with health status and system info
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "system": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "architecture": platform.machine(),
            },
        }
