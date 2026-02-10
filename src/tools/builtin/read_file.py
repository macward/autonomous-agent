"""Read file tool - reads files with security constraints."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tools.base import Tool, ToolDefinition, ToolExecutionError, ToolPermission

if TYPE_CHECKING:
    from src.tools.executor import ToolExecutor


class ReadFileTool(Tool):
    """Tool that reads file contents within the workspace.

    Security constraints:
    - Only reads files within the workspace directory
    - Path traversal attempts are blocked
    - File size is limited
    - Binary files are rejected
    """

    MAX_FILE_SIZE = 100 * 1024  # 100 KB
    BINARY_EXTENSIONS = {".exe", ".bin", ".dll", ".so", ".dylib", ".zip", ".tar", ".gz", ".jpg", ".png", ".gif", ".pdf"}

    def __init__(self, executor: "ToolExecutor"):
        """Initialize with executor for path validation.

        Args:
            executor: Tool executor instance for path validation
        """
        self._executor = executor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read the contents of a text file within the workspace",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file within workspace",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: all)",
                        "minimum": 1,
                        "maximum": 1000,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            permissions=[ToolPermission.READ],
            timeout_seconds=10,
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Read file contents.

        Args:
            path: Relative path to file within workspace
            max_lines: Optional maximum lines to read

        Returns:
            Dictionary with file contents

        Raises:
            ToolExecutionError: If file is invalid or not readable
        """
        path_str = kwargs["path"]
        max_lines = kwargs.get("max_lines")

        # Validate path is within workspace
        try:
            validated_path = self._executor.validate_path(path_str)
        except ToolExecutionError:
            raise

        if not validated_path.exists():
            raise ToolExecutionError(
                message=f"File does not exist: {path_str}",
                tool_name="read_file",
            )

        if not validated_path.is_file():
            raise ToolExecutionError(
                message=f"Path is not a file: {path_str}",
                tool_name="read_file",
            )

        # Check for binary files
        if validated_path.suffix.lower() in self.BINARY_EXTENSIONS:
            raise ToolExecutionError(
                message=f"Cannot read binary file: {path_str}",
                tool_name="read_file",
            )

        # Check file size
        try:
            file_size = validated_path.stat().st_size
        except OSError as e:
            raise ToolExecutionError(
                message=f"Cannot access file: {e}",
                tool_name="read_file",
            )

        if file_size > self.MAX_FILE_SIZE:
            raise ToolExecutionError(
                message=f"File too large: {file_size} bytes (max: {self.MAX_FILE_SIZE})",
                tool_name="read_file",
            )

        # Read file
        try:
            content = validated_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolExecutionError(
                message=f"File is not valid UTF-8 text: {path_str}",
                tool_name="read_file",
            )
        except PermissionError:
            raise ToolExecutionError(
                message=f"Permission denied: {path_str}",
                tool_name="read_file",
            )

        # Apply line limit if specified
        lines = content.splitlines()
        truncated = False
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True
            content = "\n".join(lines)

        return {
            "path": path_str,
            "content": content,
            "size": file_size,
            "lines": len(lines),
            "truncated": truncated,
        }
