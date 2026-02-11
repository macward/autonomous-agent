"""List directory tool - lists files in allowed directories."""

from typing import TYPE_CHECKING, Any

from src.tools.base import Tool, ToolDefinition, ToolExecutionError, ToolPermission

if TYPE_CHECKING:
    from src.tools.executor import ToolExecutor


class ListDirTool(Tool):
    """Tool that lists files in a directory within the workspace.

    Security constraints:
    - Only lists files within the workspace directory
    - Path traversal attempts are blocked
    - Limits number of entries returned
    """

    MAX_ENTRIES = 100

    def __init__(self, executor: "ToolExecutor"):
        """Initialize with executor for path validation.

        Args:
            executor: Tool executor instance for path validation
        """
        self._executor = executor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_dir",
            description="List files and directories in a path within the workspace",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within workspace (default: root)",
                        "default": ".",
                    },
                },
                "additionalProperties": False,
            },
            permissions=[ToolPermission.READ],
            timeout_seconds=10,
        )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """List directory contents.

        Args:
            path: Relative path within workspace

        Returns:
            Dictionary with list of entries

        Raises:
            ToolExecutionError: If path is invalid or not accessible
        """
        path_str = kwargs.get("path", ".")

        # Validate path is within workspace
        try:
            validated_path = self._executor.validate_path(path_str)
        except ToolExecutionError:
            raise

        if not validated_path.exists():
            raise ToolExecutionError(
                message=f"Path does not exist: {path_str}",
                tool_name="list_dir",
            )

        if not validated_path.is_dir():
            raise ToolExecutionError(
                message=f"Path is not a directory: {path_str}",
                tool_name="list_dir",
            )

        entries = []
        try:
            for i, entry in enumerate(sorted(validated_path.iterdir())):
                if i >= self.MAX_ENTRIES:
                    break

                entry_info = {
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                }

                if entry.is_file():
                    try:
                        entry_info["size"] = entry.stat().st_size
                    except OSError:
                        entry_info["size"] = None

                entries.append(entry_info)

        except PermissionError as e:
            raise ToolExecutionError(
                message=f"Permission denied: {path_str}",
                tool_name="list_dir",
            ) from e

        return {
            "path": path_str,
            "entries": entries,
            "count": len(entries),
            "truncated": len(list(validated_path.iterdir())) > self.MAX_ENTRIES,
        }
