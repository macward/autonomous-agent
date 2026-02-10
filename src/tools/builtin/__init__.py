# Built-in MVP tools

from src.tools.builtin.health_check import HealthCheckTool
from src.tools.builtin.list_dir import ListDirTool
from src.tools.builtin.read_file import ReadFileTool

__all__ = [
    "HealthCheckTool",
    "ListDirTool",
    "ReadFileTool",
]


def register_builtin_tools(registry, executor) -> None:
    """Register all built-in tools in the registry.

    Args:
        registry: Tool registry to register tools in
        executor: Tool executor for path validation
    """
    registry.register(HealthCheckTool())
    registry.register(ListDirTool(executor))
    registry.register(ReadFileTool(executor))
