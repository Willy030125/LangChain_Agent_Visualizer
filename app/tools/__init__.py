"""Stage 1 — Tools. Grouped so agents can pick a subset."""
from app.tools.filesystem import FS_TOOLS
from app.tools.execution import EXEC_TOOLS
from app.tools.servers import SERVER_TOOLS

# Convenience bundle: everything a coder/debugger might need.
ALL_TOOLS = [*FS_TOOLS, *EXEC_TOOLS, *SERVER_TOOLS]

__all__ = ["FS_TOOLS", "EXEC_TOOLS", "SERVER_TOOLS", "ALL_TOOLS"]
