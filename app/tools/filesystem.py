"""Sandboxed filesystem tools.

Every path the agent gives is resolved and checked to live *inside* the
workspace root. This is the single most important guardrail: it stops a
hallucinated path like "../../Windows/System32" from ever resolving outside
the sandbox. (Guardrail note: no internal policy catalog was provided to this
session, so no policy number is cited — the control implemented here is
workspace path confinement + explicit allow-listing of operations.)

A LangChain tool is just a Python function + a docstring. The @tool decorator
turns the function signature into a JSON schema the model sees, and the
docstring is the tool's "instructions" to the model — so write it for the LLM.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from app.config import get_settings


class SandboxError(ValueError):
    """Raised when a path escapes the workspace sandbox."""


def _resolve(relative_path: str) -> Path:
    """Resolve a user/agent-supplied path against the workspace, and refuse
    anything that escapes it."""
    root = get_settings().workspace_path
    # Treat every incoming path as relative to the sandbox root.
    candidate = (root / relative_path).resolve()
    if root != candidate and root not in candidate.parents:
        raise SandboxError(
            f"Path {relative_path!r} escapes the workspace sandbox."
        )
    return candidate


@tool
def read_file(path: str) -> str:
    """Read and return the full text contents of a file in the workspace.

    Args:
        path: File path relative to the workspace root, e.g. "src/main.py".
    """
    target = _resolve(path)
    if not target.is_file():
        return f"ERROR: file not found: {path}"
    return target.read_text(encoding="utf-8", errors="replace")


@tool
def write_file(path: str, content: str) -> str:
    """Create a new file (or overwrite an existing one) with the given content.

    This is the MAIN tool for producing code. Pass the COMPLETE file content —
    it replaces the whole file. Parent folders are created automatically.

    Args:
        path: Destination path relative to the workspace, e.g. "main.py" or
            "static/index.html".
        content: The full text/code to write into the file.

    Returns:
        "OK: wrote N chars to <path>" on success.

    Example:
        write_file(path="main.py", content="from fastapi import FastAPI\\napp = FastAPI()")
    """
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {path}"


@tool
def edit_file(path: str, find: str, replace: str) -> str:
    """Replace the first exact occurrence of `find` with `replace` in a file.
    Use this for small, surgical edits instead of rewriting the whole file.

    Args:
        path: File to edit, relative to the workspace root.
        find: Exact substring to search for (must be unique enough to match once).
        replace: Text to substitute in.
    """
    target = _resolve(path)
    if not target.is_file():
        return f"ERROR: file not found: {path}"
    text = target.read_text(encoding="utf-8", errors="replace")
    if find not in text:
        return f"ERROR: `find` text not found in {path}; no change made."
    target.write_text(text.replace(find, replace, 1), encoding="utf-8")
    return f"OK: edited {path}"


@tool
def list_dir(path: str = ".") -> str:
    """List files and folders at a path inside the workspace (non-recursive).

    Args:
        path: Directory relative to the workspace root. Defaults to root.
    """
    target = _resolve(path)
    if not target.is_dir():
        return f"ERROR: not a directory: {path}"
    entries = sorted(
        f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()
    )
    return "\n".join(entries) if entries else "(empty)"


@tool
def make_dir(path: str) -> str:
    """Create a directory (and any missing parents) inside the workspace.

    Args:
        path: Directory path relative to the workspace root.
    """
    target = _resolve(path)
    target.mkdir(parents=True, exist_ok=True)
    return f"OK: created {path}"


FS_TOOLS = [read_file, write_file, edit_file, list_dir, make_dir]
