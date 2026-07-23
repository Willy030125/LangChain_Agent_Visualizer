"""Execution tools: shell, Python interpreter, Node interpreter.

These give the agent its "run commands / tests" and "deep-debug" powers. Every
process:
  * runs with cwd = workspace root (so relative paths match the FS tools),
  * has a wall-clock timeout (no runaway loops), and
  * returns a single, uniform "exit/stdout/stderr" string the LLM can read and
    reason about — which is exactly what makes iterative debugging work.

SECURITY: arbitrary code execution is powerful. It is intentionally scoped to
the workspace and gated by ALLOW_SHELL for the raw shell tool. Run this backend
only on machines/workspaces you trust. (No internal policy catalog was provided
to cite; the implemented controls are: workspace-cwd confinement, per-call
timeout, and a config flag to disable raw shell.)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from app.config import get_settings, subprocess_env


def _run(cmd: list[str], stdin_file: Path | None = None) -> str:
    """Run a subprocess in the sandbox and format the result for the LLM."""
    s = get_settings()
    try:
        proc = subprocess.run(
            cmd,
            cwd=s.workspace_path,
            capture_output=True,
            text=True,
            timeout=s.exec_timeout,
            env=subprocess_env(),  # use the agentic conda env (fastapi/uvicorn on PATH)
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {s.exec_timeout}s: {' '.join(cmd)}"
    except FileNotFoundError:
        return f"ERROR: executable not found: {cmd[0]!r} (is it installed / on PATH?)"

    out = proc.stdout.strip()
    err = proc.stderr.strip()
    # Uniform, compact report. The exit code first so the model can branch on it.
    parts = [f"exit_code: {proc.returncode}"]
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    return "\n".join(parts)


def _run_source(code: str, suffix: str, runner: list[str]) -> str:
    """Write `code` to a temp file in the workspace and execute it with `runner`."""
    root = get_settings().workspace_path
    # Keep the temp file inside the sandbox so imports of workspace files work.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, dir=root, delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        temp_path = Path(fh.name)
    try:
        return _run([*runner, str(temp_path)])
    finally:
        temp_path.unlink(missing_ok=True)


@tool
def run_shell(command: str) -> str:
    """Run a shell command inside the workspace and return exit code, stdout,
    and stderr. Use for running tests, installing deps, git, etc.

    Args:
        command: The command line to execute, e.g. "pytest -q" or "npm test".
    """
    if not get_settings().allow_shell:
        return "ERROR: raw shell execution is disabled (ALLOW_SHELL=false)."
    # shell=True lets the model use pipes/&&; scoped to the sandbox cwd.
    s = get_settings()
    try:
        proc = subprocess.run(
            command,
            cwd=s.workspace_path,
            shell=True,
            capture_output=True,
            text=True,
            timeout=s.exec_timeout,
            env=subprocess_env(),  # agentic conda env on PATH (python/uvicorn/pip)
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {s.exec_timeout}s: {command}"
    out, err = proc.stdout.strip(), proc.stderr.strip()
    parts = [f"exit_code: {proc.returncode}"]
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    return "\n".join(parts)


@tool
def run_python(code: str) -> str:
    """Run a snippet of Python code and return exit code + stdout + stderr.

    Use this to TEST your work: import a file you wrote, call a function, print a
    result, or check something parses. Runs in the workspace, so `import main`
    works if you created main.py there. Do NOT start blocking servers here
    (use start_backend for that) — this waits for the code to finish.

    Args:
        code: Python source to execute.

    Returns:
        "exit_code: N" plus any stdout/stderr.

    Example:
        run_python(code="import main; print('import ok')")
        run_python(code="from calc import add; print(add(2, 3))")
    """
    return _run_source(code, suffix=".py", runner=[sys.executable])


@tool
def run_node(code: str) -> str:
    """Run a snippet of JavaScript with Node.js and return exit code + output.

    Use this to test/debug JS logic (e.g. functions in index.js). Runs in the
    workspace directory.

    Args:
        code: JavaScript source to execute with Node.

    Returns:
        "exit_code: N" plus any stdout/stderr.

    Example:
        run_node(code="const {add}=require('./calc.js'); console.log(add(2,3));")
    """
    return _run_source(code, suffix=".js", runner=["node"])


EXEC_TOOLS = [run_shell, run_python, run_node]
