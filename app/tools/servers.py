"""Server-runner tools: let agents actually RUN and TEST the app they build.

A normal `run_shell("uvicorn main:app")` would block forever (the server never
exits), so these tools launch servers as *background processes* and return
immediately with a URL. A companion `http_get` tool lets the agent hit that URL
to verify it works. On app shutdown we kill everything we started.

Everything runs with cwd = workspace, and HTTP checks are restricted to
localhost, so this stays inside the sandbox.
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from langchain_core.tools import tool

from app.config import get_settings, subprocess_env

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # dir containing the `app` package

# port -> {"proc": Popen, "kind": str, "url": str}
_SERVERS: dict[int, dict] = {}


def _wait_ready(url: str, tries: int = 15) -> bool:
    for _ in range(tries):
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def _launch(port: int, args: list[str], kind: str, ready_url: str,
            cwd=None, env=None) -> str:
    if port in _SERVERS and _SERVERS[port]["proc"].poll() is None:
        _SERVERS[port]["proc"].terminate()
        time.sleep(0.5)
    proc = subprocess.Popen(
        args,
        cwd=str(cwd or get_settings().workspace_path),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env or subprocess_env(),  # run inside the agentic conda env
    )
    _SERVERS[port] = {"proc": proc, "kind": kind, "url": ready_url}
    ok = _wait_ready(ready_url)
    status = "READY" if ok else "started (not yet responding — check logs/route)"
    return f"{kind} {status} at {ready_url} (pid {proc.pid}, port {port})"


# ---- plain functions (callable by both the @tool wrappers and the API) ----
def launch_backend(port: int = 8090, module: str = "main", app_var: str = "app") -> str:
    if port == 8000:  # 8000 is the control-plane API — never let an app bind it
        port = 8090
    url = f"http://127.0.0.1:{port}"
    args = [sys.executable, "-m", "uvicorn", f"{module}:{app_var}",
            "--host", "127.0.0.1", "--port", str(port)]
    return _launch(port, args, "backend", url + "/docs")


def launch_frontend(port: int = 8091) -> str:
    """Serve the workspace on <port> AND proxy /api to the running backend, so the
    rendered frontend loads live data on its OWN port."""
    ws = get_settings().workspace_path
    backend = next(
        (f"http://127.0.0.1:{s['port']}" for s in running_servers()
         if s["kind"] == "backend" and s["running"]),
        "http://127.0.0.1:8090",
    )
    env = subprocess_env()
    env["PREVIEW_DIR"] = str(ws)
    env["PREVIEW_BACKEND"] = backend
    args = [sys.executable, "-m", "uvicorn", "app.tools.static_proxy:app",
            "--host", "127.0.0.1", "--port", str(port)]
    # cwd = project root so `app.tools.static_proxy` imports; static server still
    # serves the workspace via PREVIEW_DIR.
    return _launch(port, args, "frontend", f"http://127.0.0.1:{port}/index.html",
                   cwd=_PROJECT_ROOT, env=env)


@tool
def start_backend(port: int = 8090, module: str = "main", app_var: str = "app") -> str:
    """Start the FastAPI BACKEND as a background server so it can be tested.

    Runs `uvicorn <module>:<app_var>` on 127.0.0.1:<port> from the workspace.
    Use this AFTER you have created the backend file. Returns the base URL. The
    server keeps running in the background; call http_get to test its endpoints.

    Args:
        port: Port to bind (default 8090). Use 8090-8099.
        module: Python module (filename without .py) holding the app. Default "main".
        app_var: The FastAPI instance variable name in that module. Default "app".

    Returns:
        A status line with the URL, e.g. "backend READY at http://127.0.0.1:8090".
    """
    return launch_backend(port, module, app_var)


@tool
def start_frontend(port: int = 8091) -> str:
    """Start a static FRONTEND server (python -m http.server) for the workspace.

    Serves the workspace folder (so index.html / index.js are reachable) on
    127.0.0.1:<port>. Use this AFTER creating index.html. Returns the URL to open.

    Args:
        port: Port to bind (default 8091). Use 8091-8099.

    Returns:
        A status line with the URL, e.g. "frontend READY at http://127.0.0.1:8091".
    """
    return launch_frontend(port)


@tool
def http_get(url: str) -> str:
    """Make an HTTP GET request to a LOCAL url and return status + body.

    Use this to TEST a running server — e.g. after start_backend, call
    http_get("http://127.0.0.1:8090/api/hello") and check the JSON is correct.
    Only localhost URLs are allowed.

    Args:
        url: Full URL, must be http://127.0.0.1:... or http://localhost:...

    Returns:
        "HTTP <status>\\n<body>" (body truncated), or an ERROR line if it failed.
    """
    if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
        return "ERROR: only localhost URLs are allowed."
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            body = r.read().decode("utf-8", errors="replace")
            return f"HTTP {r.status}\n{body[:1500]}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


@tool
def http_post(url: str, json_body: str = "{}") -> str:
    """Make an HTTP POST request to a LOCAL url with a JSON body, and return
    status + body. Use this to test POST endpoints (http_get only does GET).

    Args:
        url: Full URL, must be http://127.0.0.1:... or http://localhost:...
        json_body: JSON string to send as the request body, e.g. '{"numbers":"2,3,4"}'.

    Returns:
        "HTTP <status>\\n<body>" (truncated), or an ERROR line.
    """
    if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
        return "ERROR: only localhost URLs are allowed."
    try:
        req = urllib.request.Request(
            url, data=json_body.encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            return f"HTTP {r.status}\n{r.read().decode('utf-8', errors='replace')[:1500]}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


@tool
def list_servers() -> str:
    """List background servers this agent has started (backend/frontend) and
    whether each is still running. Useful before starting a new one."""
    if not _SERVERS:
        return "No servers started yet."
    lines = []
    for port, info in _SERVERS.items():
        alive = info["proc"].poll() is None
        lines.append(f"- port {port} [{info['kind']}] {'running' if alive else 'stopped'} -> {info['url']}")
    return "\n".join(lines)


@tool
def stop_server(port: int) -> str:
    """Stop a background server previously started on the given port.

    Args:
        port: The port whose server should be stopped.
    """
    info = _SERVERS.get(port)
    if not info:
        return f"No server on port {port}."
    info["proc"].terminate()
    return f"Stopped {info['kind']} on port {port}."


def stop_all_servers() -> None:
    """Kill every background server (called on app shutdown)."""
    for info in _SERVERS.values():
        try:
            info["proc"].terminate()
        except Exception:
            pass
    _SERVERS.clear()


def running_servers() -> list[dict]:
    """Snapshot for the API/UI: list of {port, kind, url, running}."""
    out = []
    for port, info in _SERVERS.items():
        out.append({
            "port": port, "kind": info["kind"], "url": info["url"],
            "running": info["proc"].poll() is None,
        })
    return out


SERVER_TOOLS = [start_backend, start_frontend, http_get, http_post, list_servers, stop_server]
