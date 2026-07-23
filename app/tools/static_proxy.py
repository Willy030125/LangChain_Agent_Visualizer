"""Tiny static + API-proxy server used by `start_frontend` / the Run-frontend button.

Serving the frontend from a *plain* file server breaks its relative fetch() calls
(they'd hit the file server, which has no API). So this app:
  * serves workspace files (index.html/index.js/...) from its own port, AND
  * proxies ANY non-file path (e.g. /api/data, /menu, /orders) to the BACKEND.
So opening http://127.0.0.1:<frontend-port>/ renders the page AND its data loads —
regardless of which route the frontend calls — on the frontend's own port.

Config via env vars (set by launch_frontend):
  PREVIEW_DIR      = absolute workspace dir to serve
  PREVIEW_BACKEND  = base URL of the backend to proxy to (e.g. http://127.0.0.1:8090)
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

WSDIR = Path(os.environ.get("PREVIEW_DIR", ".")).resolve()
BACKEND = os.environ.get("PREVIEW_BACKEND", "http://127.0.0.1:8090")

app = FastAPI()


def _proxy(path: str, request: Request) -> Response:
    query = f"?{request.url.query}" if request.url.query else ""
    try:
        with urllib.request.urlopen(f"{BACKEND}/{path}{query}", timeout=8) as r:
            return Response(
                content=r.read(),
                media_type=r.headers.get_content_type() or "application/json",
            )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"backend proxy failed: {e}")


@app.get("/{path:path}")
def serve(path: str, request: Request):
    # Root -> index.html
    rel = path or "index.html"
    candidate = (WSDIR / rel).resolve()
    # Serve a real workspace file if it exists (and is inside the sandbox)...
    if candidate.is_file() and (WSDIR == candidate.parent or WSDIR in candidate.parents):
        return FileResponse(candidate)
    # ...otherwise treat it as an API route and proxy it to the backend.
    return _proxy(path, request)
