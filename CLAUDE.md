# CLAUDE.md — guidance for Claude Code in this repo

Backend "Agentic Coder Orchestrator": LangChain 1.x + LangGraph + Deep Agents,
served by FastAPI with a live-flow Web UI, using local Ollama models. Agents build
and run full-stack demos inside `./workspace`.

## Run / environment (read first)
- **Conda env `agentic` (Python 3.12)** — `deepagents` needs Python ≥ 3.11. The
  system Python is 3.10 and will NOT work.
- Interpreter: `C:\Users\Admin\miniconda3\envs\agentic\python.exe`.
- Start (serves API + UI on one port, no npm/build):
  `conda run -n agentic uvicorn app.api.main:app --host 127.0.0.1 --port 8000`
  or `conda run -n agentic python run.py`.
- **Ollama host** at `http://127.0.0.1:11434` (or can be external local server).
- Each shell is fresh; use `conda run -n agentic ...` or `conda activate agentic`.
- On Windows use the env python by full path in PowerShell; `conda create` uses
  `-c conda-forge --override-channels` to avoid Anaconda ToS prompts.

## Layout
- `app/config.py` — settings (`pydantic-settings`), `get_model(**overrides)` factory,
  `subprocess_env()`.
- `app/tools/` — `filesystem.py` (sandboxed FS), `execution.py` (run_python/node/shell),
  `servers.py` (start_backend/start_frontend/http_get/http_post + `static_proxy.py`).
- `app/agents/` — `single_agent.py`, `parallel.py` (LangGraph fan-out),
  `orchestrator.py` (deepagents).
- `app/api/main.py` — FastAPI app, SSE stream, topology/servers/run/preview endpoints.
- `app/web/index.html` — the single-page Web UI (vanilla JS; served static).
- `run.py`, `smoke_test.py`, `requirements.txt`, `.env(.example)`, `workspace/`.

## Model mapping (per role, in `config.py`)
orchestrator=`qwen3:8b` (reasoning off) · planner=`qwen3.5:latest` (reasoning off) ·
coder/debugger=`qwen3.5:latest` · reviewer/reporter=`qwen3.5:4b` · single=`qwen3:8b`.
Pass a `ChatOllama` **instance** (via `get_model`), never an `"ollama:..."` string
(the server is remote; a string would default to localhost).

## Critical conventions & gotchas (do NOT regress these)
1. **Subprocess conda env** — every child process (run_shell, run_python,
   uvicorn, http.server) MUST pass `env=subprocess_env()`, or it uses system
   Python and fails with "No module named fastapi/uvicorn".
2. **Reasoning streaming** — `ollama_reasoning=True` (`think:true`). Streamed
   reasoning arrives in `chunk.additional_kwargs['reasoning_content']`; content in
   `chunk.content`. The stream emits it as `thinking` events (coalesced).
3. **SSE framing is CRLF** — `sse-starlette` separates events with `\r\n\r\n`. Any
   browser/JS parser MUST normalize `\r\n`→`\n` before splitting on `\n\n`.
4. **DeepAgents = LLM-driven, kept intentionally** (user chose this over a
   deterministic pipeline). Small models won't sustain the delegation loop unless
   the orchestrator is centered on the **`write_todos` checklist**: prompt makes
   step 1 = write the 4-item checklist, step 2 = "work it top-to-bottom, forbidden
   to stop while any item is unchecked." This is what makes it continue past the
   planner. Orchestrator runs with `reasoning=False` and `tools=[]` (declutter) so
   it delegates instead of ego-writing code/narrating a fake `task(...)` as text.
5. **Deep Agents filesystem** — uses deepagents' built-in file tools (NOT ours) +
   `CompositeBackend(default=FilesystemBackend(workspace), routes={"/workspace/":…})`
   so files persist to disk. Do NOT also pass our `write_file` (duplicate tool
   names → path-convention clashes). Planner is LOCKED to `main.py` + `/api/data` +
   `index.html`/`index.js` to prevent drift.
6. **Full-stack contract** — backend `GET /api/data` → JSON; frontend
   `fetch('/api/data')`; SAME field names both sides. Enforced in every mode's prompt.
7. **Port 8000 is the control API** — `launch_backend` forces 8000→8090. Never let
   an agent bind 8000.
8. **Bounds** — `recursion_limit=35` (main.py `_config`), `num_predict=6144`,
   `/chat/stream` always emits `done` in `finally`. Keep these; they prevent
   token-burning loops.
9. **Preview** — `static_proxy.py` serves the workspace on the frontend port AND
   proxies any non-file path to the running backend, so a rendered frontend shows
   live data regardless of route.

## Testing / verifying a change
- Prefer hitting the real endpoints (same path the UI uses). Example: POST
  `/chat/stream` with `{"message":...,"mode":"orchestrator"}` and parse the SSE.
- For long agent runs, run a client in the background and **monitor** an event log
  (grep for `DELEGATE|write_file|DONE|ERROR`). If files stop changing for minutes
  while the run continues, it's looping — stop it (TaskStop) to save tokens.
- Verify a built web app: `POST /api/run/backend`, `POST /api/run/frontend`, then
  `GET http://127.0.0.1:8091/index.html` and `GET http://127.0.0.1:8091/api/data`.
- Restart the server after editing anything under `app/` (prompts/models/tools are
  bound at startup). `app/web/index.html` is static — just hard-refresh the browser.

## Docs
- Use Context7 MCP for current LangChain / LangGraph / deepagents / Ollama docs
  before changing agent construction or model/streaming behavior — the APIs move.
