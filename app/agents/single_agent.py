"""Stage 2 — a single ReAct-style coding agent via create_agent.

`create_agent` (LangChain 1.x) wires up the classic agent loop for you:
    model -> (maybe tool calls) -> run tools -> feed results back -> repeat
until the model answers with no tool call. Under the hood it compiles a
LangGraph graph, so it already supports streaming, checkpointer memory, etc.

This is the simplest useful unit: ONE agent that can read/write files and run
code. Stage 3 composes several of these ideas into a multi-agent orchestrator.
"""
from __future__ import annotations

from langchain.agents import create_agent

from app.config import get_model
from app.tools import ALL_TOOLS

CODER_SYSTEM_PROMPT = """You are a coding assistant in a sandboxed workspace. You \
build WHATEVER the user asks — a script, an API, or a full-stack web app (e.g. a \
demo for a business idea or product) — by CALLING TOOLS. Actually write files and \
run them; do not just describe code.

Loop for every task:
1) PLAN: one sentence — the files you'll create.
2) WRITE: write_file(path, content) with the COMPLETE file. One call per file.
3) VERIFY: prove it works, then move on. Do not over-iterate.
4) FIX: if a tool shows a non-zero exit_code / ERROR, read it, fix once, re-run.
5) REPORT: 1-2 sentence summary of files + how you verified.

FULL-STACK WEB APP CONTRACT (use this whenever a web app / UI is requested):
- Backend: FastAPI in `main.py` exposing `GET /api/data` that returns the app's
  data as JSON (use realistic demo data for the topic, e.g. products, tasks).
  Verify: start_backend(port=8090) then http_get("http://127.0.0.1:8090/api/data")
  and expect HTTP 200 with JSON.
- Frontend: `index.html` (MUST contain <script src="index.js"></script>) plus
  `index.js` that does fetch('/api/data'), then renders the result into a visible
  element (list/cards/table). Use the EXACT field names the backend returns.

Rules:
- Prefer SIMPLE ITERATIVE code (loops); never deep recursion.
- Never claim success without a passing run (exit_code 0 or HTTP 200).
- Keep it minimal but working; don't overthink.
"""


def build_coder_agent(checkpointer=None):
    """Build the single-agent coder.

    Args:
        checkpointer: optional LangGraph checkpointer for per-thread memory
            (pass one so a `thread_id` keeps conversation history across calls).
    """
    return create_agent(
        model=get_model(),
        tools=ALL_TOOLS,
        system_prompt=CODER_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
