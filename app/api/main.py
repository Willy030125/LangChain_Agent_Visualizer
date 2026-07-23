"""Stage 4 — FastAPI application (+ Stage 5: web UI).

Endpoints:
  GET  /            -> the web UI (app/web/index.html)
  GET  /health      -> readiness + which model/workspace is active
  GET  /api/files   -> list files in the workspace sandbox
  GET  /api/file    -> read one workspace file's contents
  POST /chat        -> run the agent to completion, return final reply (JSON)
  POST /chat/stream -> stream tokens + tool activity as Server-Sent Events (SSE)

Memory: we build each agent ONCE at startup with a shared MemorySaver
checkpointer. LangGraph then keys conversation state by `thread_id`, so two
requests with the same thread_id continue the same conversation.
"""
from __future__ import annotations

import json
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from sse_starlette.sse import EventSourceResponse

from app.agents import (
    build_coder_agent,
    build_orchestrator,
    build_parallel_team,
)
from app.agents.orchestrator import orchestrator_topology
from app.agents.parallel import parallel_topology
from app.config import get_settings
from app.schemas import AgentMode, ChatRequest, ChatResponse, HealthResponse
from app.tools.filesystem import SandboxError, _resolve
from app.tools.servers import (
    launch_backend,
    launch_frontend,
    running_servers,
    stop_all_servers,
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Agents are created at startup and reused across requests (they're stateless
# except for the checkpointer, which holds per-thread memory).
AGENTS: dict[AgentMode, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = MemorySaver()
    AGENTS[AgentMode.single] = build_coder_agent(checkpointer=checkpointer)
    AGENTS[AgentMode.parallel] = build_parallel_team(checkpointer=checkpointer)
    AGENTS[AgentMode.orchestrator] = build_orchestrator(checkpointer=checkpointer)
    yield
    AGENTS.clear()
    stop_all_servers()  # kill any backend/frontend the agents launched


app = FastAPI(title="Agentic Coder Orchestrator", version="1.0.0", lifespan=lifespan)

# Allow the previewed frontend (and dev tools) to call the API cross-origin.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Serve the workspace statically so the WebUI can RENDER a built frontend
# (index.html/index.js) in an iframe or a new tab, at /preview/.
app.mount(
    "/preview",
    StaticFiles(directory=str(get_settings().workspace_path), html=True),
    name="preview",
)


def _config(thread_id: str) -> dict:
    """LangGraph run config: memory thread + a generous recursion limit.

    `recursion_limit` caps how many node steps a run may take. Small local models
    take more back-and-forth steps (extra tool calls, re-tries) than big models,
    so the default (25) can be hit mid-build. We raise it so DeepAgents runs with
    subagents can finish.
    """
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 35}


def _inputs(message: str) -> dict:
    return {"messages": [HumanMessage(content=message)]}


def _as_text(content) -> str:
    """Normalize message content (str OR list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content-blocks form
        parts = [
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        ]
        return "".join(parts)
    return str(content or "")


def _final_text(messages: list) -> str:
    """Extract the agent's final answer robustly.

    Small local models (qwen3 etc.) often narrate intent BEFORE calling tools
    and then end the loop with an EMPTY message. Naively taking the last, or the
    last non-empty, assistant turn can return stale pre-tool preamble. So:

      1. Prefer assistant prose that appears AFTER the last tool call — that's a
         genuine post-work summary.
      2. If the model gave none, synthesize a short summary from the tool results
         so the caller still learns what happened.
      3. If no tools ran at all, return the last non-empty assistant turn (plain chat).
    """
    last_tool_idx = max(
        (i for i, m in enumerate(messages) if isinstance(m, ToolMessage)),
        default=-1,
    )

    if last_tool_idx >= 0:
        # (1) real summary produced after tools finished
        for msg in messages[last_tool_idx + 1 :]:
            if isinstance(msg, AIMessage):
                text = _as_text(msg.content).strip()
                if text:
                    return text
        # (2) no summary — report what the tools did
        tool_results = [
            f"- {getattr(m, 'name', 'tool')}: {_as_text(m.content)}"
            for m in messages
            if isinstance(m, ToolMessage)
        ]
        return (
            "(model gave no closing summary) Completed "
            f"{len(tool_results)} tool step(s):\n" + "\n".join(tool_results)
        )

    # (3) plain conversation, no tools
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = _as_text(msg.content).strip()
            if text:
                return text
    return "(agent completed without a text reply)"


@app.get("/", include_in_schema=False)
async def index():
    """Serve the single-page web UI."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        status="ok", model=s.ollama_model, workspace=str(s.workspace_path)
    )


@app.get("/api/files")
async def list_files() -> dict:
    """List every file the agents have in the workspace sandbox (recursive)."""
    root = get_settings().workspace_path
    files = [
        str(p.relative_to(root)).replace("\\", "/")
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.name != ".gitkeep"
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    ]
    return {"files": files}


@app.get("/api/file")
async def get_file(path: str) -> dict:
    """Read a single workspace file (path-confined to the sandbox)."""
    try:
        target = _resolve(path)
    except SandboxError:
        raise HTTPException(status_code=400, detail="path escapes sandbox")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return {
        "path": path,
        "content": target.read_text(encoding="utf-8", errors="replace"),
    }


def _single_topology() -> dict:
    s = get_settings()
    return {
        "type": "react",
        "nodes": [{"id": "main", "label": "Coder Agent", "model": s.ollama_model, "kind": "agent"}],
        "edges": [],
    }


@app.get("/api/topology")
async def topology(mode: AgentMode) -> dict:
    """Return the graph shape + per-node model for the selected mode, so the UI
    can draw the RIGHT diagram (ReAct loop / LangGraph DAG / DeepAgents tree)."""
    if mode == AgentMode.parallel:
        return parallel_topology()
    if mode == AgentMode.orchestrator:
        return orchestrator_topology()
    return _single_topology()


@app.get("/api/servers")
async def servers() -> dict:
    """List background servers the agents have started (for the UI's link panel)."""
    return {"servers": running_servers()}


@app.post("/api/run/backend")
async def run_backend(port: int = 8090, module: str = "main", app_var: str = "app") -> dict:
    """Let the USER launch the built FastAPI backend on demand (Run button)."""
    return {"status": launch_backend(port, module, app_var), "servers": running_servers()}


@app.post("/api/run/frontend")
async def run_frontend(port: int = 8091) -> dict:
    """Let the USER launch the static frontend server on demand (Run button)."""
    return {"status": launch_frontend(port), "servers": running_servers()}


@app.post("/api/run/stop")
async def run_stop() -> dict:
    """Stop all background servers (Stop button)."""
    stop_all_servers()
    return {"status": "stopped all", "servers": running_servers()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Run the selected agent to completion and return its final message."""
    agent = AGENTS[req.mode]
    result = await agent.ainvoke(_inputs(req.message), config=_config(req.thread_id))
    reply = _final_text(result["messages"])
    return ChatResponse(reply=reply, thread_id=req.thread_id, mode=req.mode)


def _sse(event: str, **data) -> dict:
    return {"event": event, "data": json.dumps(data)}


def _scope(namespace) -> str:
    """Human-readable "which agent" from a LangGraph subgraph namespace.

    namespace is a tuple like ('backend:uuid', 'tools:uuid'); we take the nearest
    parent node's name (before the ':') so the UI can show, e.g., which parallel
    worker or which subgraph produced an event. Empty tuple = top level.
    """
    if not namespace:
        return "main"
    return str(namespace[-1]).split(":")[0]


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream the run as Server-Sent Events, with tool activity + subgraph scope.

    We ask LangGraph for TWO stream modes AND to descend into subgraphs:
      * stream_mode="messages" -> LLM token chunks as generated.
      * stream_mode="updates"  -> state deltas, exposing tool CALLS and RESULTS.
      * subgraphs=True         -> also stream from nested graphs (parallel workers,
        DeepAgents subagents), tagged with a namespace so the UI knows WHO acted.

    With subgraphs=True + a list stream_mode, each item is
    (namespace, mode, payload). Event types emitted: token, tool_call,
    tool_result, delegate, done.
    """
    agent = AGENTS[req.mode]

    async def event_generator():
      # coalesce noisy reasoning streams: one 'thinking' event per few chunks
      think_buf: dict[str, str] = {}
      think_cnt: dict[str, int] = {}
      try:
        async for namespace, stream_mode, payload in agent.astream(
            _inputs(req.message),
            config=_config(req.thread_id),
            stream_mode=["updates", "messages"],
            subgraphs=True,
        ):
            scope = _scope(namespace)
            if stream_mode == "messages":
                chunk, metadata = payload
                node = metadata.get("langgraph_node", "")
                ak = getattr(chunk, "additional_kwargs", {}) or {}
                reasoning = ak.get("reasoning_content")
                if reasoning:
                    # THINKING tokens (Ollama `think`) — stream them, coalesced.
                    think_buf[scope] = think_buf.get(scope, "") + reasoning
                    think_cnt[scope] = think_cnt.get(scope, 0) + 1
                    if think_cnt[scope] == 1 or len(think_buf[scope]) >= 180:
                        yield _sse("thinking", scope=scope, node=node,
                                   text=think_buf[scope][-180:])
                        think_buf[scope] = ""
                    continue
                text = _as_text(getattr(chunk, "content", ""))
                if text:
                    think_cnt[scope] = 0  # reasoning phase ended for this scope
                    yield _sse("token", text=text, node=node, scope=scope)
            elif stream_mode == "updates":
                for node, update in (payload or {}).items():
                    msgs = update.get("messages", []) if isinstance(update, dict) else []
                    for m in msgs:
                        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                name = tc.get("name", "tool")
                                args = tc.get("args", {})
                                # DeepAgents delegate to a subagent via the `task` tool.
                                if name == "task":
                                    yield _sse(
                                        "delegate",
                                        node=node,
                                        scope=scope,
                                        subagent=args.get("subagent_type", "subagent"),
                                        description=args.get("description", ""),
                                    )
                                else:
                                    yield _sse(
                                        "tool_call",
                                        node=node,
                                        scope=scope,
                                        name=name,
                                        args=args,
                                    )
                        elif isinstance(m, ToolMessage):
                            yield _sse(
                                "tool_result",
                                node=node,
                                scope=scope,
                                name=getattr(m, "name", "tool"),
                                content=_as_text(m.content),
                            )
      except Exception as e:  # noqa: BLE001 — surface any run error to the UI
        yield _sse("error", message=str(e)[:400])
      finally:
        # ALWAYS close the stream cleanly so the UI can finalize (✅ + refresh),
        # even if the graph raised or hit the recursion limit mid-run.
        yield _sse("done", thread_id=req.thread_id)

    return EventSourceResponse(event_generator())


# NOTE: declared LAST so specific /api/* routes (files, file, topology, servers)
# match first. This catch-all lets a previewed frontend's relative /api/... calls
# reach whichever backend the agents launched (start_backend), so the rendered
# page in the WebUI shows live data on the same origin.
@app.get("/api/{rest:path}")
async def preview_api_proxy(rest: str, request: Request):
    backends = [s for s in running_servers() if s["kind"] == "backend" and s["running"]]
    if not backends:
        raise HTTPException(status_code=503, detail="no backend server is running")
    base = f"http://127.0.0.1:{backends[0]['port']}"
    query = f"?{request.url.query}" if request.url.query else ""
    try:
        with urllib.request.urlopen(f"{base}/api/{rest}{query}", timeout=8) as r:
            return Response(content=r.read(),
                            media_type=r.headers.get_content_type() or "application/json")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"proxy failed: {e}")
