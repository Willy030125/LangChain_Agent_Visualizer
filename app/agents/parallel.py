"""Stage 3a — Parallel multi-agent team (raw LangGraph).

This is the "Multi-Agents / Parallel" pattern, built directly on LangGraph so you
can SEE the graph. Unlike the DeepAgents orchestrator (one supervisor that
delegates sequentially), here three specialist coder agents run CONCURRENTLY and
then a join node merges their results:

        ┌─> backend  ─┐
   START├─> frontend ─┤─> aggregate ─> END
        └─> tests   ─┘
   (all three start at once; aggregate waits for all = a "join"/barrier)

LangGraph runs nodes with no dependency between them in parallel. Because our
node functions are async and call `await agent.ainvoke(...)`, the three Ollama
calls genuinely overlap. `add_messages` safely merges their concurrent writes to
shared state.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import get_model, get_settings
from app.tools import ALL_TOOLS


class TeamState(TypedDict):
    # add_messages is a reducer: it appends, and tolerates concurrent writes
    # from the parallel branches (essential — three nodes write at once).
    messages: Annotated[list, add_messages]


def _task_text(messages: list) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return str(messages[-1].content) if messages else ""


def _summary(messages: list) -> str:
    """Short summary of what a worker did (last assistant text, else file writes)."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()[:400]
    writes = [
        str(m.content)
        for m in messages
        if getattr(m, "name", "") in ("write_file", "edit_file")
    ]
    return "; ".join(writes)[:400] or "(done)"


# Each worker uses a DIFFERENT model so all 3 run concurrently and are visible
# in the UI:  (node name, role label, model-setting attr, instruction)
WORKERS = [
    (
        "backend",
        "smart thinker → API/logic",
        "model_orchestrator",  # qwen3.5:latest
        "Build the BACKEND only for the requested app (any topic — e.g. a business/"
        "product demo). Create a FastAPI app in `main.py`. SHARED API CONTRACT (must "
        "match EXACTLY): expose GET /api/data returning the app's data as JSON — a "
        "list of objects with realistic demo data for the topic, e.g. "
        "[{\"name\":..., \"price\":...}]. No request body needed (GET). Use write_file, "
        "then verify with run_python(\"import main; print('ok')\"). No frontend files.",
    ),
    (
        "frontend",
        "code model → HTML/JS",
        "model_coder",  # qwen3:8b
        "Build the FRONTEND only for the requested app. Create `index.html` and "
        "`index.js` (plain JS, no framework). index.html MUST include "
        "`<script src=\"index.js\"></script>` before </body> (or the JS never runs), "
        "and a container like <div id=\"app\"></div>. SHARED API CONTRACT (must match "
        "EXACTLY): index.js does fetch('/api/data'), reads the JSON list, and renders "
        "each item (cards/table/list) into the container. Show a title relevant to the "
        "app. Use write_file for each file. Do NOT create backend files.",
    ),
    (
        "docs",
        "mini model → README/report",
        "model_reporter",  # qwen3.5:4b
        "Build the DOCS only. Create a `README.md` that explains what the app does "
        "and gives the exact commands to run the backend (uvicorn main:app) and "
        "frontend. Use write_file. Do NOT modify main.py, index.html, or index.js.",
    ),
]


def parallel_topology() -> dict:
    """Graph shape + per-node model, for the UI's LangGraph diagram."""
    from app.config import get_settings
    s = get_settings()
    nodes = [{"id": "START", "label": "START", "kind": "io"}]
    for name, role, attr, _ in WORKERS:
        nodes.append({"id": name, "label": name, "role": role,
                      "model": getattr(s, attr), "kind": "worker"})
    nodes.append({"id": "aggregate", "label": "aggregate (join)", "kind": "join"})
    nodes.append({"id": "END", "label": "END", "kind": "io"})
    edges = [["START", w[0]] for w in WORKERS] + \
            [[w[0], "aggregate"] for w in WORKERS] + [["aggregate", "END"]]
    return {"type": "graph", "nodes": nodes, "edges": edges}


def _make_worker(name: str, role: str, agent):
    async def node(state: TeamState):
        task = _task_text(state["messages"])
        prompt = (
            f"OVERALL PROJECT REQUEST:\n{task}\n\n"
            f"YOUR ROLE ({name}): {role}\n"
            "Write your file(s) with write_file. Keep it minimal but working. "
            "When done, reply with one sentence listing the files you created."
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        return {"messages": [AIMessage(content=f"[{name}] {_summary(result['messages'])}")]}

    return node


async def _aggregate(state: TeamState):
    lines = [
        m.content
        for m in state["messages"]
        if isinstance(m, AIMessage)
        and isinstance(m.content, str)
        and m.content.startswith("[")
    ]
    combined = "Parallel build complete. Each agent worked concurrently:\n" + "\n".join(
        lines
    )
    return {"messages": [AIMessage(content=combined)]}


def build_parallel_team(checkpointer=None):
    """Compile the fan-out/fan-in team graph."""
    s = get_settings()
    graph = StateGraph(TeamState)

    for name, role, model_attr, instruction in WORKERS:
        agent = create_agent(
            model=get_model(model=getattr(s, model_attr)),
            tools=ALL_TOOLS,
            system_prompt=(
                f"You are the {name} specialist ({role}) on a software team. "
                "Do your part by CALLING TOOLS (write_file, run_python, etc.). "
                "Write complete files and verify them. Then reply in ONE sentence "
                "listing the files you created."
            ),
        )
        graph.add_node(name, _make_worker(name, instruction, agent))
        graph.add_edge(START, name)      # fan-out: all start together
        graph.add_edge(name, "aggregate")  # fan-in: join at aggregate

    graph.add_node("aggregate", _aggregate)
    graph.add_edge("aggregate", END)
    return graph.compile(checkpointer=checkpointer)
