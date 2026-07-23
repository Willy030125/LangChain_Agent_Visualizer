"""Stage 3 — the Deep Agents multi-agent orchestrator.

WHAT A "DEEP AGENT" IS
----------------------
A plain agent (Stage 2) plans in its head and does everything itself; on long
tasks it loses the thread. `create_deep_agent` adds four things on top of the
same LangGraph loop:

  1. A PLANNING tool (write_todos) — the agent writes an explicit to-do list and
     checks items off, so multi-step work stays on track.
  2. SUBAGENTS — specialized agents it can delegate to via a built-in `task`
     tool. Each subagent has its own prompt + tools and its own clean context
     window, so the orchestrator's context doesn't fill up with detail.
  3. A FILESYSTEM (here a real one, confined to the workspace) as shared memory.
  4. A detailed system prompt teaching it to plan → delegate → verify.

So the orchestrator is the SUPERVISOR: it plans and routes work to
planner / coder / reviewer / debugger subagents. That is the multi-agent
pattern you wanted — deepagents is itself built on LangGraph.
"""
from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from app.config import get_model, get_settings
from app.tools import EXEC_TOOLS, SERVER_TOOLS

# The deep agent uses DeepAgents' OWN built-in file tools (write_file/read_file/
# edit_file/ls) — NOT ours — so there is exactly one filesystem toolset and the
# "/workspace/..." path convention works. We only add execution + server tools.
# The orchestrator gets no coding tools of its own, so it MUST delegate.
COORDINATOR_TOOLS = [*EXEC_TOOLS, *SERVER_TOOLS]

ORCHESTRATOR_PROMPT = """You are the orchestrator. You do NOT plan, code, or write \
files — your subagents do. You drive the team with two tools: write_todos (your \
checklist) and task() (delegate). Everything is a tool call — never write prose, \
a plan, code, or a fake "task(...)" as text.

STEP 1 — call write_todos with EXACTLY these 4 items:
  1. plan the app (planner)
  2. code ALL files (coder)
  3. start server and verify (debugger)
  4. write README (reporter)

STEP 2 — work the checklist top to bottom. For each unchecked item, call the
matching task() and then mark it done:
  - task(subagent_type="planner",  description="<repeat the user's request in full>")
  - task(subagent_type="coder",    description="Create ALL files from the plan:
    main.py AND index.html AND index.js, full contents, one write_file each")
  - task(subagent_type="debugger", description="start_backend(8090) and verify /api/data")
  - task(subagent_type="reporter", description="write a short README")

THE ONE RULE YOU KEEP BREAKING: getting the planner's plan back means item 1 is
done — it is NOT the end. You MUST immediately continue to item 2 and call
task("coder"). You are FORBIDDEN from stopping or writing a summary while ANY
checklist item is unchecked. Keep calling task() until all 4 are done.

Only after ALL 4 items are checked off, reply with ONE sentence. You may see a
write_file tool — never use it; always delegate to "coder".

STRICT RULES — read carefully:
- Your FIRST action MUST be an actual task(...) TOOL CALL. Do NOT write a plan or
  any markdown yourself — that is the planner's job.
- NEVER type "task(...)" as text in your message, and NEVER paste a plan or code.
  Typing it does NOTHING — you must invoke the real tool. If you catch yourself
  writing a plan, STOP and call task("planner") instead.
- You may see a write_file tool; do NOT use it. Always delegate coding to "coder".
- ANTI-LOOP: planner once; debugger at most twice; one fix cycle max; then FINISH.
- Works for ANY app (scripts, APIs, business/product web demos) — not just numbers.
"""

# --- Subagent definitions ---------------------------------------------------
# We do NOT set "tools" per subagent: subagents inherit the main agent's tools
# (execution + server) PLUS DeepAgents' built-in file tools (write_file/read_file/
# edit_file/ls) which operate on the shared backend. This keeps ONE filesystem
# toolset and avoids the path-convention clash we hit before.
#
# PATH CONVENTION: DeepAgents' built-in file tools use a virtual filesystem where
# project files live under "/workspace/". Our CompositeBackend routes "/workspace/"
# to the real ./workspace on disk. The execution tools (run_python, start_backend)
# already run *inside* ./workspace, so there you use PLAIN names (main.py), not
# the /workspace/ prefix. We spell this out in the prompts.

def _build_subagents():
    """Construct subagent specs with their per-role model instances."""
    s = get_settings()
    return [
        {
            "name": "planner",
            "description": "MUST run first. Designs the whole solution: file structure, a text flow diagram, and the exact task list for coder/debugger/reviewer/reporter. No code.",
            "system_prompt": (
                "You are ONLY a planner/architect. HARD RULE: you MUST NOT write any code "
                "(no Python, HTML, JS, or snippets) and MUST NOT call write_file or edit_file. "
                "If you output code or touch a file, you FAIL. Your entire output is a plain-text "
                "PLAN + a text flow diagram — the coder writes the actual code later.\n"
                "For ANY app (script, API, or a full-stack web demo for a business/product idea), "
                "produce ONE COMPLETE, concrete plan the team follows exactly. Output these sections:\n"
                "1. FILE STRUCTURE: every file (e.g. /workspace/main.py, /workspace/index.html, "
                "/workspace/index.js) + one line each.\n"
                "2. FLOW (text diagram): e.g. `browser -> GET /api/data -> build_data() -> JSON "
                "-> index.js renders cards`.\n"
                "3. FUNCTIONS & DATA: functions/signatures and the SHAPE of the data (exact JSON "
                "field names). Prefer SIMPLE ITERATIVE code; never deep recursion.\n"
                "4. CODER TASKS: numbered, one per file, each a tight single-pass spec.\n"
                "5. TEST PLAN: exactly what the debugger runs / which URL.\n"
                "6. REVIEW & REPORT: brief.\n"
                "FIXED WEB-APP TEMPLATE — you MUST use EXACTLY these 3 files and this route; "
                "do NOT invent other names/paths (no app.py, no /static/, no /products):\n"
                "  • /workspace/main.py   — FastAPI, route EXACTLY @app.get('/api/data'), returns a "
                "JSON list of realistic demo objects for the topic.\n"
                "  • /workspace/index.html — has <div id=\"app\"></div> and <script src=\"index.js\">.\n"
                "  • /workspace/index.js   — fetch('/api/data') and render items into #app.\n"
                "Your plan describes only the DATA (exact field names + demo values) and each file's "
                "content. The server is tested with start_backend(port=8090) + http_get(.../api/data). "
                "Do NOT write code — only the plan."
            ),
            # thinking OFF: keeps the planner terse and structural (no ego-coding).
            "model": get_model(model=s.model_planner, reasoning=False),
        },
        {
            "name": "coder",
            "description": "Implements ONE file per call from the planner's spec. Writes code to /workspace/.",
            "system_prompt": (
                "You are a programmer. Implement EXACTLY the planner's spec for the file(s) you "
                "were asked to create — do not redesign. Avoid overthinking:\n"
                "- Create EVERY file you were asked for (often 3: main.py, index.html, index.js). "
                "Call write_file ONCE PER FILE. Do NOT skip files and do NOT claim a file exists "
                "without calling write_file for it.\n"
                "- Write each file in ONE pass; do not reconsider repeatedly.\n"
                "- SIMPLE ITERATIVE code (loops); NEVER deep recursion.\n"
                "- write_file('/workspace/<name>', '<COMPLETE code>').\n"
                "FULL-STACK WEB CONTRACT (if applicable): backend /workspace/main.py = FastAPI "
                "with the route EXACTLY `@app.get('/api/data')` (do NOT rename it to /menu, "
                "/orders, etc.) returning a JSON list of realistic demo objects; frontend "
                "/workspace/index.html MUST include <script src=\"index.js\"></script> and a "
                "container div; /workspace/index.js does fetch('/api/data') and renders items. "
                "CRITICAL: frontend must use the EXACT SAME JSON field names the backend returns. "
                "After writing, reply with one sentence listing the files, then STOP."
            ),
            "model": get_model(model=s.model_coder),
        },
        {
            "name": "debugger",
            "description": "RUNS and TESTS the app, diagnoses failures, and fixes them until it works.",
            "system_prompt": (
                "You are a test specialist. The files live in the project folder, which is ALSO "
                "the working dir for run tools, so use PLAIN names there (not /workspace/).\n"
                "Verify efficiently — do NOT loop:\n"
                "- Python script: run_python(\"import main; print('ok')\") or call functions.\n"
                "- Backend: ALWAYS start_backend(port=8090) (NEVER port 8000 — that's the control "
                "API). The app is main.py. Then http_get('http://127.0.0.1:8090/api/data'). If that "
                "404s, do http_get('http://127.0.0.1:8090/') and read main.py to find the real route, "
                "test that ONCE. A running server returning ANY 200 counts as PASS.\n"
                "Total checks: AT MOST TWO. Then report PASS or the issue and STOP. Do at most ONE "
                "fix (edit_file, re-run once). NEVER keep retrying — stopping is required."
            ),
            "tools": COORDINATOR_TOOLS,  # start_backend / http_get / run_python / etc.
            "model": get_model(model=s.model_debugger),
        },
        {
            "name": "reviewer",
            "description": "Reads code and reports concrete problems. Does not edit.",
            "system_prompt": (
                "You are a code reviewer. Read files with read_file (/workspace/ paths) and list "
                "concrete issues (bugs, missing pieces, wrong routes) with a suggested fix. Be brief."
            ),
            "model": get_model(model=s.model_reviewer),
        },
        {
            "name": "reporter",
            "description": "Writes the README and the final human-readable summary.",
            "system_prompt": (
                "You write clear docs. Create /workspace/README.md with what the app is and the "
                "exact commands to run it (uvicorn main:app ...; python -m http.server ...). "
                "Use write_file. Keep it short."
            ),
            "model": get_model(model=s.model_reporter),
        },
    ]


def orchestrator_topology() -> dict:
    """Supervisor→subagents tree + per-node model, for the UI's DeepAgents view."""
    s = get_settings()
    subs = [
        ("planner", s.model_planner), ("coder", s.model_coder),
        ("debugger", s.model_debugger), ("reviewer", s.model_reviewer),
        ("reporter", s.model_reporter),
    ]
    nodes = [{"id": "main", "label": "orchestrator", "model": s.model_orchestrator, "kind": "supervisor"}]
    nodes += [{"id": n, "label": n, "model": m, "kind": "subagent"} for n, m in subs]
    edges = [["main", n] for n, _ in subs]
    return {"type": "deep", "nodes": nodes, "edges": edges}


def build_orchestrator(checkpointer=None):
    """Build the Deep Agents orchestrator with its subagent team.

    Args:
        checkpointer: optional LangGraph checkpointer for per-thread memory.
    """
    settings = get_settings()
    return create_deep_agent(
        # reasoning=False: the orchestrator must ACT (call task()) not narrate.
        # Reasoning models tend to essay a plan + type a fake "task(...)" as text
        # instead of emitting a real tool call. Subagents keep reasoning on.
        model=get_model(model=settings.model_orchestrator, reasoning=False),
        tools=[],  # only built-in task()/write_todos — declutter so it delegates
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=_build_subagents(),
        # CompositeBackend (the DeepAgents-recommended pattern):
        #   - "/workspace/..." paths  -> the REAL ./workspace on disk (project files)
        #   - everything else (todos, agent scratch) -> ephemeral in-memory StateBackend
        # This is why the built-in write_file('/workspace/main.py') lands on disk and
        # shows up in ./workspace — fixing the "empty workspace" bug.
        # Route BOTH the "/workspace/" convention AND any other path to the real
        # ./workspace on disk. Using FilesystemBackend as the DEFAULT (not
        # StateBackend) means even if a subagent drifts to /static/x or /main.py,
        # the file still lands on disk instead of vanishing into in-memory state.
        backend=CompositeBackend(
            default=FilesystemBackend(
                root_dir=str(settings.workspace_path), virtual_mode=True
            ),
            routes={
                "/workspace/": FilesystemBackend(
                    root_dir=str(settings.workspace_path), virtual_mode=True
                ),
            },
        ),
        checkpointer=checkpointer,
        name="orchestrator",
    )
