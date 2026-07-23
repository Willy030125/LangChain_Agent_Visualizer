"""Stage 4 — Pydantic API contracts.

FastAPI uses these models to validate requests, serialize responses, and
auto-generate the OpenAPI docs at /docs. Defining the boundary with Pydantic is
what keeps the messy agent internals from leaking into your HTTP API.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AgentMode(str, Enum):
    """Which agent handles the request."""

    single = "single"              # Stage 2: one coder agent
    parallel = "parallel"          # Stage 3a: fan-out team (LangGraph, concurrent)
    orchestrator = "orchestrator"  # Stage 3b: Deep Agents multi-agent team


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's instruction to the agent.")
    thread_id: str = Field(
        default="default",
        description="Conversation id. Same id = shared memory across calls.",
    )
    mode: AgentMode = Field(
        default=AgentMode.orchestrator,
        description="single = one coder agent; orchestrator = multi-agent team.",
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="The agent's final answer.")
    thread_id: str
    mode: AgentMode


class HealthResponse(BaseModel):
    status: str
    model: str
    workspace: str
