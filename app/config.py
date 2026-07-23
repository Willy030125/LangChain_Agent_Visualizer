"""Stage 0 — Foundation: configuration + the local model factory.

Everything in the app reads settings from here. We use pydantic-settings so
config comes from environment variables / .env with type validation, and we
expose one `get_model()` factory so every agent shares the same Ollama client.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from langchain_ollama import ChatOllama
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Ollama (can be local or remote server)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_temperature: float = 0.0
    ollama_num_ctx: int = 8192

    # Stream the model's THINKING (reasoning) tokens, not just the final answer.
    # qwen3 models "think" for many seconds before answering; without this the UI
    # sees a long silence then Done. With it, reasoning streams live so every
    # stage lights up. Maps to Ollama's `think: true`.
    ollama_reasoning: bool = True

    # Safety cap on tokens generated per model call (thinking + answer). Bounds
    # "endless thinking" spirals on small models (e.g. qwen3:8b looping on
    # recursion) so a single step can't hang forever. Generous enough for normal
    # code + reasoning.
    ollama_num_predict: int = 6144

    # Default model for the single agent.
    ollama_model: str = "qwen3:8b"

    # ---- The 3 local models, mapped to roles by strength ----
    #  qwen3.5:latest  = long-smart thinker (~9B): orchestration / planning / logic
    #  qwen3:8b        = strong coder: tool calling, code gen, code execution
    #  qwen3.5:4b      = mini thinker: summaries, README, reporting, short text
    # Orchestrator = the DECISIVE tool-caller (qwen3:8b, thinking off). The big
    # thinker (qwen3.5:latest) tends to "ego-write" code + narrate instead of
    # delegating, so it's used where real thinking helps: coder & debugger.
    model_orchestrator: str = "qwen3:8b"         # supervisor: reliably calls task()
    model_planner: str = "qwen3.5:latest"        # architect (thinking off, NO code)
    model_coder: str = "qwen3.5:latest"          # writes code (smart thinker)
    model_debugger: str = "qwen3.5:latest"       # runs + fixes (smart thinker)
    model_reviewer: str = "qwen3.5:4b"           # reviews / critiques
    model_reporter: str = "qwen3.5:4b"           # README / summaries / reports

    # Sandbox / execution
    workspace_dir: str = "./workspace"
    exec_timeout: int = 60
    allow_shell: bool = True

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @property
    def workspace_path(self) -> Path:
        """Absolute, resolved sandbox root. Created on first access."""
        root = Path(self.workspace_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root


@lru_cache
def get_settings() -> Settings:
    """Singleton settings (cached so .env is parsed once)."""
    return Settings()


def subprocess_env() -> dict:
    """Environment for CHILD processes (run_shell, uvicorn, http.server, pip…).

    The agents run shell commands like `uvicorn main:app` or `python x.py`. By
    default a child inherits the OS PATH, where those resolve to the *system*
    Python (3.10, no fastapi/uvicorn). We prepend THIS conda env's directories so
    `python`, `uvicorn`, `pip`, etc. resolve to the same `agentic` env the server
    runs in — preventing "No module named fastapi/uvicorn" errors.
    """
    env = os.environ.copy()
    env_root = Path(sys.executable).parent  # conda env root (python.exe lives here on Windows)
    extra = [
        str(env_root),
        str(env_root / "Scripts"),          # uvicorn.exe, pip.exe, etc. (Windows)
        str(env_root / "bin"),              # POSIX fallback
        str(env_root / "Library" / "bin"),  # conda DLLs (Windows)
    ]
    env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
    env["CONDA_PREFIX"] = str(env_root)
    env["VIRTUAL_ENV"] = str(env_root)
    return env


def get_model(**overrides) -> ChatOllama:
    """Factory for the local chat model.

    We return a *ChatOllama instance* (not a "ollama:name" string) so we can
    pin base_url, context window and temperature. create_agent /
    create_deep_agent both accept a model instance directly.

    `num_ctx` matters a lot for coding agents: the default Ollama context is
    tiny, so tool outputs + file contents overflow it and the agent "forgets".
    """
    s = get_settings()
    params = dict(
        model=s.ollama_model,
        base_url=s.ollama_base_url,
        temperature=s.ollama_temperature,
        num_ctx=s.ollama_num_ctx,
        num_predict=s.ollama_num_predict,  # cap per-call generation (anti-runaway)
        # Stream reasoning tokens (Ollama `think: true`). This is what makes the
        # live-flow UI light up during the model's long thinking phase.
        reasoning=s.ollama_reasoning,
        # Fail fast at startup if the model isn't pulled, instead of at 1st call.
        validate_model_on_init=True,
    )
    params.update(overrides)
    return ChatOllama(**params)
