"""Manual smoke test — verifies the whole stack without starting the API.

Run:  conda run -n agentic python smoke_test.py
It (1) prints config, (2) builds the single agent, (3) builds the orchestrator
(which validates all 5 per-role models exist on the remote server), and
(4) does one live tool-calling round-trip.
"""
from app.agents import build_coder_agent, build_orchestrator
from app.config import get_settings


def main() -> None:
    s = get_settings()
    print(f"[config] base_url={s.ollama_base_url}  default_model={s.ollama_model}")

    print("[build] single coder agent ...", flush=True)
    coder = build_coder_agent()
    print("        ok")

    print("[build] orchestrator + 4 subagents (validates 5 models) ...", flush=True)
    build_orchestrator()
    print("        ok")

    print("[live] single-agent tool call (write a file) ...", flush=True)
    result = coder.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the write_file tool to create a file named "
                        "smoke.txt containing exactly: ok. Then reply with DONE."
                    ),
                }
            ]
        }
    )
    print("[reply]", result["messages"][-1].content[:600])


if __name__ == "__main__":
    main()
