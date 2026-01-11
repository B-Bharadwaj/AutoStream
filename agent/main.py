from __future__ import annotations

import os
from agent.agent import build_app
from agent.state import AgentState, Message


def run_cli():
    # Required env var for OpenAI:
    # export OPENAI_API_KEY="..."
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY is not set.")
        print("Set it and retry:")
        print("  Windows PowerShell: setx GOOGLE_API_KEY \"YOUR_KEY\"")
        print("  macOS/Linux: export GOOGLE_API_KEY=\"YOUR_KEY\"")
        return


    app = build_app(kb_path="data/knowledge_base.json")
    state = AgentState(history=[Message(role="system", content="AutoStream agent session started.")])

    print("AutoStream Agent (type 'exit' to quit)\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
        # Let the agent handle graceful farewell routing
            state.user_input = user
            state.history.append(Message(role="user", content=user))
            out = app.invoke(state)
            state = AgentState(**out)
            last = state.history[-1]
            print(f"\nAgent: {last.content}\n")
            break


        state.user_input = user
        state.history.append(Message(role="user", content=user))

        out = app.invoke(state)          # LangGraph returns dict in your setup
        state = AgentState(**out)        # convert dict -> AgentState
        state.history = state.history[-12:] # keep last 6 turns (user+assistant)
        last = state.history[-1]
        print(f"\nAgent: {last.content}\n")

if __name__ == "__main__":
    run_cli()
