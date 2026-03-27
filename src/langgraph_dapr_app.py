"""HTTP wrapper around the LangGraph + Dapr checkpointer agent."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.langgraph_dapr_graph import graph

DEFAULT_APP_PORT = 8001


class InvokeRequest(BaseModel):
    """Single-turn invoke payload for the LangGraph runtime."""

    message: str = Field(..., description="User message for the agent")
    thread_id: str = Field(..., description="Conversation thread identifier")
    sandbox_name: str | None = Field(None, description="OpenShell sandbox name for this request")
    repo_url: str | None = Field(None, description="Git repository URL to clone")
    repo_branch: str | None = Field(None, description="Git branch to checkout")
    repo_token: str | None = Field(None, description="Git auth token")
    timeout_minutes: int = Field(30, description="Max execution time in minutes")


def _serialize_message(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": getattr(message, "type", "unknown"),
        "content": getattr(message, "content", ""),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    name = getattr(message, "name", None)
    if name:
        payload["name"] = name
    return payload


app = FastAPI(title="OpenShell LangGraph Dapr Agent")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Basic health endpoint for Kubernetes probes."""
    return {"status": "ok"}


@app.post("/invoke")
def invoke(request: InvokeRequest) -> dict[str, Any]:
    """Invoke the LangGraph agent for a single user turn."""
    if request.sandbox_name:
        from src.openshell_runtime import get_runtime
        get_runtime().set_sandbox_name(request.sandbox_name)

    setup_messages: list[tuple[str, str]] = []
    if request.repo_url:
        clone_cmd = f"git clone"
        if request.repo_branch:
            clone_cmd += f" -b {request.repo_branch}"
        if request.repo_token:
            # Insert token into URL for auth
            url = request.repo_url
            if url.startswith("https://"):
                url = url.replace("https://", f"https://oauth2:{request.repo_token}@", 1)
            clone_cmd += f" {url} /sandbox/repo"
        else:
            clone_cmd += f" {request.repo_url} /sandbox/repo"
        setup_messages.append(("system", f"Before starting, clone the repository: {clone_cmd}"))

    messages = setup_messages + [("user", request.message)]
    result = graph.invoke(
        {"messages": messages},
        {"configurable": {"thread_id": request.thread_id}},
    )
    messages = result["messages"]
    final_message = messages[-1]
    return {
        "thread_id": request.thread_id,
        "final_message": _serialize_message(final_message),
        "messages": [_serialize_message(message) for message in messages],
    }


def main() -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.langgraph_dapr_app:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", str(DEFAULT_APP_PORT))),
    )


if __name__ == "__main__":
    main()
