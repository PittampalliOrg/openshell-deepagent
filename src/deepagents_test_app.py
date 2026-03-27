"""Diagrid DeepAgents baseline service for OpenShell observability testing."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
import asyncio
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext, WorkflowActivityContext
from dapr_agents import DaprChatClient
from dapr_agents.tool.utils.tool import ToolHelper
from deepagents import create_deep_agent
from diagrid.agent.core.telemetry import instrument_grpc, setup_telemetry
from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.tools import tool as langchain_tool
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from src.dapr_langchain import DaprLangChainChatModel
from src.prompts import build_system_prompt

DEFAULT_APP_PORT = 8002
DEFAULT_LLM_COMPONENT = "llm-provider"
DEEPAGENTS_NAME = "openshell-deepagents-test"
DEEPAGENTS_WORKFLOW_NAME = "dapr.agents.OpenShellDeepagentsTest.workflow"
DEEPAGENTS_ACTIVITY_NAME = "run_openshell_deepagents_test_activity"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

load_dotenv()


def _patch_dapr_tool_format_bug() -> None:
    """Normalize dict-based Dapr tool schemas to the OpenAI-compatible format.

    `dapr_agents` already aliases callable tools from `dapr` to `openai`, but the
    dict validation path still rejects `tool_format="dapr"`. The Diagrid runner
    feeds LangChain-bound dict tools into that path, so patch it locally for the
    baseline until the upstream package is fixed.
    """

    current = ToolHelper.format_tool
    if getattr(current, "__name__", "") == "_patched_format_tool":
        return

    def _patched_format_tool(
        tool: dict[str, Any] | Any,
        tool_format: str = "openai",
        use_deprecated: bool = False,
    ) -> dict[str, Any]:
        normalized_format = (
            "openai" if str(tool_format).lower() == "dapr" else tool_format
        )
        return current(
            tool,
            tool_format=normalized_format,
            use_deprecated=use_deprecated,
        )

    ToolHelper.format_tool = staticmethod(_patched_format_tool)
    logger.info("Patched ToolHelper.format_tool to normalize Dapr dict tools")


@langchain_tool
def write_text_file(path: str, content: str) -> str:
    """Write UTF-8 text to a local file path and create parent directories if needed."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {target}"


@langchain_tool
def read_text_file(path: str) -> str:
    """Read UTF-8 text from a local file path."""
    target = Path(path).expanduser()
    return target.read_text(encoding="utf-8")


@langchain_tool
def path_exists(path: str) -> bool:
    """Return whether a local file path exists."""
    return Path(path).expanduser().exists()


BASELINE_TOOLS = [
    write_text_file,
    read_text_file,
    path_exists,
]


class InvokeRequest(BaseModel):
    """Workflow input accepted by the baseline service."""

    model_config = ConfigDict(extra="allow")

    task: str | None = Field(
        None, description="User task for the baseline agent"
    )
    prompt: str | None = Field(
        None, description="Workflow-builder prompt alias"
    )
    goal: str | None = Field(
        None, description="Legacy goal alias for task"
    )
    threadId: str | None = Field(
        None, description="Conversation thread identifier"
    )
    executionId: str | None = Field(
        None, description="Per-run execution identifier"
    )
    dbExecutionId: str | None = Field(
        None, description="Database execution identifier"
    )
    sandboxName: str | None = Field(None, description="OpenShell sandbox name")
    repoUrl: str | None = Field(None, description="Git repository URL to clone")
    repositoryUrl: str | None = Field(
        None, description="Workflow-builder repository URL alias"
    )
    repoBranch: str | None = Field(None, description="Git branch to checkout")
    repositoryBranch: str | None = Field(
        None, description="Workflow-builder repository branch alias"
    )
    repoToken: str | None = Field(None, description="Git auth token")
    repositoryToken: str | None = Field(
        None, description="Workflow-builder repository auth token alias"
    )
    provider: str | None = Field(None, description="Preferred provider label")
    model: str | None = Field(None, description="Preferred model identifier")
    mode: str | None = Field(None, description="Workflow-builder mode label")
    workflowId: str | None = Field(
        None, description="Optional wrapper workflow instance identifier"
    )
    planningThreadId: str | None = Field(
        None, description="Planning thread identifier"
    )
    executionThreadId: str | None = Field(
        None, description="Execution thread identifier"
    )
    timeoutMinutes: int = Field(30, description="Max execution time in minutes")
    traceId: str | None = Field(None, description="Parent trace identifier")


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


def _current_trace_id() -> str | None:
    span = trace.get_current_span()
    if span is None:
        return None
    span_context = span.get_span_context()
    if not span_context or not span_context.is_valid:
        return None
    return f"{span_context.trace_id:032x}"


def _first_nonempty(input_data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = input_data.get(key)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed
    return ""


def _normalize_diagrid_otlp_env() -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return

    normalized_endpoint = endpoint
    parsed = urlparse(endpoint)
    if parsed.port == 4318:
        normalized_endpoint = urlunparse(
            parsed._replace(netloc=f"{parsed.hostname}:4317", path="")
        )

    if normalized_endpoint != endpoint:
        logger.info(
            "Normalized OTLP endpoint for Diagrid gRPC exporter: %s -> %s",
            endpoint,
            normalized_endpoint,
        )
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = normalized_endpoint

    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "").strip().lower()
    if protocol == "http/protobuf":
        os.environ.pop("OTEL_EXPORTER_OTLP_PROTOCOL", None)
        logger.info("Removed OTLP HTTP protocol override for Diagrid gRPC exporter")


def _resolve_task(input_data: dict[str, Any]) -> str:
    return _first_nonempty(input_data, "task", "prompt", "goal")


def _build_messages(input_data: dict[str, Any]) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    repo_url = _first_nonempty(input_data, "repoUrl", "repositoryUrl")
    if repo_url:
        clone_cmd = "git clone"
        repo_branch = _first_nonempty(input_data, "repoBranch", "repositoryBranch")
        repo_token = _first_nonempty(input_data, "repoToken", "repositoryToken")
        if repo_branch:
            clone_cmd += f" -b {repo_branch}"
        if repo_token and repo_url.startswith("https://"):
            clone_url = repo_url.replace(
                "https://", f"https://oauth2:{repo_token}@", 1
            )
        else:
            clone_url = repo_url
        clone_cmd += f" {clone_url} /sandbox/repo"
        messages.append(
            ("system", f"Before starting, clone the repository: {clone_cmd}")
        )
    messages.append(("user", _resolve_task(input_data)))
    return messages


@lru_cache(maxsize=1)
def get_runner() -> DaprWorkflowDeepAgentRunner:
    """Create and configure the shared DeepAgents runner."""
    _patch_dapr_tool_format_bug()
    current_date = datetime.now().strftime("%Y-%m-%d")
    llm_component = os.getenv("DAPR_LLM_COMPONENT", DEFAULT_LLM_COMPONENT)
    max_steps = int(os.getenv("DEEPAGENTS_MAX_STEPS", "100"))

    model = DaprLangChainChatModel(
        client=DaprChatClient(component_name=llm_component),
    )
    agent = create_deep_agent(
        model=model,
        tools=BASELINE_TOOLS,
        system_prompt=build_system_prompt(current_date),
        name=DEEPAGENTS_NAME,
    )
    runner = DaprWorkflowDeepAgentRunner(
        agent=agent,
        name=DEEPAGENTS_NAME,
        max_steps=max_steps,
    )
    runner._workflow_runtime.register_workflow(
        openshell_deepagents_test_workflow,
        name=DEEPAGENTS_WORKFLOW_NAME,
    )
    runner._workflow_runtime.register_activity(
        run_openshell_deepagents_test_activity,
        name=DEEPAGENTS_ACTIVITY_NAME,
    )
    return runner


async def _execute_deepagents_run(input_data: dict[str, Any]) -> dict[str, Any]:
    """Execute a single DeepAgents run and adapt it to workflow-builder output."""
    task = _resolve_task(input_data)
    if not task:
        return {"success": False, "error": "task is required"}

    sandbox_name = str(input_data.get("sandboxName") or "").strip()

    thread_id = (
        str(input_data.get("threadId") or "").strip()
        or str(input_data.get("executionId") or "").strip()
        or str(input_data.get("dbExecutionId") or "").strip()
        or str(input_data.get("executionThreadId") or "").strip()
        or str(input_data.get("planningThreadId") or "").strip()
        or str(input_data.get("workflowId") or "").strip()
        or "openshell-deepagents-test"
    )
    workflow_root = (
        str(input_data.get("executionId") or "").strip()
        or str(input_data.get("dbExecutionId") or "").strip()
        or str(input_data.get("workflowId") or "").strip()
        or thread_id
    )
    nested_workflow_id = f"{workflow_root}__deepagents"
    timeout_minutes = int(input_data.get("timeoutMinutes") or 30)

    runner = get_runner()
    completion_event: dict[str, Any] | None = None
    failure_event: dict[str, Any] | None = None
    trace_id: str | None = None

    async def _collect() -> None:
        nonlocal completion_event, failure_event
        async for event in runner.run_async(
            input={"messages": _build_messages(input_data)},
            thread_id=thread_id,
            workflow_id=nested_workflow_id,
            config={"thread_id": thread_id},
        ):
            event_type = str(event.get("type") or "").strip()
            if event_type == "workflow_completed":
                completion_event = event
                return
            if event_type in {"workflow_failed", "workflow_error", "workflow_terminated"}:
                failure_event = event
                return

    tracer = trace.get_tracer(DEEPAGENTS_NAME)
    try:
        with tracer.start_as_current_span(
            f"invoke_agent {DEEPAGENTS_NAME}",
            attributes={
                "agent.name": DEEPAGENTS_NAME,
                "workflow.instance_id": nested_workflow_id,
                "workflow.id": str(input_data.get("workflowId") or ""),
                "workflow.execution_id": str(input_data.get("executionId") or ""),
                "sandbox.name": sandbox_name,
            },
        ):
            trace_id = _current_trace_id()
            await asyncio.wait_for(_collect(), timeout=timeout_minutes * 60)
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"DeepAgents baseline timed out after {timeout_minutes} minutes",
            "agentWorkflowId": nested_workflow_id,
            "daprInstanceId": nested_workflow_id,
            "sandboxName": sandbox_name or None,
            "provider": input_data.get("provider"),
            "traceId": trace_id,
        }
    except Exception as exc:
        logger.exception("DeepAgents baseline run failed")
        return {
            "success": False,
            "error": str(exc),
            "agentWorkflowId": nested_workflow_id,
            "daprInstanceId": nested_workflow_id,
            "sandboxName": sandbox_name or None,
            "provider": input_data.get("provider"),
            "traceId": trace_id,
        }

    if failure_event is not None:
        error = failure_event.get("error")
        if isinstance(error, dict):
            error = error.get("message") or error.get("error_type") or str(error)
        return {
            "success": False,
            "error": str(error or "DeepAgents workflow failed"),
            "agentWorkflowId": nested_workflow_id,
            "daprInstanceId": nested_workflow_id,
            "sandboxName": sandbox_name or None,
            "provider": input_data.get("provider"),
            "traceId": trace_id,
        }

    output = (
        completion_event.get("output")
        if isinstance(completion_event, dict)
        else None
    )
    if not isinstance(output, dict):
        output = {}
    messages = output.get("messages")
    last_message = messages[-1] if isinstance(messages, list) and messages else None
    final_text = ""
    if isinstance(last_message, dict):
        final_text = str(last_message.get("content") or "").strip()
    elif last_message is not None:
        final_text = str(getattr(last_message, "content", last_message) or "").strip()

    updated_at = datetime.now().isoformat()

    return {
        "success": True,
        "content": final_text,
        "text": final_text,
        "final_answer": final_text,
        "assistantMessage": (
            _serialize_message(last_message) if last_message is not None else None
        ),
        "messages": [
            _serialize_message(message) if not isinstance(message, dict) else message
            for message in messages
        ]
        if isinstance(messages, list)
        else [],
        "traceId": trace_id,
        "agentProgress": {
            "status": "completed",
            "phase": str(input_data.get("mode") or "execute_direct"),
            "summary": final_text or None,
            "currentStepName": sandbox_name or None,
            "activeToolName": None,
            "stopReason": "workflow completed",
            "agentWorkflowId": nested_workflow_id,
            "daprInstanceId": nested_workflow_id,
            "traceId": trace_id,
            "updatedAt": updated_at,
            "recentTurns": [],
        },
        "agentWorkflowId": nested_workflow_id,
        "daprInstanceId": nested_workflow_id,
        "sandboxName": sandbox_name or None,
        "provider": input_data.get("provider"),
        "threadId": thread_id,
        "planningThreadId": input_data.get("planningThreadId"),
        "executionThreadId": input_data.get("executionThreadId"),
    }


def run_openshell_deepagents_test_activity(
    ctx: WorkflowActivityContext,
    input_data: dict[str, Any],
) -> dict[str, Any]:
    """Activity wrapper used by the simple native child workflow."""
    del ctx
    timeout_minutes = int(input_data.get("timeoutMinutes") or 30)
    runner = get_runner()
    return runner._run_sync(
        _execute_deepagents_run(input_data),
        timeout=max(timeout_minutes * 60 + 30, 300.0),
    )


def openshell_deepagents_test_workflow(
    ctx: DaprWorkflowContext,
    input_data: dict[str, Any],
) -> dict[str, Any]:
    """Simple workflow-builder compatible wrapper over the Diagrid runner."""
    workflow_input = dict(input_data or {})
    workflow_input["workflowId"] = getattr(ctx, "instance_id", None)
    result = yield ctx.call_activity(
        run_openshell_deepagents_test_activity,
        input=workflow_input,
    )
    return result


@asynccontextmanager
async def lifespan(_: FastAPI):
    _normalize_diagrid_otlp_env()
    setup_telemetry(DEEPAGENTS_NAME)
    instrument_grpc()
    runner = get_runner()
    runner.start()
    try:
        yield
    finally:
        runner.shutdown()


app = FastAPI(
    title="OpenShell DeepAgents Baseline",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/invoke")
async def invoke(request: InvokeRequest) -> dict[str, Any]:
    return await _execute_deepagents_run(request.model_dump())


@app.post("/api/run")
async def run_api(request: InvokeRequest) -> dict[str, Any]:
    return await _execute_deepagents_run(request.model_dump())


@app.get("/api/run/{workflow_id}")
def get_status(workflow_id: str) -> dict[str, Any]:
    status = get_runner().get_workflow_status(workflow_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return status


def main() -> None:
    """Run the baseline service with uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.deepagents_test_app:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", str(DEFAULT_APP_PORT))),
    )


if __name__ == "__main__":
    main()
