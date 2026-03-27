"""Native Dapr DurableAgent runtime for the OpenShell agent."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from dapr_agents import AgentRunner, DaprChatClient, DurableAgent
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dotenv import load_dotenv

from src.openshell_runtime import get_runtime
from src.openshell_tools import DURABLE_TOOLS
from src.prompts import build_system_prompt
from src.dapr_observability_patch import patch_workflow_monitor_wrapper

DEFAULT_APP_PORT = 8000
DEFAULT_LLM_COMPONENT = "llm-provider"
DEFAULT_MEMORY_STORE = "agent-memory"
DEFAULT_WORKFLOW_STORE = "agent-workflow"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

load_dotenv()
patch_workflow_monitor_wrapper()


def _trace_id_from_traceparent(traceparent: object) -> str | None:
    if not isinstance(traceparent, str):
        return None
    parts = traceparent.strip().split("-")
    if len(parts) != 4:
        return None
    trace_id = parts[1].strip().lower()
    if len(trace_id) != 32:
        return None
    try:
        int(trace_id, 16)
    except ValueError:
        return None
    return trace_id


def _trace_id_from_payload(payload: dict[str, Any]) -> str | None:
    direct = str(payload.get("traceId") or payload.get("trace_id") or "").strip()
    if direct:
        return direct
    otel_ctx = payload.get("_otel")
    if isinstance(otel_ctx, dict):
        direct = str(
            otel_ctx.get("traceId") or otel_ctx.get("trace_id") or ""
        ).strip()
        if direct:
            return direct
        return _trace_id_from_traceparent(otel_ctx.get("traceparent"))
    return None


def _workflow_context_key(instance_id: object) -> str | None:
    value = str(instance_id or "").strip()
    if not value:
        return None
    return f"__workflow_context_{value}__"


def _trace_id_from_workflow_context(instance_id: object) -> str | None:
    try:
        from dapr_agents.observability.context_storage import get_workflow_context

        key = _workflow_context_key(instance_id)
        if not key:
            return None
        context = get_workflow_context(key)
        if not isinstance(context, dict):
            return None
        trace_id = str(
            context.get("trace_id") or context.get("traceId") or ""
        ).strip().lower()
        if trace_id:
            return trace_id
        return _trace_id_from_traceparent(context.get("traceparent"))
    except Exception:
        return None


def _store_runtime_workflow_context(instance_id: object) -> None:
    try:
        from dapr_agents.observability.context_propagation import extract_otel_context
        from dapr_agents.observability.context_storage import store_workflow_context

        key = _workflow_context_key(instance_id)
        if not key:
            return
        context = extract_otel_context()
        if not isinstance(context, dict) or not context.get("traceparent"):
            return
        store_workflow_context(key, context)
        store_workflow_context("__current_workflow_context__", context)
    except Exception:
        logger.debug("Failed to store runtime workflow context", exc_info=True)


class OpenShellDurableAgent(DurableAgent):
    """DurableAgent with automatic sandbox targeting and repo clone injection."""

    def agent_workflow(self, ctx, message: dict):
        if not ctx.is_replaying:
            logger.info(
                "OpenShellDurableAgent.agent_workflow keys=%s sandboxName=%r repoUrl=%r repoBranch=%r",
                sorted(message.keys()),
                message.get("sandboxName"),
                message.get("repoUrl"),
                message.get("repoBranch"),
            )
        _store_runtime_workflow_context(getattr(ctx, "instance_id", None))
        sandbox_name = (message.get("sandboxName") or "").strip()
        repo_url = (message.get("repoUrl") or "").strip()
        repo_branch = (message.get("repoBranch") or "").strip()
        repo_token = (message.get("repoToken") or "").strip()
        sandbox_repo_path = (message.get("sandboxRepoPath") or "/sandbox/repo").strip()

        # Target the orchestrator-assigned sandbox (only set env var if not
        # already pointing to a real sandbox — avoids resetting the session
        # on Dapr workflow replays after a sandbox was already created).
        if sandbox_name:
            current = os.environ.get("OPENSHELL_SANDBOX_NAME", "")
            if not current:
                os.environ["OPENSHELL_SANDBOX_NAME"] = sandbox_name

        # Prepend clone instructions to the task.
        # After sandbox creation, apply a network policy that allows Gitea
        # access so git clone can reach the in-cluster mirror.
        if repo_url:
            task = message.get("task") or ""
            url = repo_url
            if repo_token and url.startswith("https://"):
                url = url.replace("https://", f"https://oauth2:{repo_token}@", 1)
            clone_cmd = "GIT_SSL_NO_VERIFY=true git clone --depth 1"
            if repo_branch:
                clone_cmd += f" -b {repo_branch}"
            clone_cmd += f" {url} {sandbox_repo_path}"

            message = {**message, "task": (
                f"SETUP: First, clone the repository:\n"
                f"  {clone_cmd}\n"
                f"Then cd into {sandbox_repo_path} and proceed with:\n\n{task}"
            )}

        final_message = yield from super().agent_workflow(ctx, message)

        content = ""
        if isinstance(final_message, dict):
            content = str(final_message.get("content") or "").strip()
        elif final_message is not None:
            content = str(final_message).strip()

        workflow_id = str(getattr(ctx, "instance_id", "") or "").strip()
        trace_id = _trace_id_from_workflow_context(workflow_id)
        if not trace_id:
            logger.warning(
                "No child workflow trace context found for %s; leaving traceId unset",
                workflow_id or "<unknown>",
            )
        updated_at = datetime.now().isoformat()

        agent_progress = {
            "status": "completed",
            "phase": str(message.get("mode") or "execute_direct"),
            "summary": content or None,
            "currentStepName": message.get("sandboxName") or sandbox_name or None,
            "activeToolName": None,
            "stopReason": "workflow completed",
            "agentWorkflowId": workflow_id or None,
            "daprInstanceId": workflow_id or None,
            "traceId": trace_id,
            "updatedAt": updated_at,
            "recentTurns": [],
        }

        return {
            "success": True,
            "content": content,
            "text": content,
            "final_answer": content,
            "assistantMessage": (
                final_message if isinstance(final_message, dict) else None
            ),
            "traceId": trace_id,
            "agentProgress": agent_progress,
            "agentWorkflowId": workflow_id or None,
            "daprInstanceId": workflow_id or None,
            "sandboxName": message.get("sandboxName") or sandbox_name or None,
            "provider": message.get("provider"),
            "threadId": message.get("threadId"),
            "planningThreadId": message.get("planningThreadId"),
            "executionThreadId": message.get("executionThreadId"),
        }


def create_durable_agent() -> OpenShellDurableAgent:
    """Create the native Dapr DurableAgent instance."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    agent_name = os.getenv("AGENT_NAME", "OpenShellDeepAgent")
    llm_component = os.getenv("DAPR_LLM_COMPONENT", DEFAULT_LLM_COMPONENT)
    memory_store = os.getenv("DAPR_MEMORY_STORE", DEFAULT_MEMORY_STORE)
    workflow_store = os.getenv("DAPR_WORKFLOW_STORE", DEFAULT_WORKFLOW_STORE)

    # OTEL observability — rely on env vars only (validated by vanilla-durable-agent).
    # Explicit AgentObservabilityConfig was creating a second TracerProvider that
    # conflicted with the env-var-based auto-instrumentation. Env vars are set via
    # ConfigMap openshell-durable-agent-otel-config.
    logger.info(
        "OTEL config: env-var mode (OTEL_SERVICE_NAME=%s, OTEL_EXPORTER_OTLP_ENDPOINT=%s)",
        os.getenv("OTEL_SERVICE_NAME", "not-set"),
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "not-set"),
    )

    return OpenShellDurableAgent(
        name=agent_name,
        role="OpenShell Coding Agent",
        goal="Help users inspect, modify, and execute code safely inside an OpenShell sandbox.",
        system_prompt=build_system_prompt(current_date),
        tools=DURABLE_TOOLS,
        llm=DaprChatClient(component_name=llm_component),
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(store_name=memory_store),
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name=workflow_store),
        ),
        # NO agent_observability — env vars handle OTEL setup (demo-otel-k8s/envvars pattern)
    )


def main() -> None:
    """Serve the native durable agent over HTTP for the Dapr sidecar."""
    port = int(os.getenv("APP_PORT", str(DEFAULT_APP_PORT)))
    agent = create_durable_agent()

    runner = AgentRunner()
    try:
        runner.serve(agent, port=port)
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down durable agent")
