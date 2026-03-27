"""Native Dapr DurableAgent runtime for the OpenShell agent."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime

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

DEFAULT_APP_PORT = 8000
DEFAULT_LLM_COMPONENT = "llm-provider"
DEFAULT_MEMORY_STORE = "agent-memory"
DEFAULT_WORKFLOW_STORE = "agent-workflow"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

load_dotenv()

# In-cluster Gitea URL for repos mirrored by function-router.
# Use token auth (oauth2:<token>@) — basic auth (user:pass@) triggers 403 from
# git-http-backend when the sandbox's HTTP client doesn't send credentials on
# the initial challenge-less request.
GITEA_INTERNAL_BASE = os.getenv(
    "GITEA_INTERNAL_BASE_URL",
    "http://gitea-http.gitea.svc.cluster.local:3000",
)
GITEA_CLONE_OWNER = os.getenv("GITEA_CLONE_OWNER", "giteaadmin")
GITEA_CLONE_TOKEN = os.getenv(
    "GITEA_CLONE_TOKEN", "22aa40ad26e328745563fd5c838a3b227c980a07"
)

# Pattern: https://github.com/<owner>/<repo>[.git]
_GITHUB_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?/?$"
)


def _rewrite_github_to_gitea(url: str) -> str:
    """Rewrite a GitHub HTTPS URL to the in-cluster Gitea mirror.

    Sandboxes can't reach github.com (SSL/network), but the function-router
    mirrors repos into Gitea under ``giteaadmin/<repo>``.
    """
    m = _GITHUB_URL_RE.match(url)
    if not m:
        return url
    repo = m.group("repo")
    base = GITEA_INTERNAL_BASE.rstrip("/")
    if GITEA_CLONE_TOKEN:
        base = base.replace("://", f"://oauth2:{GITEA_CLONE_TOKEN}@", 1)
    return f"{base}/{GITEA_CLONE_OWNER}/{repo}.git"


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

        # Prepend clone instructions to the task
        if repo_url:
            task = message.get("task") or ""
            # Rewrite GitHub URLs to in-cluster Gitea mirror
            url = _rewrite_github_to_gitea(repo_url)
            if url == repo_url and repo_token and repo_url.startswith("https://"):
                # Not a GitHub URL but has token — inject auth
                url = repo_url.replace(
                    "https://", f"https://oauth2:{repo_token}@", 1
                )
            clone_cmd = "git clone --depth 1"
            if repo_branch:
                clone_cmd += f" -b {repo_branch}"
            clone_cmd += f" {url} {sandbox_repo_path}"

            message = {**message, "task": (
                f"SETUP: First, clone the repository:\n"
                f"  {clone_cmd}\n"
                f"Then cd into {sandbox_repo_path} and proceed with:\n\n{task}"
            )}

        yield from super().agent_workflow(ctx, message)


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
