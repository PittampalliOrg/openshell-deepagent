"""Native Dapr DurableAgent runtime for the OpenShell agent."""

from __future__ import annotations

import logging
import os
from datetime import datetime

from dapr_agents import AgentRunner, DaprChatClient, DurableAgent
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentObservabilityConfig,
    AgentStateConfig,
    AgentTracingExporter,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dotenv import load_dotenv

from src.openshell_tools import DURABLE_TOOLS
from src.prompts import build_system_prompt

DEFAULT_APP_PORT = 8000
DEFAULT_LLM_COMPONENT = "llm-provider"
DEFAULT_MEMORY_STORE = "agent-memory"
DEFAULT_WORKFLOW_STORE = "agent-workflow"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

load_dotenv()


def create_durable_agent() -> DurableAgent:
    """Create the native Dapr DurableAgent instance."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    agent_name = os.getenv("AGENT_NAME", "OpenShellDeepAgent")
    llm_component = os.getenv("DAPR_LLM_COMPONENT", DEFAULT_LLM_COMPONENT)
    memory_store = os.getenv("DAPR_MEMORY_STORE", DEFAULT_MEMORY_STORE)
    workflow_store = os.getenv("DAPR_WORKFLOW_STORE", DEFAULT_WORKFLOW_STORE)

    # OTEL observability — follows the official demo-otel-k8s/instantiation pattern
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://otel-collector.observability.svc.cluster.local:4317",
    )
    service_name = os.getenv("OTEL_SERVICE_NAME", "openshell-durable-agent")

    observability = AgentObservabilityConfig(
        enabled=True,
        tracing_enabled=True,
        tracing_exporter=AgentTracingExporter.OTLP_HTTP,
        endpoint=endpoint,
        service_name=service_name,
    )
    logger.info("OTEL config: endpoint=%s service=%s", endpoint, service_name)

    return DurableAgent(
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
        agent_observability=observability,
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
