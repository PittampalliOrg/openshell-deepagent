# OpenShell Dapr Agent

This repo now ships two OpenShell-based Dapr runtimes that share the same
OpenShell sandbox tool layer:

- `durable-agent`: native [`dapr-agents`](https://docs.dapr.io/developing-ai/dapr-agents/)
  `DurableAgent` runtime
- `langgraph-dapr`: LangGraph runtime using
  [`dapr-ext-langgraph`](https://github.com/dapr/python-sdk/tree/master/examples/langgraph-checkpointer)
  for checkpoint persistence

Both runtimes execute commands and manage files inside an NVIDIA OpenShell
sandbox instead of running code directly in the app container.

## Runtime Layout

- [`src/openshell_runtime.py`](src/openshell_runtime.py): shared OpenShell session
  manager
- [`src/openshell_tools.py`](src/openshell_tools.py): shared command and file tools
- [`src/dapr_durable_agent.py`](src/dapr_durable_agent.py): native Dapr durable agent
- [`src/langgraph_dapr_graph.py`](src/langgraph_dapr_graph.py): LangGraph graph with
  `DaprCheckpointer`
- [`src/langgraph_dapr_app.py`](src/langgraph_dapr_app.py): HTTP wrapper for the
  LangGraph runtime

## Environment

The app expects Dapr components to be provided by the deployment environment.
Defaults are configurable with environment variables:

```bash
DAPR_LLM_COMPONENT=llm-provider
DAPR_MEMORY_STORE=agent-memory
DAPR_WORKFLOW_STORE=agent-workflow
DAPR_LANGGRAPH_STORE=agent-workflow
DAPR_LANGGRAPH_KEY_PREFIX=openshell-deepagent
OPENSHELL_SANDBOX_NAME=deepagent-sandbox
```

OpenShell gateway selection still follows the OpenShell CLI defaults and active
cluster configuration.

## Local Setup

```bash
uv sync
```

If you use a `.env` file, place the Dapr/OpenShell variables there before
starting either runtime.

## Run Locally

Native durable agent:

```bash
uv run python -m src.dapr_durable_agent
```

LangGraph + Dapr checkpointer:

```bash
uv run python -m src.langgraph_dapr_app
```

You can also load the LangGraph graph definition through `langgraph.json`.

## Build Images

Build the native Dapr durable-agent image:

```bash
docker build --target durable-agent -t openshell-durable-agent:local .
```

Build the LangGraph + Dapr image:

```bash
docker build --target langgraph-dapr -t openshell-langgraph-dapr:local .
```

## Kubernetes Deployment

This repo only produces application images. Kubernetes manifests, Dapr
components, secrets, and namespace wiring live in the separate `stacks/main`
repo.
