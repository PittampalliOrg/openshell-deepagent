"""Runtime patches for dapr_agents observability gaps used by OpenShell."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def patch_workflow_monitor_wrapper() -> None:
    """Store per-instance workflow trace context for monitored durable runs."""
    try:
        from dapr_agents.observability.context_propagation import extract_otel_context
        from dapr_agents.observability.context_storage import store_workflow_context
        from dapr_agents.observability.wrappers import workflow as workflow_wrappers
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import dapr_agents observability wrappers: %s", exc)
        return

    wrapper_cls = workflow_wrappers.WorkflowMonitorWrapper
    if getattr(wrapper_cls, "_openshell_context_patch", False):
        return

    def _store_context(instance_id: str) -> None:
        if not instance_id:
            return
        context = extract_otel_context()
        if not context:
            return
        store_workflow_context(f"__workflow_context_{instance_id}__", context)
        store_workflow_context("__current_workflow_context__", context)

    def patched_call(self: Any, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        context_api = workflow_wrappers.context_api
        if context_api:
            if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
                bound = wrapped.__get__(instance, type(instance))
                return bound(*args, **kwargs)
            if context_api.get_value(workflow_wrappers.WORKFLOW_RUN_SUPPRESSION_KEY):
                bound = wrapped.__get__(instance, type(instance))
                return bound(*args, **kwargs)

        arguments = workflow_wrappers.bind_arguments(wrapped, instance, *args, **kwargs)
        detach = bool(arguments.get("detach"))
        if detach:
            bound = wrapped.__get__(instance, type(instance))
            return bound(*args, **kwargs)

        workflow = arguments.get("workflow")
        workflow_name = workflow_wrappers.WorkflowRunWrapper._infer_workflow_name(workflow)
        agent_details = workflow_wrappers._extract_agent_metadata(workflow)
        agent_name = agent_details.get(
            "agent.name", getattr(instance, "name", instance.__class__.__name__)
        )
        span_name = f"invoke_agent {agent_name}"
        payload = workflow_wrappers._resolve_payload(arguments, args)

        attributes = {
            workflow_wrappers.OPENINFERENCE_SPAN_KIND: workflow_wrappers.AGENT,
            "workflow.name": workflow_name,
            "workflow.operation": "run_and_wait",
            "agent.name": agent_name,
            workflow_wrappers.INPUT_MIME_TYPE: "application/json",
            workflow_wrappers.INPUT_VALUE: workflow_wrappers.safe_json_dumps(
                payload or {}
            ),
            workflow_wrappers.GEN_AI_OPERATION_NAME: (
                workflow_wrappers.GenAiOperationNameValues.INVOKE_AGENT
            ),
            workflow_wrappers.GEN_AI_AGENT_NAME: agent_name,
        }
        for key in ("agent.role", "agent.goal", "agent.execution.max_iterations"):
            if key in agent_details and agent_details[key] is not None:
                attributes[key] = agent_details[key]
        attributes.update(workflow_wrappers.get_attributes_from_context())

        suppress_token = None
        if context_api:
            suppress_token = context_api.attach(
                context_api.set_value(
                    workflow_wrappers.WORKFLOW_RUN_SUPPRESSION_KEY, True
                )
            )

        async def _async_call():
            bound = wrapped.__get__(instance, type(instance))
            with self._tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    result = await bound(*args, **kwargs)
                    if result:
                        span.set_attribute("workflow.instance_id", result)
                        _store_context(result)
                    span.set_attribute(
                        workflow_wrappers.OUTPUT_VALUE,
                        workflow_wrappers.safe_json_dumps(
                            workflow_wrappers._normalize_output(result)
                        ),
                    )
                    span.set_status(
                        workflow_wrappers.Status(workflow_wrappers.StatusCode.OK)
                    )
                    return result
                except Exception as exc:  # noqa: BLE001
                    span.set_status(
                        workflow_wrappers.Status(
                            workflow_wrappers.StatusCode.ERROR, str(exc)
                        )
                    )
                    span.set_attribute("error.type", type(exc).__qualname__)
                    span.record_exception(exc)
                    raise

        try:
            if asyncio.iscoroutinefunction(wrapped):
                return _async_call()

            bound = wrapped.__get__(instance, type(instance))
            with self._tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    result = bound(*args, **kwargs)
                    if result:
                        span.set_attribute("workflow.instance_id", result)
                        _store_context(result)
                    span.set_attribute(
                        workflow_wrappers.OUTPUT_VALUE,
                        workflow_wrappers.safe_json_dumps(
                            workflow_wrappers._normalize_output(result)
                        ),
                    )
                    span.set_status(
                        workflow_wrappers.Status(workflow_wrappers.StatusCode.OK)
                    )
                    return result
                except Exception as exc:  # noqa: BLE001
                    span.set_status(
                        workflow_wrappers.Status(
                            workflow_wrappers.StatusCode.ERROR, str(exc)
                        )
                    )
                    span.set_attribute("error.type", type(exc).__qualname__)
                    span.record_exception(exc)
                    raise
        finally:
            if suppress_token is not None and context_api:
                context_api.detach(suppress_token)

    wrapper_cls.__call__ = patched_call
    wrapper_cls._openshell_context_patch = True
    logger.info("Patched WorkflowMonitorWrapper to store per-instance context")
