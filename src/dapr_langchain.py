"""LangChain adapter for Dapr's conversation/LLM component."""

from __future__ import annotations

import json
from typing import Any, Sequence

from dapr_agents.types.message import AssistantMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            parts.append(json.dumps(item, sort_keys=True))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _serialize_tool_call(tool_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": tool_data.get("id"),
        "type": "function",
        "function": {
            "name": tool_data["name"],
            "arguments": json.dumps(tool_data.get("args", {})),
        },
    }


def _message_to_dapr(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": _coerce_message_content(message.content)}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": _coerce_message_content(message.content)}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": _coerce_message_content(message.content),
            "tool_call_id": message.tool_call_id,
            "name": message.name,
        }
    if isinstance(message, AIMessage):
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": _coerce_message_content(message.content),
        }
        if message.tool_calls:
            payload["tool_calls"] = [_serialize_tool_call(call) for call in message.tool_calls]
        return payload
    raise TypeError(f"Unsupported message type: {type(message)!r}")


def _assistant_to_ai_message(message: AssistantMessage) -> AIMessage:
    tool_calls = []
    for call in message.get_tool_calls() or []:
        tool_calls.append(
            tool_call(
                name=call.function.name,
                args=call.function.arguments_dict,
                id=call.id,
            )
        )

    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
    )


class DaprLangChainChatModel(BaseChatModel):
    """Minimal LangChain chat-model adapter around `DaprChatClient`."""

    client: Any = Field(...)
    model_name: str = Field(default="dapr-conversation")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "dapr-conversation"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        dapr_messages = [_message_to_dapr(message) for message in messages]
        response = self.client.generate(
            messages=dapr_messages,
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            model=kwargs.get("model"),
        )

        if not response.results:
            raise ValueError("DaprChatClient returned no candidates")

        assistant_message = response.results[0].message
        ai_message = _assistant_to_ai_message(assistant_message)
        generation = ChatGeneration(
            message=ai_message,
            generation_info={"metadata": response.metadata},
        )
        return ChatResult(generations=[generation])
