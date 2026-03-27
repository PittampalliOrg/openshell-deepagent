from __future__ import annotations

import unittest
from types import SimpleNamespace

from dapr_agents.types.message import AssistantMessage, FunctionCall, ToolCall
from langchain_core.messages import HumanMessage

from src.dapr_langchain import DaprLangChainChatModel


def sample_tool(value: str) -> str:
    """Echo a value."""
    return value


class FakeClient:
    def __init__(self) -> None:
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        message = AssistantMessage(
            content="tool response",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    type="function",
                    function=FunctionCall(
                        name="sample_tool",
                        arguments='{"value":"ok"}',
                    ),
                )
            ],
        )
        return SimpleNamespace(
            results=[SimpleNamespace(message=message)],
            metadata={"provider": "dapr"},
        )


class DaprLangChainChatModelTests(unittest.TestCase):
    def test_invoke_converts_tool_calls(self) -> None:
        client = FakeClient()
        model = DaprLangChainChatModel(client=client)

        result = model.invoke([HumanMessage(content="hello")])

        self.assertEqual(result.content, "tool response")
        self.assertEqual(result.tool_calls[0]["name"], "sample_tool")
        self.assertEqual(result.tool_calls[0]["args"], {"value": "ok"})

    def test_bind_tools_passes_openai_schemas(self) -> None:
        client = FakeClient()
        model = DaprLangChainChatModel(client=client)

        bound = model.bind_tools([sample_tool])
        bound.invoke([HumanMessage(content="hello")])

        tools = client.last_kwargs["tools"]
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "sample_tool")


if __name__ == "__main__":
    unittest.main()
