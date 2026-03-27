"""LangGraph agent using Dapr-backed checkpoint persistence."""

from __future__ import annotations

from functools import lru_cache
import os
from datetime import datetime

from dapr.ext.langgraph import DaprCheckpointer
from dapr_agents import DaprChatClient
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.dapr_langchain import DaprLangChainChatModel
from src.openshell_tools import TOOLS
from src.prompts import build_system_prompt

DEFAULT_LLM_COMPONENT = "llm-provider"
DEFAULT_LANGGRAPH_STORE = "agent-workflow"
DEFAULT_LANGGRAPH_KEY_PREFIX = "openshell-deepagent"

def build_graph():
    """Build the LangGraph runtime with Dapr persistence."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    llm_component = os.getenv("DAPR_LLM_COMPONENT", DEFAULT_LLM_COMPONENT)
    store_name = os.getenv("DAPR_LANGGRAPH_STORE", DEFAULT_LANGGRAPH_STORE)
    key_prefix = os.getenv("DAPR_LANGGRAPH_KEY_PREFIX", DEFAULT_LANGGRAPH_KEY_PREFIX)

    llm = DaprLangChainChatModel(
        client=DaprChatClient(component_name=llm_component),
    )
    llm_with_tools = llm.bind_tools(TOOLS)
    system_message = SystemMessage(content=build_system_prompt(current_date))

    def assistant(state: MessagesState):
        return {
            "messages": [
                llm_with_tools.invoke([system_message] + state["messages"]),
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    checkpointer = DaprCheckpointer(
        store_name=store_name,
        key_prefix=key_prefix,
    )
    return builder.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_graph():
    """Lazily construct the compiled graph."""
    load_dotenv()
    return build_graph()


class LazyGraph:
    """Proxy that initializes the compiled graph only when it is first used."""

    def __getattr__(self, name: str):
        return getattr(get_graph(), name)

    def invoke(self, *args, **kwargs):
        return get_graph().invoke(*args, **kwargs)


graph = LazyGraph()
