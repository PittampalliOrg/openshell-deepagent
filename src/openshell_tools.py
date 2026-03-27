"""Framework-neutral OpenShell tool functions.

TOOLS exports plain functions (used by LangGraph via @langchain_tool binding).
DURABLE_TOOLS exports dapr_agents.AgentTool instances (used by DurableAgent).
"""

from __future__ import annotations

from typing import Any

from dapr_agents import tool as dapr_tool

from src.openshell_runtime import get_runtime


def execute_command(command: str, timeout_seconds: int = 1800) -> dict[str, Any]:
    """Execute a shell command inside the OpenShell sandbox."""
    return get_runtime().execute(command, timeout_seconds=timeout_seconds)


def read_file(path: str) -> dict[str, Any]:
    """Read a UTF-8 text file from the OpenShell sandbox."""
    return get_runtime().read_file(path)


def write_file(path: str, content: str, append: bool = False) -> dict[str, Any]:
    """Write a UTF-8 text file inside the OpenShell sandbox."""
    return get_runtime().write_file(path, content, append=append)


def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> dict[str, Any]:
    """Replace text inside a UTF-8 file in the OpenShell sandbox."""
    return get_runtime().edit_file(
        path,
        old_text,
        new_text,
        replace_all=replace_all,
    )


def list_files(path: str = ".", recursive: bool = False, max_entries: int = 200) -> dict[str, Any]:
    """List files or directories inside the OpenShell sandbox."""
    return get_runtime().list_files(path=path, recursive=recursive, max_entries=max_entries)


def glob_files(pattern: str, root: str = ".", max_entries: int = 200) -> dict[str, Any]:
    """Return files matching a glob pattern inside the OpenShell sandbox."""
    return get_runtime().glob_files(pattern=pattern, root=root, max_entries=max_entries)


def grep_files(
    pattern: str,
    root: str = ".",
    glob_pattern: str = "*",
    max_matches: int = 200,
) -> dict[str, Any]:
    """Search sandbox files for lines matching a regular expression."""
    return get_runtime().grep_files(
        pattern=pattern,
        root=root,
        glob_pattern=glob_pattern,
        max_matches=max_matches,
    )


# Plain functions for LangGraph (bound via langchain @tool in the graph module)
TOOLS = [
    execute_command,
    read_file,
    write_file,
    edit_file,
    list_files,
    glob_files,
    grep_files,
]


# dapr_agents AgentTool instances for DurableAgent
@dapr_tool
def _execute_command(command: str, timeout_seconds: int = 1800) -> dict[str, Any]:
    """Execute a shell command inside the OpenShell sandbox."""
    return get_runtime().execute(command, timeout_seconds=timeout_seconds)


@dapr_tool
def _read_file(path: str) -> dict[str, Any]:
    """Read a UTF-8 text file from the OpenShell sandbox."""
    return get_runtime().read_file(path)


@dapr_tool
def _write_file(path: str, content: str, append: bool = False) -> dict[str, Any]:
    """Write a UTF-8 text file inside the OpenShell sandbox."""
    return get_runtime().write_file(path, content, append=append)


@dapr_tool
def _edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> dict[str, Any]:
    """Replace text inside a UTF-8 file in the OpenShell sandbox."""
    return get_runtime().edit_file(path, old_text, new_text, replace_all=replace_all)


@dapr_tool
def _list_files(path: str = ".", recursive: bool = False, max_entries: int = 200) -> dict[str, Any]:
    """List files or directories inside the OpenShell sandbox."""
    return get_runtime().list_files(path=path, recursive=recursive, max_entries=max_entries)


@dapr_tool
def _glob_files(pattern: str, root: str = ".", max_entries: int = 200) -> dict[str, Any]:
    """Return files matching a glob pattern inside the OpenShell sandbox."""
    return get_runtime().glob_files(pattern=pattern, root=root, max_entries=max_entries)


@dapr_tool
def _grep_files(
    pattern: str,
    root: str = ".",
    glob_pattern: str = "*",
    max_matches: int = 200,
) -> dict[str, Any]:
    """Search sandbox files for lines matching a regular expression."""
    return get_runtime().grep_files(
        pattern=pattern, root=root, glob_pattern=glob_pattern, max_matches=max_matches,
    )


DURABLE_TOOLS = [
    _execute_command,
    _read_file,
    _write_file,
    _edit_file,
    _list_files,
    _glob_files,
    _grep_files,
]
