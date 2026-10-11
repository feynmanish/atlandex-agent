"""Load the atlandex-mcp tools over stdio via langchain-mcp-adapters."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Protocol

from langchain_core.utils.function_calling import convert_to_openai_tool

from .config import Settings


class ToolBox(Protocol):
    """Everything the graph needs from the tools. Tests provide a fake."""

    def openai_specs(self) -> list[dict[str, Any]]: ...

    async def call(self, name: str, args: dict[str, Any]) -> str: ...


def _as_text(result: Any) -> str:
    """Flatten whatever a LangChain MCP tool returns into a string."""
    if isinstance(result, str):
        return result
    if isinstance(result, tuple):  # (content, artifact)
        result = result[0]
    if isinstance(result, list):
        parts = []
        for b in result:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            else:
                parts.append(getattr(b, "text", "") or str(b))
        return "\n".join(p for p in parts if p)
    return getattr(result, "content", None) and _as_text(result.content) or str(result)


class McpToolBox:
    def __init__(self, tools: list[Any]):
        self._tools = {t.name: t for t in tools}

    def openai_specs(self) -> list[dict[str, Any]]:
        return [convert_to_openai_tool(t) for t in self._tools.values()]

    async def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self._tools)}."
        try:
            return _as_text(await tool.ainvoke(args))
        except Exception as e:  # tool-side validation or transport errors go back to the model
            return f"Error calling {name}: {str(e)[:500]}"


@asynccontextmanager
async def open_toolbox(settings: Settings) -> AsyncIterator[McpToolBox]:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    if not settings.atlandex_api_url:
        raise RuntimeError("ATLANDEX_API_URL is not set (see .env.example).")
    env = {**os.environ, "ATLANDEX_API_URL": settings.atlandex_api_url}
    client = MultiServerMCPClient({
        "atlandex": {
            "transport": "stdio",
            "command": settings.mcp_command,
            "args": [],
            "env": {k: v for k, v in env.items() if k != "OPENROUTER_API_KEY"},  # the MCP server doesn't need the LLM key
        }
    })
    async with client.session("atlandex") as session:
        yield McpToolBox(await load_mcp_tools(session))
