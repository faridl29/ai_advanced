"""Web tools — web search (currently stub).

In production, integrate with SearXNG or similar local meta-search engine.
Configuration is via `SEARXNG_URL` env var (not yet added to Settings).
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current information.
    Use when asked about recent events, current news, or information that
    is likely not in the local knowledge base and may have changed since
    the model's training cutoff.

    Note: This is currently a STUB. To enable, deploy SearXNG and configure
    the SEARXNG_URL environment variable.
    """
    return (
        f"[Web Search Stub] Search for: '{query}'\n"
        "Note: Web search is not yet configured. To enable, deploy SearXNG "
        "and configure the SEARXNG_URL environment variable.\n"
        "For now, I'll answer based on my training knowledge."
    )


__all__ = ["web_search"]
