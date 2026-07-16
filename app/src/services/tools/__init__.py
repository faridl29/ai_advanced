"""Tool registry — single source of truth for all agent capabilities.

Architecture:
- Each tool is a module exposing LangChain `@tool`-decorated functions
- `TOOL_REGISTRY` maps name → (callable, description, category)
- `get_tools(names=None)` returns the resolved list, filtered by name
- Tools can be added/removed without touching agent graph code

Adding a new tool:
1. Create a function with `@tool` decorator in the appropriate category module
2. Register it in `TOOL_REGISTRY` below
3. Add to default `settings.tools_enabled` list

Categories:
- math: calculator, python_executor
- knowledge: knowledge_base (RAG)
- time: current_datetime
- web: web_search
- reasoning: think (scratchpad)
"""
from __future__ import annotations

import logging
from typing import Callable

from langchain_core.tools import BaseTool

from src.core.config import get_settings

logger = logging.getLogger(__name__)


# Import tool modules so @tool decorators register the callables
from src.services.tools.math_tools import calculator, python_executor
from src.services.tools.knowledge_tools import knowledge_base
from src.services.tools.time_tools import current_datetime
from src.services.tools.web_tools import web_search
from src.services.tools.reasoning_tools import think
from src.services.tools.financial_tools import financial_analyzer
from src.services.tools.report_tools import generate_excel_report

# Registry: name -> (tool_callable, category, description_short)
TOOL_REGISTRY: dict[str, tuple[BaseTool, str, str]] = {
    "calculator": (calculator, "math", "Evaluate math expressions"),
    "python_executor": (python_executor, "math", "Run sandboxed Python for complex logic"),
    "knowledge_base": (knowledge_base, "knowledge", "Search uploaded documents (RAG)"),
    "current_datetime": (current_datetime, "time", "Get current date/time/timezone"),
    "web_search": (web_search, "web", "Search the web (stub: returns guidance)"),
    "think": (think, "reasoning", "Internal scratchpad for chain-of-thought"),
    "financial_analyzer": (financial_analyzer, "finance", "Compute financial ratios and investment assessment"),
    "generate_excel_report": (generate_excel_report, "finance", "Generate downloadable Excel report"),
}


def get_tools(names: list[str] | None = None) -> list[BaseTool]:
    """Resolve tool names to LangChain tool callables.

    Args:
        names: Optional whitelist. If None, uses settings.tools_enabled.
               If a name is unknown, it's logged and skipped.

    Returns:
        List of LangChain tool objects ready to bind to LLM.
    """
    if names is None:
        names = get_settings().tools_enabled

    resolved: list[BaseTool] = []
    for name in names:
        if name not in TOOL_REGISTRY:
            logger.warning(f"Unknown tool '{name}', skipping")
            continue
        tool_obj, _category, _desc = TOOL_REGISTRY[name]
        resolved.append(tool_obj)

    if not resolved:
        logger.warning("No tools resolved — agent will be a pure LLM")

    logger.info(f"Resolved {len(resolved)} tools: {[t.name for t in resolved]}")
    return resolved


def get_tool_descriptions() -> list[dict]:
    """Return metadata for all tools (for UI/tool picker)."""
    return [
        {"name": name, "category": cat, "description": desc}
        for name, (_tool, cat, desc) in TOOL_REGISTRY.items()
    ]


__all__ = ["TOOL_REGISTRY", "get_tools", "get_tool_descriptions"]
