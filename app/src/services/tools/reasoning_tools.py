"""Reasoning tools — internal scratchpad for chain-of-thought.

`think` is a no-op tool that lets the agent write down its reasoning
without producing a user-visible response. This is useful for:
- Breaking down complex multi-step problems
- Recording intermediate state before calling other tools
- Self-checking before producing a final answer

The tool returns an empty string to the LLM (so it doesn't pollute context)
but the reasoning text is captured in the trace for observability.
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def think(reasoning: str) -> str:
    """Use this tool to think through a problem step-by-step.
    Write out your reasoning, then call other tools based on your analysis.
    This is a scratchpad — your 'reasoning' argument is NOT shown to the user,
    only logged for debugging.

    Examples of when to use:
    - Before a multi-step calculation, lay out the steps
    - When deciding which tool to call, explain your choice
    - After getting a tool result, summarize before the next step
    - When checking your own work for errors

    Do NOT use this to communicate with the user — use the final assistant
    message for that.
    """
    # Return empty — reasoning is captured in trace, not in LLM context
    return ""


__all__ = ["think"]
