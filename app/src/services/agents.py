"""Agent service — production-grade full-agentic ReAct executor.

Architecture (post-refactor):
- Single brain: every request flows through the agent
- The agent decides autonomously which tools to call (RAG, calculator, etc.)
- Multi-tool sequences are first-class (e.g. RAG → calculator → answer)
- Streaming: emits typed events for tool start/end + content deltas
- Pluggable tool registry (see src/services/tools/)
- Pluggable memory backend (see src/services/memory.py)
- LangGraph checkpointer for short-term conversation memory

Backward compat:
- `run_agent(task, model, max_steps, history)` still works (legacy single-shot)
- `run_agent_stream()` is the new streaming API
- Graph is built once and cached
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Annotated, Any, AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.core.config import get_settings
from src.services.llm import get_llm
from src.services.memory import get_memory
from src.services.prompts import get_prompt
from src.services.tools import get_tools
from src.utils.text import strip_think as _strip_think

logger = logging.getLogger(__name__)


# =============================================================================
# STATE
# =============================================================================

class AgentState(TypedDict, total=False):
    """Mutable state for the agent graph."""
    messages: Annotated[list, add_messages]
    session_id: str
    tool_calls_count: int
    error_count: int
    reasoning_steps: list[dict]  # observability trace


# =============================================================================
# PROMPTS
# =============================================================================

AGENT_SYSTEM_FALLBACK = """You are a helpful AI assistant with access to tools. Use them to provide accurate answers.

You have access to these tools:
- calculator: For math operations (expressions like "2+2", "sqrt(144)")
- python_executor: For complex calculations, data processing, or string manipulation
- knowledge_base: To search uploaded documents and internal knowledge
- web_search: To search the web for current information
- current_datetime: For current date/time info
- think: Your private scratchpad for chain-of-thought (NOT shown to user)
- financial_analyzer: Compute financial ratios (ROE, ROA, DER, etc.) and investment assessment from JSON data
- generate_excel_report: Generate a downloadable Excel (.xlsx) report from financial analysis results

DECISION RULES:
1. For math questions → ALWAYS use calculator or python_executor
2. For questions about documents/files/company data → use knowledge_base FIRST
3. For date/time questions → use current_datetime
4. For recent events → use web_search
5. For complex multi-step tasks → use think first to plan, then call other tools
6. You can call MULTIPLE tools in sequence (e.g. knowledge_base → financial_analyzer → generate_excel_report)
7. If a tool returns an error, try a different tool or answer from your own knowledge
8. After all needed tool calls, give a clear final answer in the user's language

FINANCIAL ANALYSIS RULES:
9. When user asks about financial analysis, ratios, or investment assessment:
   a. FIRST use knowledge_base to retrieve the financial data from uploaded documents
   b. THEN extract key figures (revenue, net_income, total_assets, total_equity, etc.)
   c. THEN call financial_analyzer with the extracted data as JSON
   d. If user asks for Excel/download, THEN call generate_excel_report with the analysis output
10. Always format financial data as markdown tables in your final answer
11. When presenting financial figures, use proper number formatting (e.g. Rp 500.000.000 or 500M)

Answer in the same language as the user's question. Be concise but thorough."""


def _load_agent_prompt() -> str:
    return get_prompt("agent-system", AGENT_SYSTEM_FALLBACK)


# =============================================================================
# LLM CONFIG
# =============================================================================

def _get_agent_llm(model: str | None = None, streaming: bool = False):
    s = get_settings()
    return get_llm(
        "chat",
        model=model or s.agent_model,
        temperature=s.agent_temperature,
        max_tokens=s.agent_max_tokens,
        streaming=streaming,
    )


# =============================================================================
# GRAPH NODES
# =============================================================================

def _should_continue(state: AgentState) -> str:
    """Decide next node: tools (call) / reflect (verify) / END."""
    s = get_settings()
    messages = state.get("messages", [])

    # Guard: hard cap on tool calls
    if state.get("tool_calls_count", 0) >= s.agent_max_steps:
        logger.info(f"Max steps reached ({s.agent_max_steps}), ending")
        return END

    # Guard: too many errors → bail
    if state.get("error_count", 0) >= 3:
        logger.warning("Too many errors, ending")
        return END

    last = messages[-1] if messages else None
    if last is None:
        return END

    # Has tool_calls → go to tools node
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # Optional: reflect node (only valid if it was registered in build_agent_graph)
    if s.agent_reflect:
        return "reflect"
    return END


async def _call_model(state: AgentState) -> dict:
    """Call the LLM with tools bound. Async for streaming support."""
    s = get_settings()
    llm = _get_agent_llm(streaming=False)
    tools = get_tools(s.tools_enabled)
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    try:
        response = await llm_with_tools.ainvoke(state["messages"])

        tool_count = state.get("tool_calls_count", 0)
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_count += len(response.tool_calls)

        # Track reasoning steps for observability
        steps = list(state.get("reasoning_steps", []))
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                steps.append({
                    "type": "tool_call",
                    "tool": tc["name"],
                    "args": tc.get("args", {}),
                })

        return {
            "messages": [response],
            "tool_calls_count": tool_count,
            "reasoning_steps": steps,
        }
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        error_count = state.get("error_count", 0) + 1
        try:
            # Retry without tools
            response = await _get_agent_llm().ainvoke(state["messages"])
            return {"messages": [response], "error_count": error_count}
        except Exception as e2:
            err = AIMessage(content=f"I encountered an error: {e2}")
            return {"messages": [err], "error_count": error_count}


def _track_tool_results(state: AgentState) -> dict:
    """Append tool results to reasoning_steps (called after ToolNode)."""
    messages = state.get("messages", [])
    steps = list(state.get("reasoning_steps", []))
    for m in messages:
        # ToolMessage is the result of a tool call
        from langchain_core.messages import ToolMessage
        if isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            steps.append({
                "type": "tool_result",
                "tool": m.name,
                "content_preview": content[:200],
            })
    return {"reasoning_steps": steps}


def _reflect(state: AgentState) -> dict:
    """Optional: verify the final answer. Currently a no-op stub."""
    # Future: use a lightweight LLM call to check groundedness / quality
    return {}


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_agent_graph() -> Any:
    """Compile the agent graph. Cached as singleton."""
    s = get_settings()
    graph = StateGraph(AgentState)

    graph.add_node("agent", _call_model)
    graph.add_node("tools", ToolNode(get_tools(s.tools_enabled)))
    if s.agent_reflect:
        graph.add_node("reflect", _reflect)
    # After tools run, record the results back into reasoning_steps
    graph.add_node("track_results", _track_tool_results)

    # Build the path map. If `agent_reflect` is disabled, the 'reflect'
    # branch must route to END because the node is not registered.
    path_map: dict[str, str] = {"tools": "tools", END: END}
    if s.agent_reflect:
        path_map["reflect"] = "reflect"
    else:
        path_map["reflect"] = END

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, path_map)
    graph.add_edge("tools", "track_results")
    graph.add_edge("track_results", "agent")
    if s.agent_reflect:
        graph.add_edge("reflect", END)

    return graph.compile()


# =============================================================================
# SINGLETON
# =============================================================================

_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def invalidate_agent_cache() -> None:
    """Force rebuild on next call (useful after config changes)."""
    global _agent_graph
    _agent_graph = None


# =============================================================================
# HISTORY MANAGEMENT
# =============================================================================

def _history_to_messages(history: list[dict]) -> list:
    """Convert [{"role", "content"}] to LangChain messages."""
    out: list = []
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
    return out


# =============================================================================
# LEGACY SINGLE-SHOT (backward compat)
# =============================================================================

async def run_agent(
    task: str,
    model: str | None = None,
    max_steps: int = 10,
    history: list[dict] | None = None,
    tools: list[str] | None = None,
) -> dict:
    """Run the agent and return a final dict. Used by /v1/agents/run.

    For new code, prefer `run_agent_stream` for observability.
    """
    s = get_settings()
    effective_max = max_steps or s.agent_max_steps
    graph = get_agent_graph()

    messages: list = [SystemMessage(content=_load_agent_prompt())]
    if history:
        messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=task))

    initial_state: AgentState = {
        "messages": messages,
        "session_id": "",
        "tool_calls_count": 0,
        "error_count": 0,
        "reasoning_steps": [],
    }

    try:
        result = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": effective_max * 3},
        )
        msgs = result.get("messages", [])
        final_answer = ""
        tools_used: list[str] = []
        for m in msgs:
            if isinstance(m, AIMessage):
                if m.content:
                    final_answer = m.content
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        if tc["name"] not in tools_used:
                            tools_used.append(tc["name"])

        if not final_answer and result.get("reasoning_steps"):
            last_results = [s for s in result["reasoning_steps"] if s.get("type") == "tool_result" and s.get("content_preview")]
            if last_results:
                final_answer = last_results[-1]["content_preview"]

        return {
            "status": "ok",
            "answer": _strip_think(final_answer),
            "tools_used": tools_used,
            "steps": len(result.get("reasoning_steps", [])),
            "reasoning": result.get("reasoning_steps", []),
            "task": task,
            "model": model or s.agent_model,
        }
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        return {
            "status": "error",
            "answer": f"Agent encountered an error: {e}. Please try again.",
            "tools_used": [],
            "steps": 0,
            "reasoning": [],
            "task": task,
            "error": str(e),
        }


# =============================================================================
# STREAMING API
# =============================================================================

async def run_agent_stream(
    task: str,
    session_id: str = "default",
    model: str | None = None,
    history: list[dict] | None = None,
    tools: list[str] | None = None,
) -> AsyncIterator[dict]:
    """Run the agent with full streaming observability.

    Event shapes:
      {"event": "metadata", "session_id": ..., "model": ..., "tools": [...]}
      {"event": "reasoning", "delta": "..."}     # from think tool
      {"event": "tool_start", "tool": "...", "args": {...}}
      {"event": "tool_end", "tool": "...", "output_preview": "..."}
      {"event": "content", "delta": "..."}        # final answer chunk
      {"event": "done", "answer": "...", "tools_used": [...], "latency_ms": ...}
      {"event": "error", "detail": "..."}
    """
    s = get_settings()
    start = time.perf_counter()
    effective_model = model or s.agent_model
    effective_tools = tools or s.tools_enabled

    yield {
        "event": "metadata",
        "session_id": session_id,
        "model": effective_model,
        "tools": effective_tools,
    }

    # Build messages with history
    messages: list = [SystemMessage(content=_load_agent_prompt())]
    if history:
        messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=task))

    graph = get_agent_graph()
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": s.agent_max_steps * 3}

    full_content = ""
    tools_used: list[str] = []
    reasoning_steps: list[dict] = []

    try:
        async for event in graph.astream_events(
            {"messages": messages, "session_id": session_id, "tool_calls_count": 0, "error_count": 0, "reasoning_steps": []},
            config=config,
            version="v2",
        ):
            ev_type = event.get("event")
            name = event.get("name", "")

            # LLM streaming tokens
            if ev_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and getattr(chunk, "content", None):
                    delta = chunk.content
                    if isinstance(delta, str) and delta:
                        # If model is in think block, route to reasoning
                        full_content += delta
                        yield {"event": "content", "delta": delta}

            # Tool start
            elif ev_type == "on_tool_start":
                tool_input = event.get("data", {}).get("input", {})
                if name not in tools_used:
                    tools_used.append(name)
                reasoning_steps.append({"type": "tool_call", "tool": name, "args": tool_input})
                yield {"event": "tool_start", "tool": name, "args": tool_input}

            # Tool end
            elif ev_type == "on_tool_end":
                output = event.get("data", {}).get("output")
                preview = str(output)[:200] if output is not None else ""
                reasoning_steps.append({"type": "tool_result", "tool": name, "content_preview": preview})
                yield {"event": "tool_end", "tool": name, "output_preview": preview}

        elapsed = (time.perf_counter() - start) * 1000
        final_answer = _strip_think(full_content)
        yield {
            "event": "done",
            "answer": final_answer,
            "tools_used": tools_used,
            "reasoning": reasoning_steps,
            "latency_ms": round(elapsed, 1),
            "model": effective_model,
        }

        # Persist to memory (fire-and-forget)
        try:
            mem = get_memory()
            await mem.add_message(session_id, "user", task)
            await mem.add_message(session_id, "assistant", final_answer)
        except Exception as e:
            logger.debug(f"Memory persist failed: {e}")

    except Exception as e:
        logger.exception("Agent stream failed")
        yield {"event": "error", "detail": str(e)}


__all__ = [
    "build_agent_graph",
    "get_agent_graph",
    "invalidate_agent_cache",
    "run_agent",
    "run_agent_stream",
]
