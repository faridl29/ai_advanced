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
from src.services.memory import get_memory, get_profile_memory
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
    reflection_attempts: int


# =============================================================================
# PROMPTS
# =============================================================================

def _load_agent_prompt() -> str:
    return get_prompt("agent-system")


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


async def _reflect(state: AgentState) -> dict:
    """Verify the final answer using LLM-as-judge metrics (faithfulness, coherence)."""
    messages = state.get("messages", [])
    attempts = state.get("reflection_attempts", 0)
    
    # Extract original query
    query = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            # Skip reflection corrections/feedback messages when finding original query
            if "[Koreksi Refleksi Mandiri]" not in m.content:
                query = m.content
                break
            
    # Extract final answer
    response = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if not getattr(m, "tool_calls", None): # skip if it's a tool-calling intermediate message
                response = m.content
                break
                
    if not response or not query:
        logger.info("Skipping reflection: query or response not found")
        return {"reflection_attempts": attempts}

    # Extract retrieved context chunks from ToolMessages (for RAG faithfulness check)
    from langchain_core.messages import ToolMessage
    context = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "knowledge_base":
            context.append(m.content)

    logger.info(f"Running reflection checks for query: '{query[:50]}...' with {len(context)} context chunks (attempt {attempts + 1})")
    
    from src.services.evaluation import evaluate_response
    
    # If context is available, check faithfulness, otherwise check coherence
    metrics = ["coherence"]
    if context:
        metrics.append("faithfulness")
        
    try:
        eval_result = await evaluate_response(
            query=query,
            response=response,
            context=context if context else None,
            metrics=metrics,
        )
        
        overall_score = eval_result.get("overall_score", 1.0)
        passed = eval_result.get("passed", True)
        
        logger.info(f"Reflection score: {overall_score} (passed={passed})")
        
        if not passed and attempts < 1:
            # Find why it failed
            failed_metrics = []
            for metric, details in eval_result.get("metrics", {}).items():
                if details.get("score", 1.0) < 0.6:
                    failed_metrics.append(f"{metric} (score: {details.get('score')}, reason: {details.get('reason')})")
            
            feedback_reason = "; ".join(failed_metrics)
            feedback_msg = (
                f"[Koreksi Refleksi Mandiri] Jawaban Anda kurang memenuhi kriteria kualitas (Skor: {overall_score:.2f}). "
                f"Evaluasi kegagalan: {feedback_reason}. Harap periksa kembali dan perbaiki jawaban Anda agar lebih akurat dan terstruktur."
            )
            logger.info(f"Reflection failed. Appending feedback: {feedback_msg}")
            
            return {
                "messages": [HumanMessage(content=feedback_msg)],
                "reflection_attempts": attempts + 1,
            }
    except Exception as eval_err:
        logger.warning(f"Error during self-reflection evaluation: {eval_err}")
        
    return {"reflection_attempts": attempts}


def _should_finish(state: AgentState) -> str:
    """Determine if reflection passed or failed (and needs to loop back)."""
    messages = state.get("messages", [])
    
    # If the last message is a HumanMessage containing our correction prompt, route back to agent
    if messages and isinstance(messages[-1], HumanMessage) and "[Koreksi Refleksi Mandiri]" in messages[-1].content:
        return "agent"
    return END


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
        graph.add_conditional_edges("reflect", _should_finish, {"agent": "agent", END: END})

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
    session_id: str = "default",
) -> dict:
    """Run the agent and return a final dict. Used by /v1/agents/run.

    For new code, prefer `run_agent_stream` for observability.
    """
    s = get_settings()
    effective_max = max_steps or s.agent_max_steps
    graph = get_agent_graph()

    profile_mem = get_profile_memory()
    profile_context = await profile_mem.get_profile_context(session_id)
    system_prompt = _load_agent_prompt()
    if profile_context:
        system_prompt += f"\n\n{profile_context}"

    messages: list = [SystemMessage(content=system_prompt)]
    if history:
        messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=task))

    initial_state: AgentState = {
        "messages": messages,
        "session_id": session_id,
        "tool_calls_count": 0,
        "error_count": 0,
        "reasoning_steps": [],
        "reflection_attempts": 0,
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

        # Persist to memory and trigger profile update
        try:
            mem = get_memory()
            await mem.add_message(session_id, "user", task)
            await mem.add_message(session_id, "assistant", final_answer)
            import asyncio
            asyncio.create_task(profile_mem.extract_and_update_profile(session_id, task, final_answer))
        except Exception as mem_err:
            logger.debug(f"Memory update failed in run_agent: {mem_err}")

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

    # Load profile memory
    profile_mem = get_profile_memory()
    profile_context = await profile_mem.get_profile_context(session_id)
    system_prompt = _load_agent_prompt()
    if profile_context:
        system_prompt += f"\n\n{profile_context}"

    # Build messages with history
    messages: list = [SystemMessage(content=system_prompt)]
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
            {
                "messages": messages,
                "session_id": session_id,
                "tool_calls_count": 0,
                "error_count": 0,
                "reasoning_steps": [],
                "reflection_attempts": 0,
            },
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
            import asyncio
            asyncio.create_task(profile_mem.extract_and_update_profile(session_id, task, final_answer))
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
