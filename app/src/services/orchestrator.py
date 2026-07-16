"""Orchestrator service — thin wrapper for the full-agentic agent.

Architecture (post-refactor):
- Every request goes through the agent (src.services.agents)
- The agent decides which tools to call (RAG, calculator, datetime, etc.)
- This module handles pre/post processing only:
  1. Input guardrails (PII, injection, length)
  2. Optional fast-path: skip agent for trivial requests
  3. Delegate to agent (sync or stream)
  4. Output guardrails
  5. Langfuse observability

The `Intent` enum is retained as an observability field for backward compat
with the API response shape, but it is NO LONGER used for routing.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from src.core.config import get_settings
from src.services.llm import get_llm
from src.utils.text import strip_think as _strip_think

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Intent label (observability only — not used for routing)."""
    DIRECT_CHAT = "direct_chat"
    RAG_QUERY = "rag_query"
    AGENT_TASK = "agent_task"
    BLOCKED = "blocked"


@dataclass
class OrchestratorResult:
    """Unified response from the orchestrator."""
    answer: str
    intent: Intent
    model_used: str = ""
    sources: list[dict] = field(default_factory=list)
    tools_used: list[dict] = field(default_factory=list)  # now {name, args, output_preview}
    guardrails: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "intent": self.intent.value,
            "model_used": self.model_used,
            "sources": self.sources,
            "tools_used": self.tools_used,
            "guardrails": self.guardrails,
            "latency_ms": round(self.latency_ms, 1),
            "metadata": self.metadata,
        }


# =============================================================================
# FAST PATH DETECTION
# =============================================================================

# Heuristics: skip the agent for obvious simple cases (sapaan, dll)
# Agent still works for these — fast path is purely an optimization
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|halo|hai|selamat|good\s+(morning|afternoon|evening)|apa\s+kabar|thanks|thank\s+you|terima\s+kasih)\b",
    re.IGNORECASE,
)


def _is_trivial(message: str) -> bool:
    """Cheap heuristic: very short, no question word, no special chars."""
    msg = message.strip()
    if len(msg) > 80:
        return False
    if "?" in msg or "?" in msg:
        return False
    if any(c in msg for c in "+-*/=()[]"):
        return False
    return bool(_GREETING_RE.match(msg))


class Orchestrator:
    """Slim orchestrator — delegates to the agent for actual work."""

    def __init__(self):
        self._settings = get_settings()
        self._fast_llm = None

    def _get_fast_llm(self):
        if self._fast_llm is None:
            self._fast_llm = get_llm("classifier")
        return self._fast_llm

    # --------------------------------------------------------------- sync
    async def process(
        self,
        message: str,
        history: list[dict] | None = None,
        model: str | None = None,
        force_intent: str | None = None,
    ) -> OrchestratorResult:
        """Process a user message end-to-end.

        Args:
            message: User's message
            history: Conversation history
            model: Override model (default: settings.default_model)
            force_intent: DEPRECATED. Accepted for backward compat but ignored.
        """
        if force_intent:
            logger.warning(f"`force_intent={force_intent}` is deprecated in full-agentic mode, ignoring")

        start = time.perf_counter()
        guardrails_info: dict = {"input": {}, "output": {}}

        # 1. Input guardrails
        from src.services.guardrails import run_input_guardrails

        input_guard = run_input_guardrails(message)
        guardrails_info["input"] = input_guard.to_dict()
        safe_message = input_guard.redacted_text or message

        if input_guard.blocked:
            elapsed = (time.perf_counter() - start) * 1000
            return OrchestratorResult(
                answer="I can't help with that request. It was flagged by content safety filters.",
                intent=Intent.BLOCKED,
                guardrails=guardrails_info,
                latency_ms=elapsed,
            )

        # 2. LLM-based safety check
        is_safe, reason = await self._llm_safety_check(safe_message)
        if not is_safe:
            elapsed = (time.perf_counter() - start) * 1000
            guardrails_info["input"]["llm_safety"] = {"passed": False, "reason": reason}
            return OrchestratorResult(
                answer="I can't help with that request. Please ask something appropriate.",
                intent=Intent.BLOCKED,
                guardrails=guardrails_info,
                latency_ms=elapsed,
            )

        # 3. Optional fast path for trivial messages
        if _is_trivial(safe_message):
            answer = await self._fast_path_answer(safe_message, history, model)
            elapsed = (time.perf_counter() - start) * 1000
            return OrchestratorResult(
                answer=answer,
                intent=Intent.DIRECT_CHAT,
                model_used=model or self._settings.default_model,
                guardrails=guardrails_info,
                latency_ms=elapsed,
                metadata={"path": "fast"},
            )

        # 4. Delegate to the agent (single brain)
        from src.services.agents import run_agent

        agent_result = await run_agent(
            task=safe_message,
            model=model or self._settings.agent_model,
            history=history,
        )

        # 5. Map agent result → OrchestratorResult
        answer = agent_result.get("answer", "")
        tools = [
            {"name": t, "args": {}, "output_preview": ""}
            for t in agent_result.get("tools_used", [])
        ]
        intent = self._infer_intent(agent_result.get("tools_used", []))

        # 6. Output guardrails
        from src.services.guardrails import run_output_guardrails

        out_guard = run_output_guardrails(answer)
        guardrails_info["output"] = out_guard.to_dict()
        if out_guard.redacted_text:
            answer = out_guard.redacted_text

        elapsed = (time.perf_counter() - start) * 1000
        return OrchestratorResult(
            answer=answer,
            intent=intent,
            model_used=agent_result.get("model", self._settings.agent_model),
            tools_used=tools,
            guardrails=guardrails_info,
            latency_ms=elapsed,
            metadata={
                "path": "agent",
                "steps": agent_result.get("steps", 0),
                "reasoning_steps": agent_result.get("reasoning", []),
            },
        )

    # --------------------------------------------------------------- stream
    async def process_stream(
        self,
        message: str,
        history: list[dict] | None = None,
        model: str | None = None,
        force_intent: str | None = None,
    ) -> AsyncIterator[dict]:
        """Stream a response — yields typed events from the agent.

        Event shapes:
          {"event": "metadata", "session_id", "model", "tools", "intent"}
          {"event": "tool_start", "tool", "args"}
          {"event": "tool_end", "tool", "output_preview"}
          {"event": "content", "delta"}
          {"event": "done", "answer", "tools_used", "sources", "latency_ms", ...}
          {"event": "error", "detail"}
        """
        if force_intent:
            logger.warning(f"`force_intent={force_intent}` is deprecated in full-agentic mode, ignoring")

        start = time.perf_counter()
        guardrails_info: dict = {"input": {}, "output": {}}

        # 1. Input guardrails
        from src.services.guardrails import run_input_guardrails

        input_guard = run_input_guardrails(message)
        guardrails_info["input"] = input_guard.to_dict()
        safe_message = input_guard.redacted_text or message

        if input_guard.blocked:
            elapsed = (time.perf_counter() - start) * 1000
            yield {
                "event": "done",
                "answer": "I can't help with that request. It was flagged by content safety filters.",
                "intent": Intent.BLOCKED.value,
                "sources": [],
                "tools_used": [],
                "guardrails": guardrails_info,
                "latency_ms": elapsed,
            }
            return

        # 2. LLM-based safety check
        is_safe, reason = await self._llm_safety_check(safe_message)
        if not is_safe:
            elapsed = (time.perf_counter() - start) * 1000
            guardrails_info["input"]["llm_safety"] = {"passed": False, "reason": reason}
            yield {
                "event": "done",
                "answer": "I can't help with that request. Please ask something appropriate.",
                "intent": Intent.BLOCKED.value,
                "sources": [],
                "tools_used": [],
                "guardrails": guardrails_info,
                "latency_ms": elapsed,
            }
            return

        # 3. Optional fast path (no tool events)
        if _is_trivial(safe_message):
            yield {
                "event": "metadata",
                "intent": Intent.DIRECT_CHAT.value,
                "model_used": model or self._settings.default_model,
                "path": "fast",
            }
            answer = await self._fast_path_answer(safe_message, history, model)
            # Strip <think> blocks
            answer = _strip_think(answer)

            from src.services.guardrails import run_output_guardrails

            out_guard = run_output_guardrails(answer)
            guardrails_info["output"] = out_guard.to_dict()
            if out_guard.redacted_text:
                answer = out_guard.redacted_text

            elapsed = (time.perf_counter() - start) * 1000
            yield {
                "event": "done",
                "answer": answer,
                "intent": Intent.DIRECT_CHAT.value,
                "sources": [],
                "tools_used": [],
                "guardrails": guardrails_info,
                "latency_ms": elapsed,
                "model_used": model or self._settings.default_model,
            }
            return

        # 4. Full agent stream
        from src.services.agents import run_agent_stream

        full_answer = ""
        tools_used: list[str] = []
        reasoning_steps: list[dict] = []
        session_id = "stream-" + str(int(start))

        # Emit metadata first
        yield {
            "event": "metadata",
            "intent": Intent.AGENT_TASK.value,
            "model_used": model or self._settings.agent_model,
            "path": "agent",
            "session_id": session_id,
        }

        try:
            async for ev in run_agent_stream(
                task=safe_message,
                session_id=session_id,
                model=model,
                history=history,
            ):
                ev_type = ev.get("event")

                if ev_type == "content":
                    full_answer += ev.get("delta", "")
                    yield ev
                elif ev_type in ("tool_start", "tool_end"):
                    if ev_type == "tool_start" and ev.get("tool") not in tools_used:
                        tools_used.append(ev["tool"])
                    yield ev
                elif ev_type == "done":
                    # Apply output guardrails then emit final event
                    final_answer = ev.get("answer", full_answer)
                    from src.services.guardrails import run_output_guardrails

                    out_guard = run_output_guardrails(final_answer)
                    guardrails_info["output"] = out_guard.to_dict()
                    if out_guard.redacted_text:
                        final_answer = out_guard.redacted_text

                    elapsed = (time.perf_counter() - start) * 1000
                    yield {
                        "event": "done",
                        "answer": final_answer,
                        "intent": self._infer_intent(tools_used).value,
                        "sources": [],  # populated if RAG tool was used (future)
                        "tools_used": tools_used,
                        "reasoning_steps": reasoning_steps,
                        "guardrails": guardrails_info,
                        "latency_ms": round(elapsed, 1),
                        "model_used": ev.get("model", model or self._settings.agent_model),
                    }
                else:
                    # metadata / error / other — pass through
                    yield ev
        except Exception as e:
            logger.exception("Stream failed")
            yield {"event": "error", "detail": str(e)}

    # --------------------------------------------------------------- helpers
    def _infer_intent(self, tools_used: list[str]) -> Intent:
        """Observability-only: hint which path the agent took."""
        if "knowledge_base" in tools_used:
            return Intent.RAG_QUERY
        if tools_used:
            return Intent.AGENT_TASK
        return Intent.DIRECT_CHAT

    async def _llm_safety_check(self, message: str) -> tuple[bool, str | None]:
        """LLM-based content safety (preserved from original)."""
        from src.services.prompts import get_prompt

        prompt = get_prompt("safety-check")
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = self._get_fast_llm()
            response = llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=message),
            ])
            raw = response.content.strip().upper()
            if "UNSAFE" in raw:
                return False, "Content flagged as unsafe by AI safety classifier"
            return True, None
        except Exception as e:
            logger.warning(f"Safety check failed: {e}, allowing through")
            return True, None

    async def _fast_path_answer(
        self, message: str, history: list[dict] | None, model: str | None
    ) -> str:
        """Cheap direct answer for trivial messages (greetings, etc)."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from src.services.prompts import get_prompt

        llm = get_llm("chat", model=model or self._settings.default_model)
        system_prompt = get_prompt("fast-path-system")
        messages: list = [
            SystemMessage(content=system_prompt)
        ]
        if history:
            for msg in history[-6:]:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=message))
        response = llm.invoke(messages)
        return _strip_think(response.content)


# Singleton
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


__all__ = ["Orchestrator", "Intent", "OrchestratorResult", "get_orchestrator"]
