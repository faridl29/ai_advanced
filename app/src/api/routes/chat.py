"""Unified chat route — single entry point that orchestrates all AI capabilities."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.core.config import get_settings

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class UnifiedChatRequest(BaseModel):
    """Unified chat request — the orchestrator decides how to handle it."""
    message: str = Field(..., description="User's message")
    history: list[ChatMessage] | None = Field(None, description="Conversation history")
    model: str | None = Field(None, description="Override model (default: settings)")
    force_intent: str | None = Field(None, description="Force routing: 'direct_chat', 'rag_query', 'agent_task'")


class LegacyChatRequest(BaseModel):
    """OpenAI-compatible chat completion request (legacy proxy mode)."""
    model: str | None = None
    messages: list[dict[str, Any]] = []
    max_tokens: int = 1024
    temperature: float = 0.7


# =============================================================================
# UNIFIED ENDPOINT (PRIMARY)
# =============================================================================

@router.post("/chat")
async def unified_chat(body: UnifiedChatRequest) -> ORJSONResponse:
    """
    Unified AI chat endpoint — the orchestrator handles everything:
    1. Input guardrails (safety + PII detection)
    2. Intent classification (auto-route to chat/RAG/agent)
    3. Execution via appropriate pipeline
    4. Output guardrails
    5. Response with full metadata

    Use `force_intent` to override auto-classification:
    - 'direct_chat': Simple LLM chat
    - 'rag_query': Search documents + generate answer
    - 'agent_task': Use tools (calculator, knowledge_base, etc.)
    """
    from src.services.orchestrator import get_orchestrator

    orchestrator = get_orchestrator()

    history = None
    if body.history:
        history = [{"role": m.role, "content": m.content} for m in body.history]

    result = await orchestrator.process(
        message=body.message,
        history=history,
        model=body.model,
        force_intent=body.force_intent,
    )

    return ORJSONResponse(content=result.to_dict())


@router.post("/chat/stream")
async def unified_chat_stream(body: UnifiedChatRequest):
    """Streaming chat — Server-Sent Events.

    Yields JSON-encoded events:
      - {event: "metadata", intent, model_used}
      - {event: "thinking", delta}    (qwen3 reasoning block)
      - {event: "content", delta}     (final answer chunks)
      - {event: "done", answer, sources, latency_ms, ...}
      - {event: "error", detail}
    """
    from src.services.orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    history = (
        [{"role": m.role, "content": m.content} for m in body.history]
        if body.history
        else None
    )

    async def _event_source():
        try:
            async for ev in orchestrator.process_stream(
                message=body.message,
                history=history,
                model=body.model,
                force_intent=body.force_intent,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        _event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# =============================================================================
# OPENAI-COMPATIBLE ENDPOINT (LEGACY / PROXY)
# =============================================================================

@router.post("/chat/completions")
async def chat_completions(request: Request, body: dict[str, Any]) -> ORJSONResponse:
    """OpenAI-compatible chat completion — proxies to LiteLLM."""
    settings = get_settings()
    body.setdefault("model", settings.default_model)

    try:
        r = await request.app.state.http.post("/v1/chat/completions", json=body)
        return ORJSONResponse(content=r.json(), status_code=r.status_code)
    except Exception as e:
        return ORJSONResponse(
            content={"error": "llm_unavailable", "detail": str(e)},
            status_code=502,
        )


@router.post("/completions")
async def completions(request: Request, body: dict[str, Any]) -> ORJSONResponse:
    """Text completion proxy."""
    settings = get_settings()
    body.setdefault("model", settings.default_model)

    try:
        r = await request.app.state.http.post("/v1/completions", json=body)
        return ORJSONResponse(content=r.json(), status_code=r.status_code)
    except Exception as e:
        return ORJSONResponse(
            content={"error": "llm_unavailable", "detail": str(e)},
            status_code=502,
        )


@router.get("/models")
async def list_models(request: Request) -> ORJSONResponse:
    """List available models from LiteLLM."""
    try:
        r = await request.app.state.http.get("/v1/models", timeout=5.0)
        return ORJSONResponse(content=r.json(), status_code=r.status_code)
    except Exception as e:
        return ORJSONResponse(
            content={"error": "upstream_unavailable", "detail": str(e)},
            status_code=503,
        )
