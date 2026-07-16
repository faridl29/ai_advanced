"""Agent routes — run the full-agentic ReAct executor with optional streaming."""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


class AgentRequest(BaseModel):
    task: str = Field(..., description="Task for the agent to complete")
    tools: list[str] | None = Field(
        None,
        description="Optional whitelist of tool names (default: settings.tools_enabled)",
    )
    max_steps: int = Field(10, ge=1, le=20, description="Max reasoning steps")
    model: str | None = None
    history: list[dict] | None = None
    session_id: str | None = None


@router.post("/agents/run")
async def agent_run(body: AgentRequest) -> ORJSONResponse:
    """Run a full-agentic ReAct agent and return the final answer (single-shot)."""
    from src.services.agents import run_agent

    try:
        result = await run_agent(
            task=body.task,
            model=body.model,
            max_steps=body.max_steps,
            tools=body.tools,
            history=body.history,
        )
        return ORJSONResponse(content={"status": "ok", **result})
    except Exception as e:
        return ORJSONResponse(
            content={"error": "agent_failed", "detail": str(e)},
            status_code=500,
        )


@router.post("/agents/run/stream")
async def agent_run_stream(body: AgentRequest):
    """Run the agent with full streaming observability.

    Yields SSE events:
      - {event: "metadata", session_id, model, tools}
      - {event: "tool_start", tool, args}
      - {event: "tool_end", tool, output_preview}
      - {event: "content", delta}
      - {event: "done", answer, tools_used, reasoning, latency_ms}
      - {event: "error", detail}
    """
    from src.services.agents import run_agent_stream

    session_id = body.session_id or f"agent-{id(body)}"

    async def _event_source():
        try:
            async for ev in run_agent_stream(
                task=body.task,
                session_id=session_id,
                model=body.model,
                history=body.history,
                tools=body.tools,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        _event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agents/tools")
async def list_tools() -> ORJSONResponse:
    """List all available tools (for UI tool picker / admin)."""
    from src.services.tools import get_tool_descriptions
    return ORJSONResponse(content={"tools": get_tool_descriptions()})
