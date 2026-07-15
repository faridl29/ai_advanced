"""Agent routes — run LangGraph agents with tool calling."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.services.agents import run_agent

router = APIRouter()


class AgentRequest(BaseModel):
    task: str = Field(..., description="Task for the agent to complete")
    tools: list[str] | None = Field(None, description="Optional tool filter")
    max_steps: int = Field(10, ge=1, le=20, description="Max reasoning steps")
    model: str | None = None


@router.post("/agents/run")
async def agent_run(body: AgentRequest) -> ORJSONResponse:
    """Run a LangGraph agent with tool calling."""
    try:
        result = await run_agent(
            task=body.task,
            tools=body.tools,
            max_steps=body.max_steps,
        )
        return ORJSONResponse(content={"status": "ok", **result})
    except Exception as e:
        return ORJSONResponse(
            content={"error": "agent_failed", "detail": str(e)},
            status_code=500,
        )
