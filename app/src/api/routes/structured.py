"""Structured output — JSON constrained generation via LiteLLM."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.core.config import get_settings

router = APIRouter()


class StructuredRequest(BaseModel):
    """Request for structured JSON generation."""
    prompt: str = Field(..., description="User prompt")
    schema_def: dict[str, Any] = Field(
        ..., alias="schema", description="JSON schema for desired output"
    )
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 512

    model_config = {"populate_by_name": True}


@router.post("/structured")
async def structured_output(request: Request, body: StructuredRequest) -> ORJSONResponse:
    """Generate JSON matching a schema. Uses response_format constraint."""
    settings = get_settings()
    model = body.model or settings.default_model

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise JSON generator. Output ONLY valid JSON matching the schema. No markdown, no commentary.",
            },
            {"role": "user", "content": body.prompt},
        ],
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        r = await request.app.state.http.post("/v1/chat/completions", json=payload)
        if r.status_code != 200:
            return ORJSONResponse(content=r.json(), status_code=r.status_code)

        content = r.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return ORJSONResponse(
                content={"error": "model_returned_invalid_json", "raw": content},
                status_code=422,
            )
        return ORJSONResponse(content={"data": parsed, "model": model})

    except Exception as e:
        return ORJSONResponse(
            content={"error": "llm_unavailable", "detail": str(e)},
            status_code=502,
        )
