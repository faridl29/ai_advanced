"""App lifespan: startup + shutdown hooks."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from redis.asyncio import Redis

from src.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared resources on startup, clean up on shutdown."""
    settings = get_settings()

    # Redis connection pool
    app.state.redis = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=10,
    )

    # Shared HTTP client for calling LiteLLM
    app.state.http = httpx.AsyncClient(
        base_url=settings.litellm_base_url,
        timeout=settings.request_timeout,
        headers={"Authorization": f"Bearer {settings.litellm_master_key}"},
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    )

    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.redis.aclose()
