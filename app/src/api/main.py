"""FastAPI app — main entrypoint."""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from src.core.config import get_settings
from src.core.lifespan import lifespan
from src.api.routes import agents, chat, guardrails, health, rag, reports, structured

settings = get_settings()

app = FastAPI(
    title="AI Platform API",
    version=settings.app_version,
    description="On-premise AI platform: chat, structured output, RAG, agents",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (open for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    return response


# Register routes
app.include_router(health.router)
app.include_router(chat.router, prefix="/v1", tags=["chat"])
app.include_router(structured.router, prefix="/v1", tags=["structured"])
app.include_router(rag.router, prefix="/v1", tags=["rag"])
app.include_router(agents.router, prefix="/v1", tags=["agents"])
app.include_router(reports.router, prefix="/v1", tags=["reports"])
app.include_router(guardrails.router, prefix="/v1", tags=["guardrails"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
