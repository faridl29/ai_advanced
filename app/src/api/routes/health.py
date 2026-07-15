"""Health check — liveness probe (always 200 if process alive)."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from src.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Liveness probe. Returns 200 if the process is alive."""
    s = get_settings()
    return {"status": "ok", "app": s.app_name, "version": s.app_version}


@router.get("/health/ready")
async def readiness(request: Request):
    """Readiness probe. Checks Redis + LiteLLM connectivity."""
    s = get_settings()
    checks = {}

    # Redis
    try:
        await request.app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    # LiteLLM
    try:
        r = await request.app.state.http.get("/health/liveliness", timeout=3.0)
        checks["litellm"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        checks["litellm"] = f"error: {e}"

    ok = all(v == "ok" for v in checks.values())
    return ORJSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not_ready", "checks": checks},
    )
