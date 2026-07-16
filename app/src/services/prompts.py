"""Langfuse prompt management — single source of truth for all LLM prompts.

Prompts are managed in the Langfuse UI (http://localhost:3000) and fetched
at runtime. If Langfuse is unreachable, we fall back to local hardcoded
prompts so the app keeps working in degraded mode.

Benefits:
- Edit prompts in the Langfuse UI without redeploying
- Version history, A/B testing, environment labels (dev/staging/prod)
- Per-prompt cost & latency tracking in Langfuse traces
- One place to find every prompt in the system

Prompt names (must match the Langfuse UI):
- agent-system         (full-agentic ReAct system prompt)
- safety-check         (LLM-based content safety)
- think-tool-description
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from langfuse import Langfuse
from langfuse.model import TextPromptClient

from src.core.config import get_settings

logger = logging.getLogger(__name__)

# In-process cache: {name: (text, fetched_at)}
_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 30  # refresh window — small enough to feel "live", large enough to avoid hammering

_langfuse: Optional[Langfuse] = None


def _get_client() -> Langfuse | None:
    """Lazy-init the Langfuse client. Returns None if creds are missing."""
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    s = get_settings()
    if not s.langfuse_public_key or not s.langfuse_secret_key:
        logger.warning("Langfuse credentials not set — prompts will use fallbacks")
        return None
    _langfuse = Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )
    return _langfuse


def get_prompt(
    name: str,
    fallback: str,
    *,
    label: str | None = None,
    version: int | None = None,
) -> str:
    """Fetch a prompt from Langfuse with cache + fallback.

    Args:
        name: Prompt name in Langfuse (e.g. "intent-classifier").
        fallback: Local fallback used if Langfuse is down or prompt missing.
        label: Optional label filter (e.g. "production", "staging").
        version: Optional explicit version. If None, latest.

    Returns:
        Prompt text. From Langfuse if available, otherwise fallback.
    """
    cache_key = f"{name}:{label or ''}:{version or ''}"
    cached = _CACHE.get(cache_key)
    if cached and (time.time() - cached[1]) < _CACHE_TTL_SECONDS:
        return cached[0]

    client = _get_client()
    if client is not None:
        try:
            kwargs: dict = {"name": name}
            if label is not None:
                kwargs["label"] = label
            if version is not None:
                kwargs["version"] = version
            prompt: TextPromptClient = client.get_prompt(**kwargs)
            text = prompt.prompt
            _CACHE[cache_key] = (text, time.time())
            logger.info(f"Loaded prompt '{name}' from Langfuse (label={label}, version={version})")
            return text
        except Exception as e:
            logger.warning(f"Failed to load prompt '{name}' from Langfuse: {e}. Using fallback.")

    return fallback


def invalidate_cache() -> None:
    """Clear the in-process prompt cache. Useful for testing or forced refresh."""
    _CACHE.clear()
    logger.info("Langfuse prompt cache cleared")
