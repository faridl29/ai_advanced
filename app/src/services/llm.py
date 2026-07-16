"""LLM client factory — single source of truth for ChatOpenAI construction.

DRY: replaces 3+ duplicated `ChatOpenAI(model=..., base_url=..., api_key=...)`
constructs scattered across orchestrator.py, rag.py, and evaluation.py.
"""
from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI

from src.core.config import get_settings

# Profile hint — picks sensible defaults for the task type.
LLMProfile = Literal["chat", "classifier", "rag"]


def _profile_defaults(profile: LLMProfile) -> dict:
    """Return (temperature, max_tokens, request_timeout) for a given profile."""
    if profile == "classifier":
        # Fast, deterministic, short output (e.g. intent labels).
        return {"temperature": 0.0, "max_tokens": 20, "request_timeout": 120}
    if profile == "rag":
        # Slightly cooler than chat, larger context window.
        return {"temperature": 0.3, "max_tokens": 1024, "request_timeout": 300}
    # chat (default)
    return {"temperature": 0.7, "max_tokens": 2048, "request_timeout": 300}


def get_llm(
    profile: LLMProfile = "chat",
    model: str | None = None,
    **overrides,
) -> ChatOpenAI:
    """Construct a ChatOpenAI client with profile-based defaults.

    Args:
        profile: One of 'chat', 'classifier', 'rag'. Determines temperature,
            max_tokens, and request_timeout defaults.
        model: Override model name. Defaults to settings.default_model.
        **overrides: Additional ChatOpenAI kwargs to override defaults.

    Returns:
        Configured ChatOpenAI client pointing at the configured LiteLLM gateway.
    """
    s = get_settings()
    params = {
        "model": model or s.default_model,
        "base_url": f"{s.litellm_base_url}/v1",
        "api_key": s.litellm_master_key,
        **_profile_defaults(profile),
        **overrides,
    }
    return ChatOpenAI(**params)
