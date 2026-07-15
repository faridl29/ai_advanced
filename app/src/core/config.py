"""Core configuration via environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All settings loaded from env vars. Defaults work for local Docker dev."""

    # App
    app_name: str = "ai-platform"
    app_version: str = "0.1.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    environment: Literal["dev", "staging", "prod"] = "dev"

    # LLM
    default_model: str = "qwen3:1.7b"
    litellm_base_url: str = "http://litellm:4000"
    litellm_master_key: str = "sk-dev-master-key"
    ollama_base_url: str = "http://ollama:11434"

    # Infrastructure
    redis_url: str = "redis://redis:6379"
    postgres_url: str = "postgresql+asyncpg://ai:ai_secret@postgres:5432/ai_platform"
    qdrant_url: str = "http://qdrant:6333"

    # Langfuse (Tahap 4)
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Timeouts
    request_timeout: int = 120

    # Agent
    agent_model: str = "qwen3:1.7b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
