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
    default_model: str = "qwen3:4b"
    litellm_base_url: str = "http://litellm:4000"
    litellm_master_key: str = "sk-dev-master-key"
    ollama_base_url: str = "http://ollama:11434"

    # Infrastructure
    redis_url: str = "redis://redis:6379"
    postgres_url: str = "postgresql+asyncpg://ai:ai_secret@postgres:5432/ai_platform"
    qdrant_url: str = "http://qdrant:6333"
    searxng_url: str = "http://searxng:8080"

    # Langfuse (Tahap 4)
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Timeouts
    request_timeout: int = 300

    # Agent — full agentic ReAct executor
    # NOTE: must match a model defined in litellm/config.yaml AND pulled in ollama
    agent_model: str = "qwen3:4b"
    agent_max_steps: int = 6  # smaller model = fewer steps untuk avoid loop
    agent_temperature: float = 0.1
    agent_max_tokens: int = 2048
    agent_stream: bool = True
    agent_reflect: bool = True  # Reflection node (verify answer before emit)
    agent_think_tool: bool = False  # Disabled: think tool causes loops di small models

    # Memory backend
    memory_backend: Literal["memory", "redis"] = "memory"
    memory_max_turns: int = 10

    # Tool registry
    tools_enabled: list[str] = [
        "calculator", "current_datetime", "knowledge_base",
        "web_search", "python_executor", "think",
        "financial_analyzer", "generate_excel_report",
    ]

    # Reports
    reports_dir: str = "/app/reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
