"""Conversation memory — short-term context for the agent.

Backends:
- "memory" (default): in-process dict, good for single-instance dev
- "redis": distributed, for production / multi-replica

The agent uses this to retrieve prior turns and inject them into the
LangGraph state. Long-term vector memory is out of scope for v1.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Literal

from src.core.config import get_settings

logger = logging.getLogger(__name__)

Backend = Literal["memory", "redis"]


class ConversationMemory:
    """Per-session message history with pluggable backend."""

    def __init__(self, backend: Backend | None = None, max_turns: int | None = None):
        s = get_settings()
        self._backend: Backend = backend or s.memory_backend
        self._max_turns = max_turns or s.memory_max_turns
        self._store: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._max_turns * 2))
        logger.info(f"ConversationMemory initialized (backend={self._backend}, max_turns={self._max_turns})")

    async def get_history(self, session_id: str) -> list[dict]:
        """Return the most recent turns for a session (oldest first)."""
        if self._backend == "redis":
            return await self._get_history_redis(session_id)
        return list(self._store.get(session_id, []))

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a turn to the session history."""
        if self._backend == "redis":
            await self._add_message_redis(session_id, role, content)
            return
        self._store[session_id].append({"role": role, "content": content})

    async def clear(self, session_id: str) -> None:
        """Clear all history for a session."""
        if self._backend == "redis":
            await self._clear_redis(session_id)
            return
        self._store.pop(session_id, None)

    # ------------------------------------------------------------------ Redis
    async def _get_history_redis(self, session_id: str) -> list[dict]:
        try:
            from redis.asyncio import Redis
            from src.core.config import get_settings
            s = get_settings()
            client = Redis.from_url(s.redis_url, decode_responses=True)
            try:
                raw = await client.lrange(f"memory:{session_id}", 0, -1)
                return [eval(x) if x.startswith("{") else {"role": "user", "content": x} for x in raw]  # noqa: S307
            finally:
                await client.aclose()
        except Exception as e:
            logger.warning(f"Redis read failed ({e}), falling back to in-memory")
            return list(self._store.get(session_id, []))

    async def _add_message_redis(self, session_id: str, role: str, content: str) -> None:
        try:
            from redis.asyncio import Redis
            from src.core.config import get_settings
            s = get_settings()
            client = Redis.from_url(s.redis_url, decode_responses=True)
            try:
                import json
                await client.lpush(f"memory:{session_id}", json.dumps({"role": role, "content": content}))
                await client.ltrim(f"memory:{session_id}", 0, self._max_turns * 2 - 1)
                await client.expire(f"memory:{session_id}", 86400)  # 24h TTL
            finally:
                await client.aclose()
        except Exception as e:
            logger.warning(f"Redis write failed ({e}), falling back to in-memory")
            self._store[session_id].append({"role": role, "content": content})

    async def _clear_redis(self, session_id: str) -> None:
        try:
            from redis.asyncio import Redis
            from src.core.config import get_settings
            s = get_settings()
            client = Redis.from_url(s.redis_url, decode_responses=True)
            try:
                await client.delete(f"memory:{session_id}")
            finally:
                await client.aclose()
        except Exception as e:
            logger.warning(f"Redis clear failed ({e})")
            self._store.pop(session_id, None)


# Singleton
_memory: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory


__all__ = ["ConversationMemory", "get_memory"]
