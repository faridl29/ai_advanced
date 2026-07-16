"""Conversation memory — short-term context for the agent.

Backends:
- "memory" (default): in-process dict, good for single-instance dev
- "redis": distributed, for production / multi-replica

The agent uses this to retrieve prior turns and inject them into the
LangGraph state. Long-term vector memory is out of scope for v1.
"""
from __future__ import annotations

import logging
import re
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


# Singletons
_memory: ConversationMemory | None = None
_profile_memory: UserProfileMemory | None = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory


class UserProfileMemory:
    """Long-term profile memory to persist user preferences and facts."""

    def __init__(self, backend: Backend | None = None):
        s = get_settings()
        self._backend: Backend = backend or s.memory_backend
        self._store: dict[str, list[str]] = defaultdict(list)
        logger.info(f"UserProfileMemory initialized (backend={self._backend})")

    async def get_profile_context(self, session_id: str) -> str:
        """Get user profile context formatted for system prompt introduction."""
        facts = await self.get_facts(session_id)
        if not facts:
            return ""
        
        formatted = "\n".join(f"- {f}" for f in facts)
        return (
            "--- USER PROFILE & MEMORIES (Persisted) ---\n"
            "Here are verified facts and preferences about this user from previous conversations. "
            "Adopt these preferences naturally:\n"
            f"{formatted}\n"
            "------------------------------------------"
        )

    async def get_facts(self, session_id: str) -> list[str]:
        """Retrieve list of facts for a session."""
        if self._backend == "redis":
            try:
                from redis.asyncio import Redis
                from src.core.config import get_settings
                s = get_settings()
                client = Redis.from_url(s.redis_url, decode_responses=True)
                try:
                    raw = await client.lrange(f"profile:{session_id}", 0, -1)
                    return raw if raw else []
                finally:
                    await client.aclose()
            except Exception as e:
                logger.warning(f"Redis profile read failed ({e}), falling back to in-memory")
        
        return self._store.get(session_id, [])

    async def add_fact(self, session_id: str, fact: str) -> None:
        """Add a single fact if not already present."""
        facts = await self.get_facts(session_id)
        if fact in facts:
            return
            
        if self._backend == "redis":
            try:
                from redis.asyncio import Redis
                from src.core.config import get_settings
                s = get_settings()
                client = Redis.from_url(s.redis_url, decode_responses=True)
                try:
                    await client.rpush(f"profile:{session_id}", fact)
                    await client.expire(f"profile:{session_id}", 31536000)  # 1 year TTL
                    return
                finally:
                    await client.aclose()
            except Exception as e:
                logger.warning(f"Redis profile write failed ({e}), falling back to in-memory")
                
        if fact not in self._store[session_id]:
            self._store[session_id].append(fact)

    async def clear_profile(self, session_id: str) -> None:
        """Clear profile memory for a session."""
        if self._backend == "redis":
            try:
                from redis.asyncio import Redis
                from src.core.config import get_settings
                s = get_settings()
                client = Redis.from_url(s.redis_url, decode_responses=True)
                try:
                    await client.delete(f"profile:{session_id}")
                finally:
                    await client.aclose()
            except Exception as e:
                logger.warning(f"Redis profile clear failed ({e})")
        
        self._store.pop(session_id, None)

    async def extract_and_update_profile(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Use LLM judge to extract new facts or preferences from this turn and save them."""
        clean_user = user_msg.strip()
        if len(clean_user) < 15:
            return
            
        from src.services.llm import get_llm
        from src.services.prompts import get_prompt
        from langchain_core.messages import SystemMessage, HumanMessage
        import json
        
        prompt = get_prompt("profile-fact-extractor")
        
        try:
            # Set max_tokens to 512 to ensure the reasoning/think block is not truncated
            llm = get_llm("classifier", max_tokens=512)
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"User: {user_msg}\nAssistant: {assistant_msg}")
            ]
            response = await llm.ainvoke(messages)
            raw = response.content.strip()
            
            # Remove <think>...</think> blocks from Qwen/reasoning models first
            clean_raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            
            match = re.search(r"\[.*\]", clean_raw, re.DOTALL)
            if match:
                facts = json.loads(match.group(0))
                if isinstance(facts, list):
                    for fact in facts:
                        if isinstance(fact, str) and len(fact.strip()) > 3:
                            await self.add_fact(session_id, fact.strip())
                            logger.info(f"Extracted and saved long-term profile memory: '{fact.strip()}' for session {session_id}")
        except Exception as e:
            logger.warning(f"Profile extraction failed: {e}")


def get_profile_memory() -> UserProfileMemory:
    global _profile_memory
    if _profile_memory is None:
        _profile_memory = UserProfileMemory()
    return _profile_memory


__all__ = ["ConversationMemory", "get_memory", "UserProfileMemory", "get_profile_memory"]
