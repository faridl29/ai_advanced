"""Shared text utilities.

DRY: centralizes stripping of qwen3 <think>...</think> reasoning blocks
that were emitted in earlier code paths. Safe to call on any string.
"""
from __future__ import annotations

import re

# qwen3 emits <think>...</think> blocks before answering. We strip them
# post-hoc as a safety net. Use compile at import time — single source of truth.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove qwen3 <think>...</think> reasoning blocks from model output.

    Also handles an unclosed trailing <think> (split off everything from it).
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()
