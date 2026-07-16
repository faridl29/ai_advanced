"""Web tools — web search with SearXNG and Wikipedia fallback.

Features:
- Live search via SearXNG (if deployed and settings.searxng_url is reachable)
- Graceful fallback to Wikipedia API (Indonesian and English) with User-Agent
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import logging

from langchain_core.tools import tool
from src.core.config import get_settings

logger = logging.getLogger(__name__)


def _search_wikipedia(query: str, lang: str = "id") -> str | None:
    """Helper to query Wikipedia API."""
    try:
        query_encoded = urllib.parse.quote(query)
        url = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={query_encoded}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AI Platform Agent; mailto:dev@example.com)"})
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return None
            
            formatted_results = []
            for i, r in enumerate(search_results[:3], 1):
                title = r.get("title", "No Title")
                snippet = r.get("snippet", "")
                # Clean up HTML tags (like <span class="searchmatch">)
                clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                wiki_url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                formatted_results.append(f"[Wikipedia {lang.upper()} Result {i}: {title} ({wiki_url})]\n{clean_snippet}")
            return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        logger.warning(f"Wikipedia search ({lang}) failed: {e}")
        return None


@tool
def web_search(query: str) -> str:
    """Search the web for current information.
    Use when asked about recent events, current news, or information that
    is likely not in the local knowledge base and may have changed since
    the model's training cutoff.
    """
    settings = get_settings()
    
    # 1. Try SearXNG if url is set
    if settings.searxng_url:
        try:
            query_encoded = urllib.parse.quote(query)
            url = f"{settings.searxng_url}/search?q={query_encoded}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode("utf-8"))
                results = data.get("results", [])
                if results:
                    formatted_results = []
                    for i, r in enumerate(results[:3], 1):
                        title = r.get("title", "No Title")
                        content = r.get("content", "No Content")
                        url_link = r.get("url", "")
                        formatted_results.append(f"[Web Result {i}: {title} ({url_link})]\n{content}")
                    return "\n\n---\n\n".join(formatted_results)
        except Exception as e:
            logger.info(f"SearXNG connection failed, falling back to Wikipedia: {e}")

    # 2. Fallback to Wikipedia API
    # Try ID first, then EN
    id_results = _search_wikipedia(query, "id")
    if id_results:
        return id_results
        
    en_results = _search_wikipedia(query, "en")
    if en_results:
        return en_results

    return "No relevant search results found on the web."


__all__ = ["web_search"]
