"""Knowledge tools — RAG-backed document retrieval.

`knowledge_base` is the single tool the agent uses to query the document store.
It wraps `query_rag()` from src.services.rag (Qdrant + LlamaIndex).
"""
from __future__ import annotations

import asyncio
import concurrent.futures

from langchain_core.tools import tool


@tool
def knowledge_base(query: str) -> str:
    """Search the internal knowledge base / document store for information.
    Use when the user asks about documents, files, company info, or specific knowledge
    that might be in uploaded documents.

    Returns the top-3 most relevant text passages with source attribution.
    If no relevant documents are found, returns a clear 'not found' message
    so the agent can fall back to its own knowledge.
    """
    from src.services.rag import query_rag

    try:
        # Run async function in sync context (LangChain tool is sync)
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, query_rag(query, top_k=3)).result()
        except RuntimeError:
            result = asyncio.run(query_rag(query, top_k=3))

        if not result.get("chunks"):
            return "No relevant documents found in the knowledge base."

        texts = []
        for i, c in enumerate(result["chunks"], 1):
            src = c["metadata"].get("filename", "unknown")
            score = c.get("score", 0)
            texts.append(
                f"[Source {i}: {src} (score: {score:.2f})]\n{c['text'][:400]}"
            )
        return "\n\n---\n\n".join(texts)
    except Exception as e:
        return f"Error searching knowledge base: {e}"


__all__ = ["knowledge_base"]
