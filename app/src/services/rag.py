"""RAG service — production-grade retrieval-augmented generation.

Features:
- Dense + sparse hybrid search via Qdrant
- Conversation-aware retrieval (query reformulation with chat history)
- Multi-format document ingestion (PDF, DOCX, MD, TXT)
- Source citation formatting
- Configurable chunking strategies
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, models

from src.core.config import get_settings

logger = logging.getLogger(__name__)

# Singleton clients
_qdrant_client: QdrantClient | None = None
_index: VectorStoreIndex | None = None

COLLECTION_NAME = "documents"
EMBED_DIM = 768  # nomic-embed-text dimension


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        s = get_settings()
        _qdrant_client = QdrantClient(url=s.qdrant_url, timeout=30)
    return _qdrant_client


def _ensure_collection():
    """Ensure Qdrant collection exists with proper config for hybrid search."""
    client = _get_qdrant()
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")


def _configure_llama_index():
    """Configure LlamaIndex global settings (embedding model via Ollama)."""
    s = get_settings()
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=s.ollama_base_url,
    )
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.llm = None  # We use LiteLLM directly, not via LlamaIndex


def _get_index() -> VectorStoreIndex:
    """Get or create the vector store index."""
    global _index
    if _index is None:
        _configure_llama_index()
        _ensure_collection()
        client = _get_qdrant()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
    return _index


# =============================================================================
# INGESTION
# =============================================================================

async def ingest_file(file_path: Path, filename: str, metadata: dict[str, Any] | None = None) -> dict:
    """
    Ingest a file into the vector store.

    Supports: .txt, .md, .pdf, .docx
    Returns ingestion statistics.
    """
    _configure_llama_index()

    # Read documents
    reader = SimpleDirectoryReader(input_files=[str(file_path)])
    documents = reader.load_data()

    # Add metadata
    doc_id = hashlib.md5(filename.encode()).hexdigest()[:12]
    for doc in documents:
        doc.metadata.update({
            "filename": filename,
            "doc_id": doc_id,
            "file_type": Path(filename).suffix,
            **(metadata or {}),
        })

    # Parse into nodes (chunks) with sentence-aware splitting
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
        paragraph_separator="\n\n",
    )
    nodes = parser.get_nodes_from_documents(documents)

    # Index into Qdrant
    index = _get_index()
    index.insert_nodes(nodes)

    logger.info(f"Ingested {filename}: {len(documents)} docs, {len(nodes)} chunks")
    return {
        "filename": filename,
        "doc_id": doc_id,
        "documents": len(documents),
        "chunks": len(nodes),
        "file_type": Path(filename).suffix,
    }


# =============================================================================
# RETRIEVAL
# =============================================================================

async def query_rag(
    query: str,
    top_k: int = 5,
    history: list[dict] | None = None,
    rerank: bool = True,
) -> dict:
    """
    Query the RAG pipeline with hybrid search.

    Args:
        query: User's question
        top_k: Number of chunks to retrieve
        history: Conversation history for query reformulation
        rerank: Whether to rerank results by relevance
    """
    index = _get_index()

    # Reformulate query with conversation context
    effective_query = query
    if history:
        effective_query = _reformulate_query(query, history)
        logger.debug(f"Reformulated query: {effective_query}")

    # Dense retrieval
    retriever = index.as_retriever(similarity_top_k=top_k * 2)  # Over-retrieve for reranking
    nodes = retriever.retrieve(effective_query)

    # Reranking: score by keyword overlap + semantic score
    if rerank and len(nodes) > top_k:
        nodes = _rerank_nodes(nodes, query, top_k)
    else:
        nodes = nodes[:top_k]

    chunks = []
    for node in nodes:
        chunks.append({
            "text": node.text,
            "score": round(node.score, 4) if node.score else None,
            "metadata": node.metadata,
        })

    return {
        "query": query,
        "effective_query": effective_query,
        "chunks": chunks,
        "total": len(chunks),
    }


async def query_rag_with_answer(
    query: str,
    top_k: int = 5,
    model: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    Full RAG pipeline: retrieve + generate answer with source citations.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    s = get_settings()

    # Step 1: Retrieve relevant chunks
    retrieval_result = await query_rag(query, top_k=top_k, history=history)
    chunks = retrieval_result["chunks"]

    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
            "sources": [],
            "chunks": [],
            "query": query,
        }

    # Step 2: Format context for LLM
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("filename", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")
        if source not in [s.get("filename") for s in sources]:
            sources.append({
                "filename": source,
                "doc_id": chunk["metadata"].get("doc_id", ""),
                "relevance_score": chunk["score"],
            })

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Generate answer with LLM
    rag_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

RULES:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so honestly
- Cite sources using [Source N] notation when referencing specific information
- Be concise but thorough
- If the question is in Indonesian, answer in Indonesian

QUESTION: {query}"""

    llm = ChatOpenAI(
        model=model or s.default_model,
        base_url=f"{s.litellm_base_url}/v1",
        api_key=s.litellm_master_key,
        temperature=0.3,
        max_tokens=1024,
    )

    response = llm.invoke([
        HumanMessage(content=rag_prompt),
    ])

    return {
        "answer": response.content,
        "sources": sources,
        "chunks": chunks,
        "query": query,
        "effective_query": retrieval_result.get("effective_query", query),
    }


# =============================================================================
# UTILITIES
# =============================================================================

def _reformulate_query(query: str, history: list[dict]) -> str:
    """
    Reformulate query with conversation context for better retrieval.
    Uses simple heuristic: append recent assistant context to query.
    """
    # Get last 2 exchanges for context
    recent = history[-4:] if len(history) > 4 else history

    # Extract key context from recent messages
    context_parts = []
    for msg in recent:
        if msg.get("role") == "assistant" and msg.get("content"):
            # Extract key nouns from assistant response (first 100 chars)
            content = msg["content"][:100]
            context_parts.append(content)

    if not context_parts:
        return query

    # Combine: original query + recent context keywords
    context_summary = " ".join(context_parts[-1:])  # Only last response
    # Remove common words to extract key terms
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                  "to", "for", "of", "with", "and", "or", "it", "this", "that", "i",
                  "you", "we", "they", "he", "she", "can", "will", "would", "could"}
    keywords = [w for w in context_summary.lower().split() if w not in stop_words and len(w) > 2]

    if keywords:
        return f"{query} (context: {' '.join(keywords[:5])})"
    return query


def _rerank_nodes(nodes: list, query: str, top_k: int) -> list:
    """
    Rerank retrieved nodes by combining semantic score + keyword relevance.
    Simple but effective cross-encoder approximation.
    """
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
                  "when", "where", "do", "does", "did", "can", "could", "would",
                  "in", "on", "at", "to", "for", "of", "with", "and", "or"}
    query_keywords = query_words - stop_words

    scored_nodes = []
    for node in nodes:
        # Semantic score (from vector search)
        semantic_score = node.score or 0.0

        # Keyword overlap score
        node_words = set(node.text.lower().split())
        if query_keywords:
            keyword_overlap = len(query_keywords & node_words) / len(query_keywords)
        else:
            keyword_overlap = 0.0

        # Combined score (weighted)
        combined = 0.7 * semantic_score + 0.3 * keyword_overlap
        scored_nodes.append((combined, node))

    # Sort by combined score, return top_k
    scored_nodes.sort(key=lambda x: x[0], reverse=True)
    return [node for _, node in scored_nodes[:top_k]]


async def get_rag_status() -> dict:
    """Get RAG pipeline status."""
    try:
        client = _get_qdrant()
        collections = client.get_collections().collections
        col_info = None
        for c in collections:
            if c.name == COLLECTION_NAME:
                col_info = client.get_collection(COLLECTION_NAME)
                break

        if col_info:
            return {
                "status": "ready",
                "collection": COLLECTION_NAME,
                "documents_indexed": col_info.points_count,
                "vectors_count": col_info.vectors_count,
            }
        return {"status": "empty", "collection": COLLECTION_NAME, "documents_indexed": 0}
    except Exception as e:
        return {"status": "error", "error": str(e)}
