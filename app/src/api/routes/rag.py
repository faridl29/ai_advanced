"""RAG routes — document ingestion and retrieval."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import ORJSONResponse

from src.core.config import get_settings
from src.services.rag import ingest_file, query_rag

router = APIRouter()


@router.post("/rag/ingest")
async def rag_ingest(
    request: Request,
    file: UploadFile = File(...),
) -> ORJSONResponse:
    """Upload and ingest a document into the vector store."""
    # Save uploaded file to temp
    suffix = Path(file.filename or "doc.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = await ingest_file(tmp_path, file.filename or "unknown", metadata={
            "size_bytes": len(content),
            "content_type": file.content_type,
        })
        return ORJSONResponse(content={"status": "ok", **result})
    except Exception as e:
        return ORJSONResponse(
            content={"error": "ingestion_failed", "detail": str(e)},
            status_code=500,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/rag/query")
async def rag_query(request: Request, body: dict) -> ORJSONResponse:
    """Query documents with RAG. Returns context chunks + LLM answer."""
    query = body.get("query", "")
    top_k = body.get("top_k", 5)
    model = body.get("model") or get_settings().default_model

    if not query:
        return ORJSONResponse(
            content={"error": "query is required"}, status_code=400
        )

    try:
        # Retrieve context
        retrieval = await query_rag(query, top_k=top_k)
        chunks = retrieval["chunks"]

        if not chunks:
            return ORJSONResponse(content={
                "answer": "No relevant documents found. Please ingest documents first.",
                "sources": [],
                "query": query,
            })

        # Build context for LLM
        context = "\n\n---\n\n".join(
            f"[Source: {c['metadata'].get('filename', 'unknown')}]\n{c['text']}"
            for c in chunks
        )

        # Call LLM via LiteLLM with context
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question based ONLY on the provided context. "
                        "If the context doesn't contain enough information, say so. "
                        "Cite sources using [Source: filename] format."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            "max_tokens": 512,
            "temperature": 0.3,
        }

        r = await request.app.state.http.post("/v1/chat/completions", json=payload)
        llm_response = r.json()
        answer = llm_response["choices"][0]["message"]["content"]

        sources = list({c["metadata"].get("filename", "unknown") for c in chunks})

        return ORJSONResponse(content={
            "answer": answer,
            "sources": sources,
            "query": query,
            "chunks_used": len(chunks),
        })

    except Exception as e:
        return ORJSONResponse(
            content={"error": "rag_query_failed", "detail": str(e)},
            status_code=500,
        )


@router.get("/rag/status")
async def rag_status() -> ORJSONResponse:
    """Check RAG pipeline status (Qdrant connectivity)."""
    try:
        from src.services.rag import _get_qdrant, COLLECTION_NAME
        client = _get_qdrant()
        collections = [c.name for c in client.get_collections().collections]
        doc_count = 0
        if COLLECTION_NAME in collections:
            info = client.get_collection(COLLECTION_NAME)
            doc_count = info.points_count or 0
        return ORJSONResponse(content={
            "status": "ok",
            "collections": collections,
            "documents_indexed": doc_count,
        })
    except Exception as e:
        return ORJSONResponse(
            content={"status": "error", "detail": str(e)},
            status_code=503,
        )
