# AI Platform — On-Premise Local Stack

End-to-end AI platform yang berjalan on-premise di laptop 8–16 GB RAM, CPU-only.

**Stack**: FastAPI + LiteLLM + Ollama + Langfuse + Qdrant + LangGraph + PydanticAI + LlamaIndex + DeepEval + Guardrails AI + PostgreSQL + Redis

## Quickstart

```bash
# 1. Copy env
cp .env.example .env

# 2. Start Colima (jika belum)
colima start --cpu 4 --memory 8 --disk 30

# 3. Boot stack + auto-pull model
make up

# 4. Test chat
make chat
```

## Endpoints

| Service | URL | Description |
|---|---|---|
| FastAPI App | http://localhost:8080 | Main API + Swagger UI |
| LiteLLM | http://localhost:4000 | OpenAI-compatible router |
| Ollama | http://localhost:11434 | LLM inference |
| PostgreSQL | localhost:5432 | Metadata store |
| Redis | localhost:6379 | Cache |
| Langfuse | http://localhost:3000 | Observability (profile: full) |
| Qdrant | http://localhost:6333 | Vector DB (profile: full) |

## Commands

```bash
make help       # List all commands
make up         # Start core stack
make up-full    # Start full stack (+ Langfuse, Qdrant)
make down       # Stop everything
make health     # Check all services
make chat       # Chat via LiteLLM
make chat-app   # Chat via FastAPI
make chat-raw   # Chat directly to Ollama
make model-list # List Ollama models
make logs       # Tail logs
make logs s=app # Tail specific service
```

## Architecture

```
Browser → FastAPI :8080 → LiteLLM :4000 → Ollama :11434 (phi3/qwen2.5)
              │
              ├── Redis :6379 (cache)
              ├── PostgreSQL :5432 (metadata)
              ├── Qdrant :6333 (vectors, profile: full)
              └── Langfuse :3000 (traces, profile: full)
```
# ai_advanced
