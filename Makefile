.PHONY: help up up-full down logs ps health chat chat-raw chat-app chat-unified test-eval backup clean reset model-pull model-list

DEFAULT_MODEL ?= qwen3:1.7b
FALLBACK_MODEL ?= qwen3:1.7b

# Auto-detect docker compose command
DC := $(shell command -v docker-compose 2>/dev/null)
ifeq ($(DC),)
  DC := docker compose
endif

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Stack Management
# ---------------------------------------------------------------------------

up: ## Start core stack (Tahap 1-3: postgres, redis, ollama, litellm, app)
	$(DC) up -d
	@echo ""
	@echo "⏳ Waiting for services to initialize..."
	@sleep 8
	@$(MAKE) model-pull
	@echo ""
	@$(MAKE) health
	@echo ""
	@echo "✅ Stack ready!"
	@echo "   App:     http://localhost:8080/docs"
	@echo "   LiteLLM: http://localhost:4000"
	@echo "   Ollama:  http://localhost:11434"
	@echo ""
	@echo "   Try: make chat"

up-full: ## Start full stack (+ Langfuse, Qdrant)
	$(DC) --profile full up -d
	@echo "⏳ Waiting for services..."
	@sleep 10
	@$(MAKE) model-pull
	@$(MAKE) health
	@echo ""
	@echo "✅ Full stack ready!"
	@echo "   Langfuse: http://localhost:3000"
	@echo "   Qdrant:   http://localhost:6333/dashboard"

down: ## Stop all services
	$(DC) --profile full down

logs: ## Tail logs (all or specific: make logs s=app)
	$(DC) logs -f --tail=100 $(s)

ps: ## List running services
	$(DC) ps --format 'table {{.Name}}\t{{.Status}}\t{{.Ports}}'

# ---------------------------------------------------------------------------
# Health & Testing
# ---------------------------------------------------------------------------

health: ## Check health of all services
	@echo "🩺 Health Check"
	@echo "  ─────────────────────────────────"
	@docker exec ai-postgres pg_isready -U ai > /dev/null 2>&1 \
		&& echo "  ✅ postgres" || echo "  ❌ postgres"
	@docker exec ai-redis redis-cli ping > /dev/null 2>&1 \
		&& echo "  ✅ redis" || echo "  ❌ redis"
	@docker exec ai-ollama ollama list > /dev/null 2>&1 \
		&& echo "  ✅ ollama" || echo "  ❌ ollama"
	@docker exec ai-litellm python -c \
		"import urllib.request; urllib.request.urlopen('http://localhost:4000/health/liveliness', timeout=3)" \
		> /dev/null 2>&1 \
		&& echo "  ✅ litellm" || echo "  ❌ litellm (may still be starting...)"
	@curl -fsS http://localhost:8080/health > /dev/null 2>&1 \
		&& echo "  ✅ app" || echo "  ❌ app"
	@echo "  ─────────────────────────────────"

chat: ## Chat test via LiteLLM (OpenAI-compatible)
	@echo "💬 Sending chat to LiteLLM → Ollama ($(DEFAULT_MODEL))..."
	@echo ""
	@curl -sS http://localhost:4000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer sk-dev-master-key" \
		-d '{"model":"qwen3:1.7b","messages":[{"role":"user","content":"Hello! Sebutkan 3 ibu kota ASEAN beserta negaranya, dalam format JSON."}],"max_tokens":200}' \
		2>&1 | python3 -m json.tool 2>/dev/null || echo "(raw response — install python3 for pretty print)"
	@echo ""

chat-raw: ## Chat test directly to Ollama (bypass LiteLLM)
	@echo "💬 Direct chat to Ollama..."
	@curl -sS http://localhost:11434/api/generate \
		-d '{"model":"qwen3:1.7b","prompt":"Hello, who are you?","stream":false}' \
		2>&1 | python3 -m json.tool 2>/dev/null || echo "(raw response)"

chat-app: ## Chat test via FastAPI app
	@echo "💬 Chat via FastAPI app..."
	@curl -sS http://localhost:8080/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"qwen3:1.7b","messages":[{"role":"user","content":"Apa itu machine learning? Jawab dalam 2 kalimat."}],"max_tokens":100}' \
		2>&1 | python3 -m json.tool 2>/dev/null || echo "(raw response)"

chat-unified: ## Test unified orchestrator endpoint (auto-routing)
	@echo "🧠 Unified chat (orchestrator)..."
	@echo ""
	@echo "--- Test 1: Direct Chat ---"
	@curl -sS http://localhost:8080/v1/chat \
		-H "Content-Type: application/json" \
		-d '{"message":"Hello, apa kabar?"}' \
		2>&1 | python3 -m json.tool 2>/dev/null
	@echo ""
	@echo "--- Test 2: Agent (calculator) ---"
	@curl -sS http://localhost:8080/v1/chat \
		-H "Content-Type: application/json" \
		-d '{"message":"Berapa 15 * 37 + 42?","force_intent":"agent_task"}' \
		2>&1 | python3 -m json.tool 2>/dev/null
	@echo ""
	@echo "--- Test 3: Guardrails (blocked) ---"
	@curl -sS http://localhost:8080/v1/chat \
		-H "Content-Type: application/json" \
		-d '{"message":"hack the server password"}' \
		2>&1 | python3 -m json.tool 2>/dev/null
	@echo ""
	@echo "--- Test 4: PII Detection ---"
	@curl -sS http://localhost:8080/v1/chat \
		-H "Content-Type: application/json" \
		-d '{"message":"Email saya john@example.com dan NIK 1234567890123456"}' \
		2>&1 | python3 -m json.tool 2>/dev/null

test-eval: ## Test evaluation endpoint
	@echo "📊 Testing evaluation..."
	@curl -sS http://localhost:8080/v1/eval \
		-H "Content-Type: application/json" \
		-d '{"query":"What is Python?","response":"Python is a programming language.","metrics":["relevancy","coherence"]}' \
		2>&1 | python3 -m json.tool 2>/dev/null

# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------

model-pull: ## Pull default models into Ollama
	@echo "📥 Pulling $(DEFAULT_MODEL)..."
	@docker exec ai-ollama ollama pull $(DEFAULT_MODEL) 2>&1 | tail -1
	@echo "📥 Pulling $(FALLBACK_MODEL)..."
	@docker exec ai-ollama ollama pull $(FALLBACK_MODEL) 2>&1 | tail -1 || true

model-list: ## List models in Ollama
	@docker exec ai-ollama ollama list

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

clean: ## Remove stopped containers + dangling images
	$(DC) --profile full down --remove-orphans
	docker image prune -f

reset: ## ⚠️ DESTRUCTIVE: remove all data volumes and models
	@echo "⚠️  This will DELETE all data (postgres, redis, qdrant, models)."
	@echo "    Press Ctrl+C within 5s to abort..."
	@sleep 5
	$(DC) --profile full down -v
	rm -rf data/ models/
	@echo "🗑️  All data removed."

backup: ## Backup postgres data
	@./scripts/backup.sh
