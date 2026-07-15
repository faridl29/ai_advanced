#!/usr/bin/env bash
set -euo pipefail

# Backup script: Postgres + Qdrant + volumes
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Backing up to $BACKUP_DIR ..."

# Postgres
echo "  - Postgres ..."
docker exec ai-postgres pg_dump -U "${POSTGRES_USER:-ai}" "${POSTGRES_DB:-ai_platform}" \
  > "$BACKUP_DIR/postgres.sql" 2>/dev/null || echo "    (skipped: not running)"

# Qdrant snapshot
echo "  - Qdrant ..."
if docker ps --format '{{.Names}}' | grep -q "ai-qdrant"; then
  docker exec ai-qdrant curl -s -X POST "http://localhost:6333/snapshots" \
    -o "$BACKUP_DIR/qdrant_snapshot.json" || echo "    (qdrant snapshot API may differ in version)"
fi

# Models manifest
echo "  - Ollama model list ..."
docker exec ai-ollama ollama list > "$BACKUP_DIR/ollama_models.txt" 2>/dev/null || echo "    (skipped: ollama not running)"

# Compress
echo "🗜️  Compressing ..."
tar -czf "$BACKUP_DIR.tar.gz" -C "./backups" "$(basename "$BACKUP_DIR")"
rm -rf "$BACKUP_DIR"

echo "✅ Backup saved to $BACKUP_DIR.tar.gz"
ls -lh "$BACKUP_DIR.tar.gz"
