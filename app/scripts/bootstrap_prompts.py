#!/usr/bin/env python3
"""Bootstrap Langfuse prompts — idempotent.

Run from the project root:
    docker exec ai-app python /app/scripts/bootstrap_prompts.py

This script creates 5 prompts in Langfuse that the orchestrator uses:
  - intent-classifier
  - safety-check
  - direct-chat-system
  - rag-answer-system
  - query-reformulation

Safe to re-run: existing prompts are updated, not duplicated.
"""
import sys
from pathlib import Path

# Allow running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings
from src.services.orchestrator import (
    INTENT_SYSTEM_PROMPT,
    GUARDRAIL_SYSTEM_PROMPT,
)

try:
    from langfuse import Langfuse
except ImportError:
    print("ERROR: langfuse not installed. Run: pip install 'langfuse>=2.50'")
    sys.exit(1)


PROMPTS = {
    "intent-classifier": INTENT_SYSTEM_PROMPT,
    "safety-check": GUARDRAIL_SYSTEM_PROMPT,
    "direct-chat-system": (
        "You are a helpful AI assistant. Answer questions clearly and concisely. "
        "If you don't know the answer, say so. "
        "Never include role labels like 'User:', 'Assistant:', 'A:', or 'System:' in your response. "
        "Just reply naturally."
    ),
    "rag-answer-system": (
        "You are a helpful AI assistant that answers questions based on provided context. "
        "Use the context below to answer the user's question. "
        "If the context doesn't contain relevant information, say so. "
        "Always cite your sources by referencing [Source N]. "
        "Answer in the same language as the user's question.\n\n"
        "--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
    ),
    "query-reformulation": (
        "Given the conversation history and the latest question, "
        "rewrite the question to be a standalone search query. "
        "Output ONLY the reformulated query, nothing else."
    ),
}


def main() -> int:
    s = get_settings()
    if not s.langfuse_public_key or not s.langfuse_secret_key:
        print("ERROR: Langfuse credentials not set in environment")
        return 1

    print(f"Connecting to Langfuse at {s.langfuse_host}...")
    client = Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )

    # Sanity check skipped — langfuse v4.x has strict Projects validation
    # that fails on auto-seeded projects. create_prompt will surface
    # real auth/network errors via its own 401/403 responses.
    print("✓ Skipping auth_check (use create_prompt to surface real errors)")

    # Push each prompt (create or update)
    success = 0
    for name, content in PROMPTS.items():
        try:
            # Try create first; if it exists, update.
            try:
                client.create_prompt(
                    name=name,
                    prompt=content,
                    labels=["production"],
                )
                print(f"  ✓ Created: {name}")
            except Exception as e:
                err = str(e).lower()
                if "already exists" in err or "409" in err or "unique" in err:
                    # Update existing prompt (creates a new version)
                    client.update_prompt(
                        name=name,
                        prompt=content,
                        labels=["production"],
                    )
                    print(f"  ↻ Updated: {name}")
                else:
                    raise
            success += 1
        except Exception as e:
            print(f"  ✗ Failed: {name} — {e}")

    client.flush()
    print(f"\nDone: {success}/{len(PROMPTS)} prompts synced")
    print(f"View in UI: {s.langfuse_host}/prompts")
    return 0 if success == len(PROMPTS) else 1


if __name__ == "__main__":
    sys.exit(main())
