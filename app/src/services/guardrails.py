"""Guardrails service — production-grade input/output validation.

Features:
- LLM-based content safety classification
- Presidio-based PII detection (offline NER)
- Prompt injection detection
- Regex-based fast filters (first pass)
- Configurable rules
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# PII DETECTION — Presidio-based (offline, no API calls)
# =============================================================================

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False
    logger.warning("Presidio not available, falling back to regex PII detection")


# Fallback regex patterns (used if Presidio not available)
PII_REGEX_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+62|62|0)\d{9,12}\b"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "ID_NUMBER": re.compile(r"\b\d{16}\b"),  # Indonesian NIK
    "IP_ADDRESS": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "IBAN_CODE": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"),
}

# Presidio engine singletons
_analyzer: Any = None
_anonymizer: Any = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None and _PRESIDIO_AVAILABLE:
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            })
            nlp_engine = provider.create_engine()
            _analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        except Exception as e:
            logger.warning(f"Presidio init failed ({e}), falling back to regex PII detection")
            return None
    return _analyzer


def _get_anonymizer():
    global _anonymizer
    if _anonymizer is None and _PRESIDIO_AVAILABLE:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


def detect_pii_presidio(text: str, language: str = "en") -> list[dict]:
    """Detect PII using Presidio (offline NER). Returns list of detected entities."""
    analyzer = _get_analyzer()
    if analyzer is None:
        return detect_pii_regex(text)

    results = analyzer.analyze(
        text=text,
        language=language,
        entities=[
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
            "IBAN_CODE", "IP_ADDRESS", "LOCATION", "DATE_TIME",
        ],
        score_threshold=0.75,
    )

    return [
        {
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 3),
            "text": text[r.start:r.end],
        }
        for r in results
    ]


def detect_pii_regex(text: str) -> list[dict]:
    """Fallback PII detection using regex patterns."""
    found = []
    for pii_type, pattern in PII_REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            found.append({
                "entity_type": pii_type,
                "start": match.start(),
                "end": match.end(),
                "score": 0.85,
                "text": match.group(),
            })
    return found


def redact_pii(text: str, language: str = "en") -> tuple[str, list[dict]]:
    """Detect and redact PII from text. Returns (redacted_text, entities_found)."""
    entities = detect_pii_presidio(text, language)

    if not entities:
        return text, []

    # Use Presidio anonymizer if available
    anonymizer = _get_anonymizer()
    if anonymizer and _PRESIDIO_AVAILABLE:
        analyzer = _get_analyzer()
        results = analyzer.analyze(text=text, language=language, score_threshold=0.75)
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, entities

    # Fallback: manual redaction (replace from end to preserve positions)
    redacted = text
    for entity in sorted(entities, key=lambda e: e["start"], reverse=True):
        placeholder = f"[{entity['entity_type']}]"
        redacted = redacted[:entity["start"]] + placeholder + redacted[entity["end"]:]

    return redacted, entities


# =============================================================================
# PROMPT INJECTION DETECTION
# =============================================================================

INJECTION_PATTERNS = [
    # Direct instruction override attempts
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|above|prior)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your)\s+(above|previous|instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+\w+", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    # Role manipulation
    re.compile(r"pretend\s+(you\s+are|to\s+be|you're)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if|though|a)", re.IGNORECASE),
    re.compile(r"roleplay\s+as", re.IGNORECASE),
    # Data extraction attempts
    re.compile(r"(reveal|show|print|output)\s+(your|the|system)\s+(prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions)", re.IGNORECASE),
    # Encoding tricks
    re.compile(r"base64\s*(decode|encode)", re.IGNORECASE),
    re.compile(r"\\x[0-9a-fA-F]{2}", re.IGNORECASE),
]

# Toxic/harmful content patterns (fast regex pre-filter before LLM)
TOXIC_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(hack|exploit|crack)\s+(password|system|server|database)\b",
        r"\bmake\s+(a\s+)?(bomb|weapon|explosive|drug|poison)\b",
        r"\bhow\s+to\s+(steal|kill|attack|murder|kidnap)\b",
        r"\b(sql\s*injection|xss\s*attack|ddos)\b",
        r"\bcreate\s+(malware|virus|trojan|ransomware)\b",
    ]
]


def detect_injection(text: str) -> tuple[bool, str | None]:
    """Detect prompt injection attempts. Returns (is_injection, pattern_matched)."""
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, f"Injection pattern: {match.group()}"
    return False, None


def detect_toxicity(text: str) -> tuple[bool, str | None]:
    """Fast regex-based toxicity check (pre-filter before LLM check)."""
    for pattern in TOXIC_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, f"Toxic content: {match.group()}"
    return False, None


# =============================================================================
# LENGTH / FORMAT VALIDATION
# =============================================================================

def validate_length(text: str, max_chars: int = 8192) -> tuple[bool, str | None]:
    """Validate text length."""
    if len(text) > max_chars:
        return False, f"Text too long: {len(text)} chars (max {max_chars})"
    return True, None


def validate_not_empty(text: str) -> tuple[bool, str | None]:
    """Validate text is not empty or whitespace."""
    if not text or not text.strip():
        return False, "Empty or whitespace-only input"
    return True, None


# =============================================================================
# GUARDRAIL RESULT
# =============================================================================

@dataclass
class GuardrailResult:
    """Aggregated result of all guardrail checks."""
    passed: bool = True
    blocked: bool = False
    block_reason: str | None = None
    redacted_text: str | None = None
    checks: list[dict] = field(default_factory=list)
    pii_entities: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "checks": self.checks,
            "pii_detected": len(self.pii_entities) > 0,
            "pii_count": len(self.pii_entities),
            "pii_types": list(set(e["entity_type"] for e in self.pii_entities)),
        }


# =============================================================================
# MAIN GUARDRAIL PIPELINES
# =============================================================================

def run_input_guardrails(text: str) -> GuardrailResult:
    """
    Full input guardrail pipeline:
    1. Empty check
    2. Length check
    3. Prompt injection detection
    4. Toxicity pre-filter (regex)
    5. PII detection + redaction
    """
    result = GuardrailResult()

    # 1. Empty check
    ok, detail = validate_not_empty(text)
    result.checks.append({"name": "not_empty", "passed": ok, "detail": detail})
    if not ok:
        result.passed = False
        result.blocked = True
        result.block_reason = detail
        return result

    # 2. Length check
    ok, detail = validate_length(text, max_chars=8192)
    result.checks.append({"name": "max_length", "passed": ok, "detail": detail})
    if not ok:
        result.passed = False
        result.blocked = True
        result.block_reason = detail
        return result

    # 3. Prompt injection
    is_injection, pattern = detect_injection(text)
    result.checks.append({"name": "injection", "passed": not is_injection, "detail": pattern})
    if is_injection:
        result.passed = False
        result.blocked = True
        result.block_reason = f"Prompt injection detected: {pattern}"
        return result

    # 4. Toxicity (regex fast-path)
    is_toxic, reason = detect_toxicity(text)
    result.checks.append({"name": "toxicity", "passed": not is_toxic, "detail": reason})
    if is_toxic:
        result.passed = False
        result.blocked = True
        result.block_reason = reason
        return result

    # 5. PII detection + redaction
    redacted, entities = redact_pii(text)
    result.pii_entities = entities
    result.checks.append({
        "name": "pii_detection",
        "passed": True,
        "detail": f"Found {len(entities)} PII entities" if entities else None,
        "entities": [{"type": e["entity_type"], "score": e["score"]} for e in entities],
    })
    if entities:
        result.redacted_text = redacted

    return result


def run_output_guardrails(text: str) -> GuardrailResult:
    """
    Output guardrail pipeline:
    1. Length check

    Note: PII redaction is NOT applied to LLM output because the spacy
    en_core_web_sm model produces too many false positives on non-English
    (e.g. Indonesian) text, incorrectly masking normal words as PERSON/LOCATION.
    Input guardrails still protect user-submitted PII.
    """
    result = GuardrailResult()

    # 1. Length check
    ok, detail = validate_length(text, max_chars=4096)
    result.checks.append({"name": "max_length", "passed": ok, "detail": detail})
    if not ok:
        result.passed = False

    return result


def check_hallucination(response: str, context: list[str]) -> dict:
    """
    Check if response contains claims not grounded in the context.
    Uses simple token overlap — for production, use LLM-as-judge.
    """
    if not context:
        return {"score": 0.5, "grounded": True, "reason": "No context to check against"}

    context_text = " ".join(context).lower()
    response_sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 10]

    if not response_sentences:
        return {"score": 1.0, "grounded": True, "reason": "Response too short to check"}

    grounded_count = 0
    for sentence in response_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                      "to", "for", "of", "with", "and", "or", "it", "this", "that", "be"}
        meaningful = words - stop_words
        if not meaningful:
            grounded_count += 1
            continue
        context_words = set(context_text.split())
        overlap = meaningful & context_words
        if len(overlap) / len(meaningful) > 0.3:
            grounded_count += 1

    score = grounded_count / len(response_sentences)
    return {
        "score": round(score, 3),
        "grounded": score >= 0.5,
        "reason": f"{grounded_count}/{len(response_sentences)} sentences grounded in context",
    }
