"""Guardrails + Evaluation API routes."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

router = APIRouter()


# =============================================================================
# GUARDRAILS
# =============================================================================

class GuardrailRequest(BaseModel):
    text: str = Field(..., description="Text to check")
    direction: str = Field("input", description="'input' or 'output'")
    check_injection: bool = Field(True, description="Check for prompt injection")


class HallucinationCheckRequest(BaseModel):
    response: str = Field(..., description="LLM response to check")
    context: list[str] = Field(..., description="Source context chunks")


@router.post("/guardrails/check")
async def guardrails_check(body: GuardrailRequest) -> ORJSONResponse:
    """Run guardrails on text — PII detection, toxicity, injection, length."""
    from src.services.guardrails import run_input_guardrails, run_output_guardrails

    if body.direction == "output":
        result = run_output_guardrails(body.text)
    else:
        result = run_input_guardrails(body.text)
    return ORJSONResponse(content=result.to_dict())


@router.post("/guardrails/pii/detect")
async def detect_pii_entities(body: GuardrailRequest) -> ORJSONResponse:
    """Detect PII entities in text (using Presidio or regex fallback)."""
    from src.services.guardrails import detect_pii_presidio

    entities = detect_pii_presidio(body.text)
    return ORJSONResponse(content={
        "text_length": len(body.text),
        "entities_found": len(entities),
        "entities": entities,
    })


@router.post("/guardrails/hallucination")
async def check_hallucination(body: HallucinationCheckRequest) -> ORJSONResponse:
    """Check if a response is grounded in the provided context."""
    from src.services.guardrails import check_hallucination

    result = check_hallucination(body.response, body.context)
    return ORJSONResponse(content=result)


# =============================================================================
# EVALUATION
# =============================================================================

class EvalRequest(BaseModel):
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="LLM response to evaluate")
    context: list[str] | None = Field(None, description="Retrieved context (for faithfulness)")
    metrics: list[str] | None = Field(None, description="Metrics: faithfulness, relevancy, coherence, contextual_precision")
    model: str | None = Field(None, description="Model to use for LLM-as-judge")


class BatchEvalRequest(BaseModel):
    dataset: list[EvalRequest] = Field(..., description="List of evaluation cases")


@router.post("/eval")
async def evaluate_response(body: EvalRequest) -> ORJSONResponse:
    """Evaluate a single LLM response using LLM-as-judge metrics."""
    from src.services.evaluation import evaluate_response as eval_fn

    result = await eval_fn(
        query=body.query,
        response=body.response,
        context=body.context,
        metrics=body.metrics,
        model=body.model,
    )
    return ORJSONResponse(content=result)


@router.post("/eval/batch")
async def batch_evaluate(body: BatchEvalRequest) -> ORJSONResponse:
    """Batch evaluate multiple responses. Returns aggregate scores."""
    from src.services.evaluation import evaluate_response as eval_fn

    results = []
    for item in body.dataset:
        result = await eval_fn(
            query=item.query,
            response=item.response,
            context=item.context,
            metrics=item.metrics,
            model=item.model,
        )
        results.append(result)

    # Aggregate scores
    all_scores = [r["overall_score"] for r in results if "overall_score" in r]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    pass_rate = sum(1 for r in results if r.get("passed")) / len(results) if results else 0.0

    return ORJSONResponse(content={
        "total": len(results),
        "average_score": round(avg_score, 3),
        "pass_rate": round(pass_rate, 3),
        "results": results,
    })
