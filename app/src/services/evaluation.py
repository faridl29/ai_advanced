"""Evaluation service — LLM-as-judge metrics for response quality.

Features:
- Faithfulness: Is the response grounded in provided context?
- Answer Relevancy: Does the response actually answer the question?
- Contextual Precision: Are the retrieved chunks relevant?
- Coherence: Is the response well-structured?
- Overall scoring with pass/fail threshold
- Batch evaluation for dataset testing
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.core.config import get_settings
from src.services.prompts import get_prompt

logger = logging.getLogger(__name__)


# Prompts are managed in Langfuse. Below is the Evaluator class.


# =============================================================================
# EVALUATOR
# =============================================================================

class LLMEvaluator:
    """LLM-as-judge evaluator using local models via LiteLLM."""

    def __init__(self, model: str | None = None):
        s = get_settings()
        self._llm = ChatOpenAI(
            model=model or s.default_model,
            base_url=f"{s.litellm_base_url}/v1",
            api_key=s.litellm_master_key,
            temperature=0.0,
            max_tokens=150,
        )

    def _call_judge(self, prompt: str) -> dict:
        """Call LLM judge and parse JSON response."""
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                import json
                result = json.loads(json_match.group())
                return {
                    "score": float(result.get("score", 0.5)),
                    "reason": result.get("reason", "No explanation provided"),
                }

            # Fallback: try to find a number
            score_match = re.search(r'(\d+\.?\d*)', raw)
            if score_match:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 10.0  # Handle 0-10 scale
                return {"score": min(score, 1.0), "reason": raw[:100]}

            return {"score": 0.5, "reason": f"Could not parse response: {raw[:50]}"}
        except Exception as e:
            logger.warning(f"LLM judge call failed: {e}")
            return {"score": 0.5, "reason": f"Evaluation error: {e}"}

    def evaluate_faithfulness(self, response: str, context: list[str]) -> dict:
        """Evaluate if response is grounded in context."""
        if not context:
            return {"score": 0.5, "reason": "No context provided for faithfulness check"}

        prompt_tmpl = get_prompt("eval-faithfulness")
        prompt = prompt_tmpl.format(
            context="\n---\n".join(context),
            response=response,
        )
        return self._call_judge(prompt)

    def evaluate_relevancy(self, query: str, response: str) -> dict:
        """Evaluate if response is relevant to the question."""
        prompt_tmpl = get_prompt("eval-relevancy")
        prompt = prompt_tmpl.format(query=query, response=response)
        return self._call_judge(prompt)

    def evaluate_contextual_precision(self, query: str, context: list[str]) -> dict:
        """Evaluate if retrieved context is relevant to the question."""
        if not context:
            return {"score": 0.0, "reason": "No context chunks retrieved"}

        prompt_tmpl = get_prompt("eval-contextual-precision")
        prompt = prompt_tmpl.format(
            query=query,
            context="\n---\n".join(context),
        )
        return self._call_judge(prompt)

    def evaluate_coherence(self, response: str) -> dict:
        """Evaluate response coherence and structure."""
        prompt_tmpl = get_prompt("eval-coherence")
        prompt = prompt_tmpl.format(response=response)
        return self._call_judge(prompt)


# =============================================================================
# PUBLIC API
# =============================================================================

async def evaluate_response(
    query: str,
    response: str,
    context: list[str] | None = None,
    metrics: list[str] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate LLM response quality using LLM-as-judge.

    Args:
        query: Original user question
        response: LLM response to evaluate
        context: Retrieved context chunks (for faithfulness/precision)
        metrics: Which metrics to evaluate. Default: all applicable.
        model: Override model for evaluation judge

    Available metrics:
    - faithfulness: Is response grounded in context? (requires context)
    - relevancy: Does response answer the question?
    - contextual_precision: Are retrieved chunks relevant? (requires context)
    - coherence: Is response well-structured?
    """
    if metrics is None:
        metrics = ["relevancy", "coherence"]
        if context:
            metrics.extend(["faithfulness", "contextual_precision"])

    evaluator = LLMEvaluator(model)
    results = {}

    for metric in metrics:
        if metric == "faithfulness":
            results[metric] = evaluator.evaluate_faithfulness(response, context or [])
        elif metric == "relevancy":
            results[metric] = evaluator.evaluate_relevancy(query, response)
        elif metric == "contextual_precision":
            results[metric] = evaluator.evaluate_contextual_precision(query, context or [])
        elif metric == "coherence":
            results[metric] = evaluator.evaluate_coherence(response)
        else:
            results[metric] = {"score": 0.0, "reason": f"Unknown metric: {metric}"}

    # Overall score
    scores = [r["score"] for r in results.values()]
    overall = sum(scores) / len(scores) if scores else 0.0

    return {
        "overall_score": round(overall, 3),
        "metrics": results,
        "passed": overall >= 0.6,
        "threshold": 0.6,
        "query": query,
        "response_length": len(response),
        "model_judge": model or get_settings().default_model,
    }


async def evaluate_batch(
    test_cases: list[dict],
    model: str | None = None,
) -> dict:
    """
    Evaluate a batch of test cases.

    Each test case should have:
    - query: str
    - response: str
    - context: list[str] (optional)
    - expected: str (optional, for comparison)
    """
    results = []
    total_score = 0.0

    for i, case in enumerate(test_cases):
        result = await evaluate_response(
            query=case["query"],
            response=case["response"],
            context=case.get("context"),
            model=model,
        )
        result["test_case_index"] = i
        results.append(result)
        total_score += result["overall_score"]

    avg_score = total_score / len(results) if results else 0.0

    return {
        "total_cases": len(results),
        "average_score": round(avg_score, 3),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "results": results,
    }
