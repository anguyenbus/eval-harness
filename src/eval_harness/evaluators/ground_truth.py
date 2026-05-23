"""
Ground truth evaluators for synthetic spans.

This module provides evaluators that work with synthetic spans,
using ground truth data encoded in span metadata.
"""

from __future__ import annotations

import json
from typing import Any, Final

from beartype import beartype

# Constants
SYNTHETIC_MARKER: Final[str] = "eval_harness.synthetic"


@beartype
def is_synthetic_span(span: dict[str, Any]) -> bool:
    """
    Check if a span is synthetic.

    Args:
        span: Span dictionary with attributes.

    Returns:
        True if span has eval_harness.synthetic == "true", False otherwise.

    """
    return span.get("eval_harness.synthetic") == "true"


@beartype
def extract_ground_truth(span: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract ground truth from synthetic span metadata.

    Args:
        span: Span dictionary with metadata attribute.

    Returns:
        Dictionary with ground truth data or None if not synthetic.

    """
    if not is_synthetic_span(span):
        return None

    metadata_str = span.get("metadata")
    if not metadata_str:
        return None

    try:
        metadata = json.loads(metadata_str)
        return {
            "source_question_id": metadata.get("source_question_id"),
            "expected_passage_id": metadata.get("expected_passage_id"),
            "expected_answer": metadata.get("expected_answer"),
        }
    except (json.JSONDecodeError, TypeError):
        return None


@beartype
def precision_at_k(
    retrieved_passage_ids: list[str],
    expected_passage_id: str,
    k: int = 5,
) -> float:
    """
    Calculate precision@k for retrieval.

    Args:
        retrieved_passage_ids: List of retrieved passage IDs.
        expected_passage_id: Expected passage ID from ground truth.
        k: Number of top results to consider.

    Returns:
        Precision@k score (0.0 to 1.0).

    """
    if not retrieved_passage_ids or k <= 0:
        return 0.0

    top_k = retrieved_passage_ids[:k]
    return 1.0 if expected_passage_id in top_k else 0.0


@beartype
def answer_correctness(
    generated_answer: str,
    expected_answer: str,
) -> dict[str, Any]:
    """
    Calculate answer correctness using semantic similarity.

    This is a simplified version that uses token overlap.
    For production, use LLM-judge evaluation.

    Args:
        generated_answer: Generated answer text.
        expected_answer: Expected answer text from ground truth.

    Returns:
        Dictionary with correctness score and metadata.

    """
    # Normalize answers
    gen_normalized = generated_answer.lower().strip()
    exp_normalized = expected_answer.lower().strip()

    # Simple token-based similarity (F1-like)
    gen_tokens = set(gen_normalized.split())
    exp_tokens = set(exp_normalized.split())

    if not gen_tokens or not exp_tokens:
        return {"score": 0.0, "overlap": 0.0, "coverage": 0.0}

    overlap = gen_tokens & exp_tokens
    precision = len(overlap) / len(gen_tokens) if gen_tokens else 0.0
    recall = len(overlap) / len(exp_tokens) if exp_tokens else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "score": f1,
        "overlap": len(overlap),
        "coverage": recall,
        "precision": precision,
    }


@beartype
def evaluate_synthetic_span(
    span: dict[str, Any],
    retrieved_passage_ids: list[str],
    generated_answer: str,
) -> dict[str, Any]:
    """
    Evaluate a synthetic span using ground truth.

    Args:
        span: Root span dictionary with metadata.
        retrieved_passage_ids: List of retrieved passage IDs.
        generated_answer: Generated answer text.

    Returns:
        Dictionary with evaluation scores.

    """
    ground_truth = extract_ground_truth(span)
    if ground_truth is None:
        return {
            "error": "Not a synthetic span or missing metadata",
            "precision_at_k": None,
            "answer_correctness": None,
        }

    # Calculate precision@k
    p_at_5 = precision_at_k(
        retrieved_passage_ids,
        ground_truth["expected_passage_id"],
        k=5,
    )

    # Calculate answer correctness
    correctness = answer_correctness(
        generated_answer,
        ground_truth["expected_answer"],
    )

    return {
        "source_question_id": ground_truth["source_question_id"],
        "expected_passage_id": ground_truth["expected_passage_id"],
        "precision_at_k": p_at_5,
        "answer_correctness": correctness["score"],
        "answer_correctness_details": correctness,
    }
