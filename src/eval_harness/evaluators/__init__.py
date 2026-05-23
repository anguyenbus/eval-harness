"""
Evaluators for RAG and parsing output.

This package provides evaluators for computing metrics on RAG and parsing
output, including ground truth evaluators for synthetic spans.
"""

from eval_harness.evaluators.ground_truth import (
    answer_correctness,
    evaluate_synthetic_span,
    extract_ground_truth,
    is_synthetic_span,
    precision_at_k,
)

__all__ = [
    "is_synthetic_span",
    "extract_ground_truth",
    "precision_at_k",
    "answer_correctness",
    "evaluate_synthetic_span",
]
