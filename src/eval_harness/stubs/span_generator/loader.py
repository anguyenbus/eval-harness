"""
Question loader wrapper for synthetic span generation.

This module wraps the legal_rag_bench dataset loader and adds synthetic
fields (case_id, tenant_id_hashed) for replay evaluation.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from beartype import beartype
from beartype.typing import Optional

from eval_harness.datasets import load_legal_rag_bench

NUM_SYNTHETIC_TENANTS: Final[int] = 10


@beartype
@dataclass(frozen=True)
class GeneratorQuestion:
    """
    Question with synthetic metadata for span generation.

    Attributes:
        id: Original query ID from dataset.
        question: Question text.
        expected_answer: Reference answer text.
        relevant_passage_id: Passage ID containing the answer.
        case_id: Synthetic case ID (e.g., "synth-case-0001").
        tenant_id_hashed: Synthetic tenant ID (e.g., "synth-tenant-3").

    """

    id: str
    question: str
    expected_answer: str
    relevant_passage_id: str
    case_id: str
    tenant_id_hashed: str


@beartype
def _generate_case_id(index: int) -> str:
    """
    Generate synthetic case ID from question index.

    Args:
        index: Zero-based question index.

    Returns:
        Synthetic case ID in format "synth-case-NNNN".

    """
    return f"synth-case-{index:04d}"


@beartype
def _generate_tenant_id_hashed(index: int) -> str:
    """
    Generate synthetic tenant ID using modulo.

    Args:
        index: Zero-based question index.

    Returns:
        Synthetic tenant ID in format "synth-tenant-N" where N is 0-9.

    """
    tenant_num = index % NUM_SYNTHETIC_TENANTS
    return f"synth-tenant-{tenant_num}"


@beartype
def iter_questions(
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    cache_dir: Path = Path("data/rag/legal_rag_bench"),
) -> Iterator[GeneratorQuestion]:
    """
    Iterate over dataset questions with synthetic metadata.

    Wraps the legal_rag_bench dataset loader and adds synthetic
    case_id and tenant_id_hashed fields for replay evaluation.

    Args:
        limit: Maximum number of questions to yield. None for all.
        seed: Random seed for reproducibility (not used with deterministic loader).
        cache_dir: Path to dataset cache directory.

    Yields:
        GeneratorQuestion objects with synthetic metadata.

    Example:
        >>> for q in iter_questions(limit=5):
        ...     print(f"{q.case_id}: {q.question}")

    """
    # Set seed if provided (for future randomization features)
    if seed is not None:
        random.seed(seed)

    # Load dataset through existing loader (wrap, don't modify)
    dataset_iter = load_legal_rag_bench(cache_dir=cache_dir, slice="full")

    for index, (query_id, query_text, passage_id, answer) in enumerate(dataset_iter):
        # Apply limit
        if limit is not None and index >= limit:
            break

        yield GeneratorQuestion(
            id=str(query_id),
            question=query_text,
            expected_answer=answer,
            relevant_passage_id=passage_id,
            case_id=_generate_case_id(index),
            tenant_id_hashed=_generate_tenant_id_hashed(index),
        )
