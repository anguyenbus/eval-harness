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

from beartype import beartype
from beartype.typing import Final, Optional

from eval_harness.datasets import load_legal_rag_bench

NUM_SYNTHETIC_TENANTS: Final[int] = 10

# Demo showcase default constants
DEFAULT_DEMO_QUESTIONS: Final[int] = 50
DEFAULT_DEMO_SEED: Final[int] = 42


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


@beartype
def sample_questions(
    limit: int = DEFAULT_DEMO_QUESTIONS,
    seed: int = DEFAULT_DEMO_SEED,
    cache_dir: Path = Path("data/rag/legal_rag_bench"),
) -> list[GeneratorQuestion]:
    """
    Sample questions deterministically from legal-rag-bench dataset.

    This function provides reproducible sampling by setting a random seed
    before collecting questions. The same seed and limit will always produce
    the same list of questions.

    Args:
        limit: Maximum number of questions to sample. Default: 50.
        seed: Random seed for reproducibility. Default: 42.
        cache_dir: Path to dataset cache directory.

    Returns:
        List of GeneratorQuestion objects with synthetic metadata.

    Example:
        >>> questions = sample_questions(limit=10, seed=42)
        >>> len(questions)
        10
        >>> # Same seed produces identical results
        >>> questions2 = sample_questions(limit=10, seed=42)
        >>> questions[0].question == questions2[0].question
        True

    """
    # Set seed for reproducibility
    random.seed(seed)

    # Load all questions first (need to materialise for shuffling)
    all_questions = list(iter_questions(limit=None, cache_dir=cache_dir))

    # Shuffle with the seeded random state
    random.shuffle(all_questions)

    # Return limited sample
    return all_questions[:limit]
