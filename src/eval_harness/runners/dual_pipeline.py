"""
Dual pipeline runner for side-by-side RAG comparison.

This module implements the DualPipelineRunner class that executes baseline
and candidate RAG configurations on identical questions for paired comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype

from eval_harness.adapters.rag_adapter import RagAdapter


@beartype
class DualPipelineRunner:
    """
    Runner for side-by-side RAG comparison.

    Executes baseline and candidate RAG pipelines on identical questions
    to enable paired statistical comparison.

    Attributes:
        _baseline_chunk_size: Chunk size for baseline pipeline.
        _baseline_overlap: Chunk overlap for baseline pipeline.
        _candidate_chunk_size: Chunk size for candidate pipeline.
        _candidate_overlap: Chunk overlap for candidate pipeline.

    Example:
        >>> runner = DualPipelineRunner(
        ...     baseline_chunk_size=512,
        ...     baseline_overlap=0,
        ...     candidate_chunk_size=512,
        ...     candidate_overlap=150,
        ... )
        >>> results = runner.run_comparison(
        ...     questions=["What is X?"],
        ...     corpus_dir=Path("corpus"),
        ... )

    """

    __slots__ = (
        "_baseline_chunk_size",
        "_baseline_overlap",
        "_candidate_chunk_size",
        "_candidate_overlap",
        "_adapter_factory",
    )

    def __init__(
        self,
        baseline_chunk_size: int,
        baseline_overlap: int,
        candidate_chunk_size: int,
        candidate_overlap: int,
        adapter_factory: Any | None = None,
    ) -> None:
        """
        Initialize the dual pipeline runner.

        Args:
            baseline_chunk_size: Chunk size for baseline pipeline.
            baseline_overlap: Chunk overlap for baseline pipeline.
            candidate_chunk_size: Chunk size for candidate pipeline.
            candidate_overlap: Chunk overlap for candidate pipeline.
            adapter_factory: Optional factory function for creating adapters.
                Used for testing dependency injection.

        """
        self._baseline_chunk_size = baseline_chunk_size
        self._baseline_overlap = baseline_overlap
        self._candidate_chunk_size = candidate_chunk_size
        self._candidate_overlap = candidate_overlap
        self._adapter_factory = adapter_factory or _default_adapter_factory

    def run_comparison(
        self,
        questions: list[str],
        corpus_dir: Path,
        top_k: int = 5,
        naive_mode: bool = False,
        evaluator: Any | None = None,
    ) -> dict[str, Any]:
        """
        Run baseline and candidate pipelines on questions.

        Executes both pipelines on identical questions (paired mode) or
        different random samples (naive mode).

        Args:
            questions: List of question strings to evaluate.
            corpus_dir: Path to document corpus directory.
            top_k: Number of chunks to retrieve. Default: 5.
            naive_mode: If True, use different question sets for each pipeline.
                Demonstrates uncontrolled variation. Default: False.
            evaluator: Optional evaluator instance. If None, creates DeepEval.

        Returns:
            Dictionary with baseline_scores, candidate_scores, and per_question
            metrics including delta.

        """
        # Import here to avoid circular dependencies
        from eval_harness.adapters.embeddings import get_embedder
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        # Initialize evaluator
        if evaluator is None:
            evaluator = DeepEvalEvaluator(
                llm_provider="openai",
                judge_model="gpt-4o-mini",
                temperature=0.0,
                max_concurrent=4,
            )

        # Initialize shared embedder
        embedder = get_embedder(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )

        # Prepare questions for each pipeline
        if naive_mode:
            # Naive mode: use different subsets
            import random

            mid = len(questions) // 2
            baseline_questions = questions[:mid]
            candidate_questions = questions[mid:]
        else:
            # Paired mode: identical questions
            baseline_questions = questions
            candidate_questions = questions

        # Get adapters with different chunking configs
        baseline_adapter = self._adapter_factory(
            chunk_size=self._baseline_chunk_size,
            chunk_overlap=self._baseline_overlap,
            embedder=embedder,
            top_k=top_k,
        )
        candidate_adapter = self._adapter_factory(
            chunk_size=self._candidate_chunk_size,
            chunk_overlap=self._candidate_overlap,
            embedder=embedder,
            top_k=top_k,
        )

        # Run baseline pipeline
        baseline_scores = []
        for question in baseline_questions:
            try:
                output = baseline_adapter.query(question, corpus_dir)
                generated_answer = output.get("answer", {}).get("text", "")
                metric_result = evaluator.compute_metrics_with_reasoning(
                    output, ""  # No gold answer for faithfulness
                )
                score = metric_result["scores"].get("faithfulness", 0.0)
                baseline_scores.append(score)
            except Exception:
                baseline_scores.append(0.0)

        # Run candidate pipeline
        candidate_scores = []
        for question in candidate_questions:
            try:
                output = candidate_adapter.query(question, corpus_dir)
                generated_answer = output.get("answer", {}).get("text", "")
                metric_result = evaluator.compute_metrics_with_reasoning(
                    output, ""  # No gold answer for faithfulness
                )
                score = metric_result["scores"].get("faithfulness", 0.0)
                candidate_scores.append(score)
            except Exception:
                candidate_scores.append(0.0)

        # Build per-question metrics (paired mode only)
        per_question = []
        if not naive_mode:
            for i, (q, b_score, c_score) in enumerate(
                zip(questions, baseline_scores, candidate_scores, strict=True)
            ):
                per_question.append({
                    "question_id": f"q_{i:04d}",
                    "question": q,
                    "baseline_score": b_score,
                    "candidate_score": c_score,
                    "delta": c_score - b_score,
                })

        return {
            "baseline_scores": baseline_scores,
            "candidate_scores": candidate_scores,
            "per_question": per_question,
            "baseline_config": {
                "chunk_size": self._baseline_chunk_size,
                "overlap": self._baseline_overlap,
            },
            "candidate_config": {
                "chunk_size": self._candidate_chunk_size,
                "overlap": self._candidate_overlap,
            },
            "naive_mode": naive_mode,
        }


def _default_adapter_factory(
    chunk_size: int,
    chunk_overlap: int,
    embedder: Any,
    top_k: int = 5,
) -> RagAdapter:
    """
    Default factory for creating RAG adapters with specific chunking.

    Args:
        chunk_size: Chunk size for this adapter.
        chunk_overlap: Chunk overlap for this adapter.
        embedder: Shared embedder instance.
        top_k: Number of chunks to retrieve.

    Returns:
        Configured RagAdapter instance.

    """
    from eval_harness.stubs.rag.chunking import ConfigurableChunker

    # Create chunker with specific configuration
    chunker = ConfigurableChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Wrap query function with custom chunking
    def custom_query(
        question: str, corpus_dir: Path, embedder: Any = embedder
    ) -> dict[str, Any]:
        from eval_harness.stubs.rag.chromadb_query import query as chromadb_query

        # Patch the FixedChunker import in chromadb_query module
        import eval_harness.stubs.rag.chunker as chunker_module
        original_fixed_chunker = chunker_module.FixedChunker

        # Temporarily replace with our configured chunker
        class TempFixedChunker:
            def __init__(self) -> None:
                self._chunker = chunker

            def chunk(self, doc_id: str, text: str) -> list[dict]:
                return self._chunker.chunk(doc_id, text)

        chunker_module.FixedChunker = TempFixedChunker  # type: ignore

        try:
            return chromadb_query(
                question=question,
                corpus_dir=corpus_dir,
                top_k=top_k,
                force_reingest=False,
                embedder=embedder,
            )
        finally:
            # Restore original
            chunker_module.FixedChunker = original_fixed_chunker

    return RagAdapter(query_callable=custom_query, embedder=embedder)
