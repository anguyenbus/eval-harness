"""
RAGAS adapter for transforming eval-harness output to RAGAS format.

This module provides functionality to transform standard RAG query output
into RAGAS SingleTurnSample format for LLM-judge evaluation.
"""

from __future__ import annotations

from typing import Any, Final

from beartype import beartype
from beartype.typing import Dict, List
from ragas import SingleTurnSample

from eval_harness.metrics.ragas_config import create_ragas_metrics

# Constants
DEFAULT_LLM_PROVIDER: Final[str] = "openai"
DEFAULT_JUDGE_MODEL: Final[str] = "gpt-4o"


@beartype
def transform_to_ragas_sample(
    rag_output: Dict[str, Any],
    reference_answer: str,
) -> SingleTurnSample:
    """
    Transform eval-harness RAG output to RAGAS SingleTurnSample format.

    Maps the eval-harness RAG query output structure to RAGAS SingleTurnSample
    which requires: user_input, retrieved_contexts, response, reference.

    Args:
        rag_output: Dictionary conforming to legal_rag_bench_query_output.schema.json
            with keys: query (with text), answer (with text), retrieved_chunks.
        reference_answer: Reference answer text from the dataset.

    Returns:
        SingleTurnSample instance with RAGAS-compatible format.

    Example:
        >>> rag_output = {
        ...     "query": {"text": "What is the termination clause?"},
        ...     "answer": {"text": "The contract can be terminated..."},
        ...     "retrieved_chunks": [{"text": "Context 1"}, {"text": "Context 2"}]
        ... }
        >>> sample = transform_to_ragas_sample(rag_output, "Reference answer")
        >>> print(sample.user_input)
        'What is the termination clause?'

    """
    # Extract question from query.text
    user_input = rag_output.get("query", {}).get("text", "")

    # Extract retrieved contexts from retrieved_chunks
    retrieved_chunks = rag_output.get("retrieved_chunks", [])
    retrieved_contexts: List[str] = [
        chunk.get("text", "") for chunk in retrieved_chunks if chunk.get("text")
    ]

    # Extract response from answer.text
    response = rag_output.get("answer", {}).get("text", "")

    # Create SingleTurnSample
    return SingleTurnSample(
        user_input=user_input,
        retrieved_contexts=retrieved_contexts,
        response=response,
        reference=reference_answer,
    )


@beartype
class RagasEvaluator:
    """
    RAGAS metrics evaluator for RAG output.

    The RagasEvaluator computes LLM-judge metrics including:
    - Faithfulness: Detects hallucinations in generated answers
    - ContextPrecision: Measures signal-to-noise in retrieved contexts
    - ContextRecall: Evaluates coverage of relevant information
    - AnswerRelevancy: Assesses directness of response to question

    Attributes:
        _metrics: Dictionary of RAGAS metric instances.
        _track_costs: Whether to track token usage costs.

    Example:
        >>> evaluator = RagasEvaluator(llm_provider="openai", judge_model="gpt-4o")
        >>> scores = evaluator.compute_metrics(rag_output, reference_answer)
        >>> print(scores["faithfulness"])

    """

    __slots__ = ("_metrics", "_track_costs")

    def __init__(
        self,
        llm_provider: str = DEFAULT_LLM_PROVIDER,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = 0.0,
        track_costs: bool = False,
    ) -> None:
        """
        Initialize RAGAS evaluator with LLM backend.

        Args:
            llm_provider: LLM provider ("openai" or "bedrock"). Default: "openai".
            judge_model: Judge model name. Default: gpt-4o.
            temperature: Sampling temperature. Default: 0.0.
            track_costs: Whether to track token usage costs. Default: False.

        Raises:
            ValueError: If provider is not supported or API key is missing.

        """
        self._metrics = create_ragas_metrics(
            llm_provider=llm_provider,
            judge_model=judge_model,
            temperature=temperature,
        )
        self._track_costs = track_costs

    @beartype
    def compute_metrics(
        self,
        rag_output: Dict[str, Any],
        reference_answer: str,
    ) -> Dict[str, Any]:
        """
        Compute RAGAS metrics for RAG output.

        Transforms the RAG output to RAGAS format and evaluates with
        all configured metrics (Faithfulness, ContextPrecision, ContextRecall,
        AnswerRelevancy).

        Args:
            rag_output: Dictionary conforming to legal_rag_bench_query_output.schema.json.
            reference_answer: Reference answer text from the dataset.

        Returns:
            Dictionary mapping metric names to scores:
                - faithfulness: float (0-1)
                - context_precision: float (0-1)
                - context_recall: float (0-1)
                - answer_relevancy: float (0-1)

        Example:
            >>> scores = evaluator.compute_metrics(rag_output, "Reference answer")
            >>> print(f"Faithfulness: {scores['faithfulness']}")

        """
        # Transform to RAGAS format
        sample = transform_to_ragas_sample(rag_output, reference_answer)

        # Compute metrics
        results: Dict[str, Any] = {}

        import asyncio

        # Each metric has different ascore signature in RAGAS 0.4+
        # Faithfulness: ascore(user_input, response, retrieved_contexts)
        # ContextPrecision: ascore(user_input, reference, retrieved_contexts)
        # ContextRecall: ascore(user_input, retrieved_contexts, reference)
        # AnswerRelevancy: ascore(user_input, response)

        # Compute Faithfulness
        if "faithfulness" in self._metrics:
            try:
                metric = self._metrics["faithfulness"]
                score_result = asyncio.run(
                    metric.ascore(
                        user_input=sample.user_input,
                        response=sample.response,
                        retrieved_contexts=sample.retrieved_contexts,
                    )
                )
                results["faithfulness"] = float(score_result)
            except Exception as e:
                import sys

                print(f"[ERROR] faithfulness failed: {e}", file=sys.stderr)
                results["faithfulness"] = 0.0

        # Compute ContextPrecision
        if "context_precision" in self._metrics:
            try:
                metric = self._metrics["context_precision"]
                score_result = asyncio.run(
                    metric.ascore(
                        user_input=sample.user_input,
                        reference=sample.reference,
                        retrieved_contexts=sample.retrieved_contexts,
                    )
                )
                results["context_precision"] = float(score_result)
            except Exception as e:
                import sys

                print(f"[ERROR] context_precision failed: {e}", file=sys.stderr)
                results["context_precision"] = 0.0

        # Compute ContextRecall
        if "context_recall" in self._metrics:
            try:
                metric = self._metrics["context_recall"]
                score_result = asyncio.run(
                    metric.ascore(
                        user_input=sample.user_input,
                        retrieved_contexts=sample.retrieved_contexts,
                        reference=sample.reference,
                    )
                )
                results["context_recall"] = float(score_result)
            except Exception as e:
                import sys

                print(f"[ERROR] context_recall failed: {e}", file=sys.stderr)
                results["context_recall"] = 0.0

        # Compute AnswerRelevancy
        if "answer_relevancy" in self._metrics:
            try:
                metric = self._metrics["answer_relevancy"]
                score_result = asyncio.run(
                    metric.ascore(
                        user_input=sample.user_input,
                        response=sample.response,
                    )
                )
                results["answer_relevancy"] = float(score_result)
            except Exception as e:
                import sys

                print(f"[ERROR] answer_relevancy failed: {e}", file=sys.stderr)
                results["answer_relevancy"] = 0.0

        return results
