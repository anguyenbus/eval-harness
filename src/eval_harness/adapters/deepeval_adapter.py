"""
DeepEval adapter for transforming eval-harness output to DeepEval format.

This module provides functionality to transform standard RAG query output
into DeepEval LLMTestCase format for LLM-judge evaluation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Final

from beartype import beartype
from beartype.typing import Dict, List

# Constants
DEFAULT_LLM_PROVIDER: Final[str] = "openai"
DEFAULT_JUDGE_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_TEMPERATURE: Final[float] = 0.0
DEFAULT_MAX_CONCURRENT: Final[int] = 10


@beartype
def transform_to_deepeval_sample(
    rag_output: Dict[str, Any],
    reference_answer: str,
) -> Any:
    """
    Transform eval-harness RAG output to DeepEval LLMTestCase format.

    Maps the eval-harness RAG query output structure to DeepEval LLMTestCase
    which requires: input, retrieval_context, actual_output, expected_output.

    Args:
        rag_output: Dictionary conforming to legal_rag_bench_query_output.schema.json
            with keys: query (with text), answer (with text), retrieved_chunks.
        reference_answer: Reference answer text from the dataset.

    Returns:
        LLMTestCase instance with DeepEval-compatible format.

    Example:
        >>> rag_output = {
        ...     "query": {"text": "What is the termination clause?"},
        ...     "answer": {"text": "The contract can be terminated..."},
        ...     "retrieved_chunks": [{"text": "Context 1"}, {"text": "Context 2"}]
        ... }
        >>> sample = transform_to_deepeval_sample(rag_output, "Reference answer")
        >>> print(sample.input)
        'What is the termination clause?'

    """
    from deepeval.test_case import LLMTestCase

    # Extract question from query.text -> input
    input_text = rag_output.get("query", {}).get("text", "")

    # Extract retrieved contexts from retrieved_chunks -> retrieval_context
    retrieved_chunks = rag_output.get("retrieved_chunks", [])
    retrieval_context: List[str] = [
        chunk.get("text", "") for chunk in retrieved_chunks if chunk.get("text")
    ]

    # Extract response from answer.text -> actual_output
    actual_output = rag_output.get("answer", {}).get("text", "")

    # Create LLMTestCase
    return LLMTestCase(
        input=input_text,
        retrieval_context=retrieval_context,
        actual_output=actual_output,
        expected_output=reference_answer,
    )


@beartype
class DeepEvalEvaluator:
    """
    DeepEval metrics evaluator for RAG output.

    The DeepEvalEvaluator computes LLM-judge metrics including:
    - Faithfulness: Detects hallucinations in generated answers
    - ContextualPrecision: Measures signal-to-noise in retrieved contexts
    - ContextualRecall: Evaluates coverage of relevant information
    - AnswerRelevancy: Assesses directness of response to question

    Attributes:
        _metrics: Dictionary of DeepEval metric instances.
        _llm_provider: LLM provider name (openai or bedrock).
        _judge_model: Judge model name.
        _temperature: Sampling temperature.
        _max_concurrent: Maximum concurrent evaluations.
        _embedder: Optional shared embedder instance.

    Example:
        >>> evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")
        >>> scores = evaluator.compute_metrics(rag_output, reference_answer)
        >>> print(scores["faithfulness"])

    """

    __slots__ = (
        "_metrics",
        "_llm_provider",
        "_judge_model",
        "_temperature",
        "_max_concurrent",
        "_embedder",
    )

    def __init__(
        self,
        llm_provider: str = DEFAULT_LLM_PROVIDER,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        embedder: Any = None,
    ) -> None:
        """
        Initialize DeepEval evaluator with LLM backend.

        Args:
            llm_provider: LLM provider ("openai" or "bedrock"). Default: "openai".
            judge_model: Judge model name. Default: gpt-4o-mini.
            temperature: Sampling temperature. Default: 0.0.
            max_concurrent: Maximum concurrent evaluations. Default: 10.
            embedder: Optional shared embedder instance. If provided, used for
                AnswerRelevancy metric instead of creating a new one. This
                allows sharing the embedder with RAG retrieval to avoid
                duplicate model loads. Default: None (creates own embedder).

        Raises:
            ValueError: If provider is not supported or API key is missing.

        """
        from eval_harness.metrics.deepeval_config import create_deepeval_metrics

        self._llm_provider = llm_provider
        self._judge_model = judge_model
        self._temperature = temperature
        self._max_concurrent = max_concurrent
        self._embedder = embedder
        self._metrics = create_deepeval_metrics(
            llm_provider=llm_provider,
            judge_model=judge_model,
            temperature=temperature,
            embedder=embedder,
        )

    @beartype
    def compute_metrics(
        self,
        rag_output: Dict[str, Any],
        reference_answer: str,
    ) -> Dict[str, Any]:
        """
        Compute DeepEval metrics for RAG output.

        Transforms the RAG output to DeepEval format and evaluates with
        all configured metrics (Faithfulness, ContextualPrecision,
        ContextualRecall, AnswerRelevancy).

        Args:
            rag_output: Dictionary conforming to legal_rag_bench schema.
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
        # Transform to DeepEval format
        test_case = transform_to_deepeval_sample(rag_output, reference_answer)

        # Compute metrics
        results: Dict[str, Any] = {}

        # Compute Faithfulness
        if "faithfulness" in self._metrics:
            try:
                metric = self._metrics["faithfulness"]
                metric.measure(test_case)
                results["faithfulness"] = float(metric.score)
            except Exception as e:
                import sys

                print(f"[ERROR] faithfulness failed: {e}", file=sys.stderr)
                results["faithfulness"] = 0.0

        # Compute ContextualPrecision
        if "context_precision" in self._metrics:
            try:
                metric = self._metrics["context_precision"]
                metric.measure(test_case)
                results["context_precision"] = float(metric.score)
            except Exception as e:
                import sys

                print(f"[ERROR] context_precision failed: {e}", file=sys.stderr)
                results["context_precision"] = 0.0

        # Compute ContextualRecall
        if "context_recall" in self._metrics:
            try:
                metric = self._metrics["context_recall"]
                metric.measure(test_case)
                results["context_recall"] = float(metric.score)
            except Exception as e:
                import sys

                print(f"[ERROR] context_recall failed: {e}", file=sys.stderr)
                results["context_recall"] = 0.0

        # Compute AnswerRelevancy
        if "answer_relevancy" in self._metrics:
            try:
                metric = self._metrics["answer_relevancy"]
                metric.measure(test_case)
                results["answer_relevancy"] = float(metric.score)
            except Exception as e:
                import sys

                print(f"[ERROR] answer_relevancy failed: {e}", file=sys.stderr)
                results["answer_relevancy"] = 0.0

        return results

    @beartype
    async def async_batch_compute_metrics(
        self,
        rag_outputs: List[Dict[str, Any]],
        reference_answers: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Compute DeepEval metrics for multiple RAG outputs asynchronously.

        Uses asyncio.Semaphore for concurrency control to prevent rate limiting.

        Args:
            rag_outputs: List of dictionaries conforming to legal_rag_bench schema.
            reference_answers: List of reference answer texts from the dataset.

        Returns:
            List of dictionaries mapping metric names to scores,
            maintaining input order.

        Example:
            >>> scores = await evaluator.async_batch_compute_metrics(
            ...     rag_outputs, references
            ... )
            >>> print(f"First query faithfulness: {scores[0]['faithfulness']}")

        """
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def compute_with_semaphore(
            rag_output: Dict[str, Any],
            reference_answer: str,
        ) -> Dict[str, Any]:
            """Compute metrics with semaphore control."""
            async with semaphore:
                # Transform to DeepEval format
                test_case = transform_to_deepeval_sample(rag_output, reference_answer)

                results: Dict[str, Any] = {}

                # Compute all metrics with error handling
                if "faithfulness" in self._metrics:
                    try:
                        metric = self._metrics["faithfulness"]
                        # DeepEval metrics are synchronous, run in thread pool
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, metric.measure, test_case)
                        results["faithfulness"] = float(metric.score)
                    except Exception as e:
                        import sys

                        print(f"[ERROR] faithfulness failed: {e}", file=sys.stderr)
                        results["faithfulness"] = 0.0

                if "context_precision" in self._metrics:
                    try:
                        metric = self._metrics["context_precision"]
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, metric.measure, test_case)
                        results["context_precision"] = float(metric.score)
                    except Exception as e:
                        import sys

                        print(f"[ERROR] context_precision failed: {e}", file=sys.stderr)
                        results["context_precision"] = 0.0

                if "context_recall" in self._metrics:
                    try:
                        metric = self._metrics["context_recall"]
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, metric.measure, test_case)
                        results["context_recall"] = float(metric.score)
                    except Exception as e:
                        import sys

                        print(f"[ERROR] context_recall failed: {e}", file=sys.stderr)
                        results["context_recall"] = 0.0

                if "answer_relevancy" in self._metrics:
                    try:
                        metric = self._metrics["answer_relevancy"]
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, metric.measure, test_case)
                        results["answer_relevancy"] = float(metric.score)
                    except Exception as e:
                        import sys

                        print(f"[ERROR] answer_relevancy failed: {e}", file=sys.stderr)
                        results["answer_relevancy"] = 0.0

                return results

        # Process all queries concurrently with semaphore control
        tasks = [
            compute_with_semaphore(rag_output, reference_answer)
            for rag_output, reference_answer in zip(
                rag_outputs, reference_answers, strict=True
            )
        ]

        return await asyncio.gather(*tasks)

    @beartype
    def _extract_verdicts(self, metric: Any) -> List[Dict[str, Any]]:
        """
        Extract verdicts from a DeepEval metric.

        Verdicts contain per-chunk or per-claim judgments with rationale.

        Args:
            metric: DeepEval metric instance after measure() has been called.

        Returns:
            List of verdict dictionaries with verdict, reason, and optionally
            chunk_id.

        """
        verdicts = getattr(metric, "verdicts", None)
        if not verdicts:
            return []

        extracted = []
        for v in verdicts:
            try:
                verdict_dict = v.model_dump() if hasattr(v, "model_dump") else dict(v)
                extracted.append(verdict_dict)
            except Exception:
                # Fallback to string representation if model_dump fails
                extracted.append({"verdict": str(v)})

        return extracted

    @beartype
    def _extract_claims_truths(self, metric: Any) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract claims and truths from a DeepEval metric (Faithfulness).

        Args:
            metric: DeepEval metric instance after measure() has been called.

        Returns:
            Dictionary with keys: claims, truths, statements.

        """
        result = {}

        for attr in ["claims", "truths", "statements"]:
            items = getattr(metric, attr, None)
            if items:
                try:
                    result[attr] = [
                        i.model_dump() if hasattr(i, "model_dump") else dict(i)
                        for i in items
                    ]
                except Exception:
                    result[attr] = [{"text": str(i)} for i in items]

        return result

    @beartype
    def compute_metrics_with_timing(
        self,
        rag_output: Dict[str, Any],
        reference_answer: str,
    ) -> Dict[str, Any]:
        """
        Compute DeepEval metrics with timing information.

        Same as compute_metrics but includes metric_computation_time_ms
        in results.

        Args:
            rag_output: Dictionary conforming to legal_rag_bench schema.
            reference_answer: Reference answer text from the dataset.

        Returns:
            Dictionary mapping metric names to scores plus timing metadata:
                - faithfulness: float (0-1)
                - context_precision: float (0-1)
                - context_recall: float (0-1)
                - answer_relevancy: float (0-1)
                - metric_computation_time_ms: float

        """
        start_time = time.time()
        results = self.compute_metrics(rag_output, reference_answer)
        end_time = time.time()

        results["metric_computation_time_ms"] = (end_time - start_time) * 1000
        return results

    @beartype
    def compute_metrics_with_reasoning(
        self,
        rag_output: Dict[str, Any],
        reference_answer: str,
    ) -> Dict[str, Any]:
        """
        Compute DeepEval metrics with full reasoning extraction.

        Extracts three layers of reasoning:
        - L1: metric.reason (overall explanation)
        - L2: metric.verdicts (per-chunk yes/no with rationale)
        - L3: metric.claims/truths (detailed claim analysis for Faithfulness)

        Args:
            rag_output: Dictionary conforming to legal_rag_bench schema.
            reference_answer: Reference answer text from the dataset.

        Returns:
            Dictionary with:
                - scores: dict of metric name -> float score
                - reasoning: dict of metric name -> dict with:
                    - reason: str (L1 overall explanation)
                    - verdicts: list[dict] (L2 per-chunk judgments)
                    - claims/truths: list[dict] (L3 breakdown for faithfulness)

        """
        # Transform to DeepEval format
        test_case = transform_to_deepeval_sample(rag_output, reference_answer)

        scores: Dict[str, Any] = {}
        reasoning: Dict[str, Any] = {}

        # Compute Faithfulness with reasoning
        if "faithfulness" in self._metrics:
            try:
                metric = self._metrics["faithfulness"]
                metric.measure(test_case)
                scores["faithfulness"] = float(metric.score)

                # Extract reasoning
                metric_reasoning = {
                    "reason": getattr(metric, "reason", ""),
                }
                # Add claims/truths for Faithfulness (L3 reasoning)
                claims_truths = self._extract_claims_truths(metric)
                metric_reasoning.update(claims_truths)
                reasoning["faithfulness"] = metric_reasoning

            except Exception as e:
                import sys

                print(f"[ERROR] faithfulness failed: {e}", file=sys.stderr)
                scores["faithfulness"] = 0.0
                reasoning["faithfulness"] = {"reason": f"ERROR: {e}", "verdicts": []}

        # Compute ContextualPrecision with reasoning
        if "context_precision" in self._metrics:
            try:
                metric = self._metrics["context_precision"]
                metric.measure(test_case)
                scores["context_precision"] = float(metric.score)

                # Extract reasoning - verdicts show which chunks were relevant
                reasoning["context_precision"] = {
                    "reason": getattr(metric, "reason", ""),
                    "verdicts": self._extract_verdicts(metric),
                }

            except Exception as e:
                import sys

                print(f"[ERROR] context_precision failed: {e}", file=sys.stderr)
                scores["context_precision"] = 0.0
                reasoning["context_precision"] = {
                    "reason": f"ERROR: {e}",
                    "verdicts": [],
                }

        # Compute ContextualRecall with reasoning
        if "context_recall" in self._metrics:
            try:
                metric = self._metrics["context_recall"]
                metric.measure(test_case)
                scores["context_recall"] = float(metric.score)

                reasoning["context_recall"] = {
                    "reason": getattr(metric, "reason", ""),
                    "verdicts": self._extract_verdicts(metric),
                }

            except Exception as e:
                import sys

                print(f"[ERROR] context_recall failed: {e}", file=sys.stderr)
                scores["context_recall"] = 0.0
                reasoning["context_recall"] = {"reason": f"ERROR: {e}", "verdicts": []}

        # Compute AnswerRelevancy with reasoning
        if "answer_relevancy" in self._metrics:
            try:
                metric = self._metrics["answer_relevancy"]
                metric.measure(test_case)
                scores["answer_relevancy"] = float(metric.score)

                reasoning["answer_relevancy"] = {
                    "reason": getattr(metric, "reason", ""),
                }

            except Exception as e:
                import sys

                print(f"[ERROR] answer_relevancy failed: {e}", file=sys.stderr)
                scores["answer_relevancy"] = 0.0
                reasoning["answer_relevancy"] = {"reason": f"ERROR: {e}"}

        return {"scores": scores, "reasoning": reasoning}
