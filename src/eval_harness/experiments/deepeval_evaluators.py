"""
DeepEval metrics wrapped as Phoenix evaluators.

Uses @create_evaluator decorator to make DeepEval metrics compatible with
Phoenix's run_experiment() API. This preserves DeepEval's LLM-judge quality
while gaining Phoenix's experiment tracking, comparison UI, and schema.

Example:
    >>> from phoenix.client import Client
    >>> from eval_harness.experiments.deepeval_evaluators import (
    ...     create_faithfulness_evaluator,
    ...     create_context_precision_evaluator,
    ... )
    >>>
    >>> client = Client(endpoint="http://localhost:6006")
    >>> dataset = client.datasets.create_dataset(...)
    >>>
    >>> experiment = client.experiments.run_experiment(
    ...     dataset=dataset,
    ...     task=rag_task,
    ...     evaluators=[
    ...         create_faithfulness_evaluator(llm_provider="openai"),
    ...         create_context_precision_evaluator(llm_provider="openai"),
    ...     ],
    ... )

"""

from __future__ import annotations

from typing import Any, Final

from beartype import beartype
from beartype.typing import Callable

# Phoenix evaluator decorator
try:
    from phoenix.evals import create_evaluator
except ImportError:
    create_evaluator = None  # type: ignore[assignment]

# Constants
DEFAULT_JUDGE_MODEL: Final[str] = "gpt-4o-mini"


@beartype
def _suppress_tracing_if_available() -> Any:
    """
    Get Phoenix suppress_tracing context manager if available.

    Prevents DeepEval LLM judge calls from creating noisy child spans.

    """
    try:
        from phoenix.core.tracing import suppress_tracing

        return suppress_tracing()
    except (ImportError, AttributeError):
        # Return no-op context manager
        from contextlib import contextmanager

        @contextmanager
        def _noop_context():
            yield

        return _noop_context()


@beartype
def create_faithfulness_evaluator(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedder: Any = None,
) -> Callable:
    """
    Create a Phoenix evaluator for DeepEval Faithfulness metric.

    Measures hallucination by verifying that generated claims are supported
    by retrieved context.

    Args:
        judge_model: Judge model name.
        embedder: Optional shared embedder instance.

    Returns:
        Phoenix evaluator function compatible with run_experiment().

    """
    if create_evaluator is None:
        raise ImportError("phoenix.evals.create_evaluator not available")

    # Lazy import to avoid circular dependencies
    def _metric_factory():
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        # Initialize metric with configuration
        # Note: temperature is set via environment variable by DeepEval
        metric = FaithfulnessMetric(
            model=judge_model,
            include_reason=True,
        )

        return metric, LLMTestCase

    @create_evaluator(name="faithfulness", direction="maximize")
    def faithfulness_evaluator(
        output: dict[str, Any] | None,
        expected: str | dict[str, str] | None = None,
        input: str | dict[str, str] | None = None,
    ) -> float:
        """
        Evaluate faithfulness using DeepEval.

        Args:
            output: Task output dict with 'answer' and 'retrieval_context' keys.
            expected: Reference answer (string or dict with 'expected' key).
            input: Original input question (string or dict with 'input' key).

        Returns:
            Faithfulness score (0-1, higher is better).

        """
        metric, LLMTestCase = _metric_factory()

        # Extract input string (handle both string and dict formats)
        if isinstance(input, dict):
            input_str = input.get("input", "") or input.get("expected", "")
        else:
            input_str = input or ""

        # Extract expected string (handle both string and dict formats)
        if isinstance(expected, dict):
            expected_str = expected.get("expected", "") or expected.get("output", "")
        else:
            expected_str = expected or ""

        # Extract values from task output dict
        if isinstance(output, dict):
            actual_answer = output.get("answer", "")
            retrieval_context = output.get("retrieval_context", [])
        else:
            actual_answer = str(output) if output else ""
            retrieval_context = []

        # Create DeepEval test case
        test_case = LLMTestCase(
            input=input_str,
            actual_output=actual_answer,
            retrieval_context=retrieval_context,
            expected_output=expected_str,
        )

        # Measure with trace suppression
        with _suppress_tracing_if_available():
            metric.measure(test_case)

        return float(metric.score)

    return faithfulness_evaluator


@beartype
def create_context_precision_evaluator(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedder: Any = None,
) -> Callable:
    """
    Create a Phoenix evaluator for DeepEval ContextualPrecision metric.

    Measures signal-to-noise ratio in retrieved context by counting relevant
    vs irrelevant chunks.

    Args:
        judge_model: Judge model name.
        embedder: Optional shared embedder instance.

    Returns:
        Phoenix evaluator function compatible with run_experiment().

    """
    if create_evaluator is None:
        raise ImportError("phoenix.evals.create_evaluator not available")

    def _metric_factory():
        from deepeval.metrics import ContextualPrecisionMetric
        from deepeval.test_case import LLMTestCase

        metric = ContextualPrecisionMetric(
            model=judge_model,
            include_reason=True,
        )

        return metric, LLMTestCase

    @create_evaluator(name="context_precision", direction="maximize")
    def context_precision_evaluator(
        output: dict[str, Any] | None,
        expected: str | dict[str, str],
        input: str | dict[str, str] | None = None,
    ) -> float:
        """
        Evaluate contextual precision using DeepEval.

        Args:
            output: Task output dict with 'answer' and 'retrieval_context' keys.
            expected: Reference answer (string or dict with 'expected'/'output' key).
            input: Original input question (string or dict with 'input' key).

        Returns:
            Contextual precision score (0-1, higher is better).

        """
        metric, LLMTestCase = _metric_factory()

        # Extract input string (handle both string and dict formats)
        if isinstance(input, dict):
            input_str = input.get("input", "") or input.get("expected", "")
        else:
            input_str = input or ""

        # Extract expected string (handle both string and dict formats)
        if isinstance(expected, dict):
            expected_str = expected.get("expected", "") or expected.get("output", "")
        else:
            expected_str = expected or ""

        # Extract values from task output dict
        if isinstance(output, dict):
            actual_answer = output.get("answer", "")
            retrieval_context = output.get("retrieval_context", [])
        else:
            actual_answer = str(output) if output else ""
            retrieval_context = []

        test_case = LLMTestCase(
            input=input_str,
            actual_output=actual_answer,
            retrieval_context=retrieval_context,
            expected_output=expected_str,
        )

        with _suppress_tracing_if_available():
            metric.measure(test_case)

        return float(metric.score)

    return context_precision_evaluator


@beartype
def create_context_recall_evaluator(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedder: Any = None,
) -> Callable:
    """
    Create a Phoenix evaluator for DeepEval ContextualRecall metric.

    Measures coverage by checking if expected answer information is present
    in retrieved context.

    Args:
        judge_model: Judge model name.
        embedder: Optional shared embedder instance.

    Returns:
        Phoenix evaluator function compatible with run_experiment().

    """
    if create_evaluator is None:
        raise ImportError("phoenix.evals.create_evaluator not available")

    def _metric_factory():
        from deepeval.metrics import ContextualRecallMetric
        from deepeval.test_case import LLMTestCase

        metric = ContextualRecallMetric(
            model=judge_model,
            include_reason=True,
        )

        return metric, LLMTestCase

    @create_evaluator(name="context_recall", direction="maximize")
    def context_recall_evaluator(
        output: dict[str, Any] | None,
        expected: str | dict[str, str],
        input: str | dict[str, str] | None = None,
    ) -> float:
        """
        Evaluate contextual recall using DeepEval.

        Args:
            output: Task output dict with 'answer' and 'retrieval_context' keys.
            expected: Reference answer (string or dict with 'expected'/'output' key).
            input: Original input question (string or dict with 'input' key).

        Returns:
            Contextual recall score (0-1, higher is better).

        """
        metric, LLMTestCase = _metric_factory()

        # Extract input string (handle both string and dict formats)
        if isinstance(input, dict):
            input_str = input.get("input", "") or input.get("expected", "")
        else:
            input_str = input or ""

        # Extract expected string (handle both string and dict formats)
        if isinstance(expected, dict):
            expected_str = expected.get("expected", "") or expected.get("output", "")
        else:
            expected_str = expected or ""

        # Extract values from task output dict
        if isinstance(output, dict):
            actual_answer = output.get("answer", "")
            retrieval_context = output.get("retrieval_context", [])
        else:
            actual_answer = str(output) if output else ""
            retrieval_context = []

        test_case = LLMTestCase(
            input=input_str,
            actual_output=actual_answer,
            retrieval_context=retrieval_context,
            expected_output=expected_str,
        )

        with _suppress_tracing_if_available():
            metric.measure(test_case)

        return float(metric.score)

    return context_recall_evaluator


@beartype
def create_answer_relevancy_evaluator(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedder: Any = None,
) -> Callable:
    """
    Create a Phoenix evaluator for DeepEval AnswerRelevancy metric.

    Measures how directly the generated response addresses the input question.

    Args:
        llm_provider: LLM provider ("openai" or "bedrock").
        judge_model: Judge model name.
        temperature: Sampling temperature.
        embedder: Optional shared embedder instance (required).

    Returns:
        Phoenix evaluator function compatible with run_experiment().

    """
    if create_evaluator is None:
        raise ImportError("phoenix.evals.create_evaluator not available")

    if embedder is None:
        # Need embedder for AnswerRelevancy
        from eval_harness.adapters.embeddings import get_embedder

        embedder = get_embedder(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )

    def _metric_factory():
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        metric = AnswerRelevancyMetric(
            model=judge_model,
            include_reason=True
        )

        return metric, LLMTestCase

    @create_evaluator(name="answer_relevancy", direction="maximize")
    def answer_relevancy_evaluator(
        output: dict[str, Any] | None,
        input: str | dict[str, str],
        expected: str | dict[str, str] | None = None,
    ) -> float:
        """
        Evaluate answer relevancy using DeepEval.

        Args:
            output: Task output dict with 'answer' and 'retrieval_context' keys.
            input: Original input question (string or dict with 'input' key).
            expected: Reference answer (string or dict, optional).

        Returns:
            Answer relevancy score (0-1, higher is better).

        """
        metric, LLMTestCase = _metric_factory()

        # Extract input string (handle both string and dict formats)
        if isinstance(input, dict):
            input_str = input.get("input", "") or input.get("expected", "")
        else:
            input_str = input or ""

        # Extract expected string (handle both string and dict formats)
        if expected:
            if isinstance(expected, dict):
                expected_str = (
                    expected.get("expected", "") or expected.get("output", "")
                )
            else:
                expected_str = expected
        else:
            expected_str = ""

        # Extract values from task output dict
        if isinstance(output, dict):
            actual_answer = output.get("answer", "")
        else:
            actual_answer = str(output) if output else ""

        test_case = LLMTestCase(
            input=input_str,
            actual_output=actual_answer,
            expected_output=expected_str,
        )

        with _suppress_tracing_if_available():
            metric.measure(test_case)

        return float(metric.score)

    return answer_relevancy_evaluator
