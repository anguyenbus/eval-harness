"""
Span generator runner for synthetic trace generation.

This module orchestrates question iteration and stub pipeline execution
to emit OpenInference-compliant spans to Phoenix.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
from openinference.semconv.trace import OpenInferenceSpanKindValues

from eval_harness.stubs.span_generator.config import GeneratorConfig
from eval_harness.stubs.span_generator.loader import GeneratorQuestion, iter_questions
from eval_harness.stubs.span_generator.span_schema import (
    ANSWER_RELEVANCY,
    CASE_ID,
    CONTEXT_PRECISION,
    CONTEXT_RECALL,
    EVAL_HARNESS_PREFIX,
    FAITHFULNESS,
    GENERATOR_VERSION,
    GENERATOR_VERSION_VALUE,
    INPUT_VALUE,
    LATENCY_GENERATION_MS,
    LATENCY_RETRIEVAL_MS,
    LATENCY_TOTAL_MS,
    METADATA,
    METADATA_KEY_EXPECTED_ANSWER,
    METADATA_KEY_EXPECTED_PASSAGE_ID,
    METADATA_KEY_SOURCE_QUESTION_ID,
    METADATA_KEY_STUB_PIPELINE_VERSION,
    METADATA_KEY_SYNTHETIC_RUN_ID,
    METADATA_KEY_TENANT_ID_HASHED,
    OUTPUT_VALUE,
    SESSION_ID,
    SOURCE_DATASET,
    SOURCE_DATASET_VALUE,
    SYNTHETIC_MARKER,
    SYNTHETIC_MARKER_VALUE,
    TENANT_ID_HASHED,
)
from eval_harness.stubs.span_generator.tracer import setup_tracer

CHAIN = OpenInferenceSpanKindValues.CHAIN


@beartype
@dataclass(frozen=True)
class GeneratorResult:
    """
    Result of span generation run.

    Attributes:
        successes: Number of successfully processed questions.
        failures: Number of failed questions.
        run_id: Unique run ID for this generation batch.

    """

    successes: int
    failures: int
    run_id: str


@beartype
def _build_metadata_json(
    question: GeneratorQuestion,
    run_id: str,
    pipeline_version: str,
) -> str:
    """
    Build canonical metadata JSON String attribute.

    Encodes ground truth data for evaluators that need to access
    expected answers and passage IDs.

    Args:
        question: Generator question with ground truth data.
        run_id: Unique run ID for this generation batch.
        pipeline_version: Version identifier for stub pipeline.

    Returns:
        JSON string with metadata fields.

    """
    metadata_dict = {
        METADATA_KEY_SOURCE_QUESTION_ID: question.id,
        METADATA_KEY_EXPECTED_PASSAGE_ID: question.relevant_passage_id,
        METADATA_KEY_EXPECTED_ANSWER: question.expected_answer,
        METADATA_KEY_SYNTHETIC_RUN_ID: run_id,
        METADATA_KEY_STUB_PIPELINE_VERSION: pipeline_version,
        METADATA_KEY_TENANT_ID_HASHED: question.tenant_id_hashed,
    }
    return json.dumps(metadata_dict)


@beartype
def run_generator(
    config: GeneratorConfig,
    corpus_dir: Path,
    limit: int | None = None,
    pipeline_version: str = "0.1.0",
) -> GeneratorResult:
    """
    Run synthetic span generation.

    Iterates over questions, creates root CHAIN spans with all required
    OpenInference attributes, and calls the stub RAG pipeline to emit
    child spans (EMBEDDING, RETRIEVER, LLM).

    Args:
        config: Generator configuration.
        corpus_dir: Path to document corpus directory.
        limit: Maximum number of questions to process.
        pipeline_version: Version identifier for stub pipeline.

    Returns:
        GeneratorResult with success/failure counts and run_id.

    Example:
        >>> from pathlib import Path
        >>> from eval_harness.stubs.span_generator.config import load_generator_config
        >>> config = load_generator_config()
        >>> result = run_generator(config, Path("data/corpus"), limit=5)
        >>> print(f"Generated {result.successes} traces")

    """
    # Generate unique run_id for this invocation
    run_id = str(uuid.uuid4())

    # Setup tracer
    tracer_provider, tracer = setup_tracer(
        phoenix_endpoint=config.phoenix_endpoint,
        project_name=config.project_name,
        batch=config.batch_export,
    )

    if tracer is None:
        return GeneratorResult(successes=0, failures=0, run_id=run_id)

    successes = 0
    failures = 0

    try:
        # Import stub pipeline
        # Ensure instrumentation is applied (reload once, not per-question)
        import importlib

        import eval_harness.stubs.rag.embedder as embedder_module
        import eval_harness.stubs.rag.generator as generator_module
        from eval_harness.stubs.rag.chromadb_query import query

        importlib.reload(embedder_module)
        importlib.reload(generator_module)

        # Create session_id for grouping
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_id = f"synth-{timestamp}"

        # Iterate over questions
        for question in iter_questions(limit=limit, seed=config.seed):
            try:
                # Create root CHAIN span with all required attributes
                with tracer.start_as_current_span(
                    name="synthetic_rag_query",
                    openinference_span_kind=CHAIN,
                ) as span:
                    # Set required OpenInference attributes
                    span.set_attribute(INPUT_VALUE, question.question)
                    span.set_attribute(OUTPUT_VALUE, "")  # Will be set by pipeline
                    span.set_attribute(SESSION_ID, session_id)

                    # Set vendor-prefixed queryable attributes
                    span.set_attribute(SYNTHETIC_MARKER, SYNTHETIC_MARKER_VALUE)
                    span.set_attribute(
                        f"{EVAL_HARNESS_PREFIX}.{SOURCE_DATASET}",
                        SOURCE_DATASET_VALUE,
                    )
                    span.set_attribute(
                        f"{EVAL_HARNESS_PREFIX}.{GENERATOR_VERSION}",
                        GENERATOR_VERSION_VALUE,
                    )
                    span.set_attribute(
                        f"{EVAL_HARNESS_PREFIX}.{CASE_ID}",
                        question.case_id,
                    )
                    span.set_attribute(
                        f"{EVAL_HARNESS_PREFIX}.{TENANT_ID_HASHED}",
                        question.tenant_id_hashed,
                    )

                    # Encode ground truth in canonical metadata JSON
                    metadata_json = _build_metadata_json(
                        question, run_id, pipeline_version
                    )
                    span.set_attribute(METADATA, metadata_json)

                    # Execute stub pipeline query (child spans auto-instrumented)
                    rag_output = query(
                        question=question.question,
                        corpus_dir=corpus_dir,
                        top_k=5,
                        phoenix_trace_id=None,  # Handled by our span context
                    )

                    # Update output with actual answer
                    answer_text = rag_output.get("answer", {}).get("text", "")
                    span.set_attribute(OUTPUT_VALUE, answer_text)

                    # Store latency metrics from RAG pipeline
                    timings = rag_output.get("timings_ms", {})
                    span.set_attribute(
                        LATENCY_RETRIEVAL_MS, timings.get("retrieval", 0.0)
                    )
                    span.set_attribute(
                        LATENCY_GENERATION_MS, timings.get("generation", 0.0)
                    )
                    span.set_attribute(LATENCY_TOTAL_MS, timings.get("total", 0.0))

                    # Compute and store all RAG quality metrics
                    # IMPORTANT: Production systems MUST store evaluation metrics as
                    # span attributes to enable replay evaluation.
                    # Without stored scores, replay evaluation cannot compare
                    # new approaches against baseline.
                    from eval_harness.adapters.deepeval_adapter import (
                        DeepEvalEvaluator,
                    )

                    evaluator = DeepEvalEvaluator(
                        llm_provider="openai",
                        judge_model="gpt-4o-mini",
                        temperature=0.0,
                        max_concurrent=1,
                    )
                    metric_result = evaluator.compute_metrics_with_reasoning(
                        rag_output, question.expected_answer
                    )
                    scores = metric_result["scores"]
                    span.set_attribute(FAITHFULNESS, scores.get("faithfulness", 0.0))
                    span.set_attribute(
                        CONTEXT_PRECISION, scores.get("context_precision", 0.0)
                    )
                    span.set_attribute(
                        CONTEXT_RECALL, scores.get("context_recall", 0.0)
                    )
                    span.set_attribute(
                        ANSWER_RELEVANCY, scores.get("answer_relevancy", 0.0)
                    )

                    # Print metrics summary
                    import sys

                    print(
                        f"  [{successes + failures + 1}] "
                        f"faithfulness={scores.get('faithfulness', 0):.2f} "
                        f"context_precision={scores.get('context_precision', 0):.2f} "
                        f"context_recall={scores.get('context_recall', 0):.2f} "
                        f"answer_relevancy={scores.get('answer_relevancy', 0):.2f} "
                        f"latency_total={timings.get('total', 0):.0f}ms"
                    )

                    successes += 1

            except Exception as e:
                failures += 1
                import sys

                print(
                    f"[ERROR] Failed to generate span for question {question.id}: {e}",
                    file=sys.stderr,
                )
                # Continue processing other questions

    finally:
        # Force flush to ensure spans are exported
        if tracer_provider:
            tracer_provider.force_flush()

    return GeneratorResult(
        successes=successes,
        failures=failures,
        run_id=run_id,
    )
