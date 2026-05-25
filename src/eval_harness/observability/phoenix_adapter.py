"""
Phoenix adapter for RAG pipeline observability.

Uses phoenix.otel.register() to configure OpenTelemetry OTLP export via gRPC.

KEY INSIGHT: OpenTelemetry parent-child relationships are established through
"current span" context. When you use start_as_current_span() within another
span's context, it automatically becomes a child.

Our challenge: The RAG pipeline is spread across multiple function calls.
Solution: Use tracer.start_as_current_span() which properly propagates context.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from beartype import beartype

if TYPE_CHECKING:
    pass

from openinference.semconv.trace import OpenInferenceSpanKindValues

# OpenInference span kind values
CHAIN = OpenInferenceSpanKindValues.CHAIN
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
LLM = OpenInferenceSpanKindValues.LLM
EVALUATOR = OpenInferenceSpanKindValues.EVALUATOR

# Constants
DEFAULT_ENDPOINT: Final[str] = "http://localhost:6006"
DEFAULT_PROJECT_NAME: Final[str] = "eval-harness"


@beartype
class PhoenixAdapter:
    """
    Phoenix adapter for RAG pipeline observability.

    Creates span hierarchy: eval_run (root) -> rag_query -> retrieval,
    generation, evaluation.

    IMPORTANT: For proper parent-child relationships, child spans must be created
    while the parent span is the "current" span in OpenTelemetry context.

    Example usage with evaluation run grouping:
        >>> with adapter.eval_run_span("legal-rag-bench-nano", num_questions=10):
        ...     for query in dataset:
        ...         with adapter.rag_query_span(query) as trace_id:
        ...             adapter.retrieval_span(trace_id, ...)
        ...             adapter.evaluation_span(trace_id, ...)

    """

    __slots__ = (
        "_endpoint",
        "_project_name",
        "_enabled",
        "_export_path",
        "_tracer_provider",
        "_tracer",
        "_evaluations",
        "_active_root_span",  # Track root span for non-context-manager usage
        "_active_eval_run",  # Track eval run parent span
        "_current_session_id",  # Track session_id for grouping
    )

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        project_name: str = DEFAULT_PROJECT_NAME,
        enabled: bool = True,
        export_path: Path | None = None,
    ) -> None:
        """
        Initialize Phoenix adapter.

        Args:
            endpoint: Phoenix UI endpoint (e.g., http://localhost:6006).
                Internally converted to gRPC endpoint (http://localhost:4317).
            project_name: Phoenix project name for grouping traces.
            enabled: Whether to enable Phoenix tracing.
            export_path: Fallback path for Parquet export when Phoenix unavailable.

        """
        self._endpoint: str = self._validate_endpoint(endpoint)
        self._project_name: str = project_name
        self._enabled: bool = enabled
        self._export_path: Path | None = export_path
        self._tracer_provider: Any = None
        self._tracer: Any = None
        self._evaluations: list[dict[str, Any]] = []
        self._active_root_span: Any = None
        self._active_eval_run: Any = None
        self._current_session_id: str | None = None

        if enabled:
            self._initialize()

    def _validate_endpoint(self, endpoint: str) -> str:
        """Validate Phoenix endpoint URL format."""
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid Phoenix endpoint URL: {endpoint}. "
                "Endpoint must start with http:// or https://"
            )
        return endpoint

    def _get_otlp_endpoint(self, ui_endpoint: str) -> str:
        """
        Convert UI endpoint to gRPC endpoint (Phoenix's recommended default).

        Phoenix accepts OTLP via:
        - HTTP at http://localhost:6006/v1/traces (UI port + path)
        - gRPC at http://localhost:4317 (separate gRPC listener)

        We use gRPC (4317) as it's Phoenix's default and more efficient.
        """
        import re

        # Replace UI port (6006) with gRPC port (4317)
        match = re.match(r"(https?://[^:]+):\d+", ui_endpoint)
        if match:
            return f"{match.group(1)}:4317"
        return "http://localhost:4317"

    def _initialize(self) -> None:
        """Initialize OpenTelemetry tracer using Phoenix register()."""
        try:
            from phoenix.otel import register

            # Use gRPC endpoint (Phoenix's recommended default)
            otlp_endpoint = self._get_otlp_endpoint(self._endpoint)

            # Register with Phoenix: batch processing, no global registration, quiet
            self._tracer_provider = register(
                endpoint=otlp_endpoint,
                project_name=self._project_name,
                protocol="grpc",
                batch=True,  # Use BatchSpanProcessor (production-ready)
                set_global_tracer_provider=False,  # Don't set global default
                verbose=False,  # Suppress verbose output
            )

            self._tracer = self._tracer_provider.get_tracer("eval-harness")

            # Instrument OpenAI to capture RAGAS's internal LLM calls
            self._instrument_openai()

        except Exception as e:
            import sys

            print(
                f"[WARN] Phoenix initialization failed: {e}. "
                "Traces will be buffered to Parquet.",
                file=sys.stderr,
            )
            self._tracer = None

    def _instrument_openai(self) -> None:
        """
        Instrument OpenAI to capture RAGAS internal LLM judge calls.

        NOTE: RAGAS currently only supports OpenAI as the LLM backend.
        When Bedrock support is added, we'll need to add Anthropic instrumentor.
        """
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            instrumentor = OpenAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
                print("[INFO] OpenAI instrumentor enabled")
                print("       RAGAS internal LLM calls will appear in Phoenix traces")
        except Exception as e:
            import sys

            print(f"[WARN] OpenAI instrumentation failed: {e}", file=sys.stderr)

    @beartype
    def is_connected(self) -> bool:
        """Check if Phoenix tracer is available."""
        return self._tracer is not None

    @contextmanager
    def eval_run_span(
        self,
        run_name: str,
        num_questions: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for evaluation run span (groups all queries).

        This creates a parent span that groups all RAG query traces.
        In Phoenix UI, you'll see one "eval_run" trace with all queries as children.

        Usage:
            with adapter.eval_run_span("legal-rag-bench-nano", num_questions=10):
                for query in dataset:
                    with adapter.rag_query_span(query) as trace_id:
                        ...

        Args:
            run_name: Name of the evaluation run.
            num_questions: Number of questions in the dataset.
            metadata: Additional metadata (slice_name, top_k, etc.)

        Yields:
            run_id: Unique identifier for this evaluation run.

        """
        run_id = str(uuid.uuid4())

        if self._tracer:
            # Set session_id for grouping in Phoenix UI
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._current_session_id = f"eval-{run_name}-{timestamp}"

            with self._tracer.start_as_current_span(
                name="eval_run", openinference_span_kind=CHAIN
            ) as span:
                span.set_attribute("eval.run_name", run_name)
                span.set_attribute("eval.num_questions", num_questions)
                span.set_attribute("eval.session_id", self._current_session_id)
                if metadata:
                    for key, value in metadata.items():
                        span.set_attribute(f"eval.{key}", str(value))
                self._active_eval_run = span
                try:
                    yield run_id
                finally:
                    self._active_eval_run = None
                    self._current_session_id = None
        else:
            yield run_id

    @contextmanager
    def rag_query_span(self, question: str):
        """
        Context manager for root RAG query span.

        This establishes the root span as the "current" span. Any child spans
        created within this context will automatically become children.

        If called within eval_run_span context, becomes a child of eval_run.

        Usage:
            with adapter.rag_query_span("What is contract law?") as trace_id:
                adapter.retrieval_span(trace_id, ...)
                adapter.generation_span(trace_id, ...)

        Yields:
            trace_id: Unique identifier for this trace.

        """
        trace_id = str(uuid.uuid4())

        if self._tracer:
            from openinference.instrumentation import using_attributes

            # Add session_id from parent eval_run if available
            if self._current_session_id:
                with using_attributes(session_id=self._current_session_id):
                    with self._tracer.start_as_current_span(
                        name="rag_query", openinference_span_kind=CHAIN
                    ) as span:
                        span.set_attribute("question", question)
                        span.set_attribute("input", question)
                        self._active_root_span = span
                        try:
                            yield trace_id
                        finally:
                            self._active_root_span = None
            else:
                with self._tracer.start_as_current_span(
                    name="rag_query", openinference_span_kind=CHAIN
                ) as span:
                    span.set_attribute("question", question)
                    span.set_attribute("input", question)
                    self._active_root_span = span
                    try:
                        yield trace_id
                    finally:
                        self._active_root_span = None
        else:
            yield trace_id

    @beartype
    def start_rag_query_span(self, question: str) -> str:
        """
        Start a root RAG query span (non-context-manager version).

        NOTE: For proper parent-child relationships in Phoenix UI, prefer
        using the rag_query_span() context manager. This version creates
        a root span but child spans won't automatically become children
        unless they're called while this span is active.

        Args:
            question: The user question.

        Returns:
            trace_id: Unique identifier for this trace.

        """
        trace_id = str(uuid.uuid4())

        if self._tracer:
            span = self._tracer.start_span(
                name="rag_query", openinference_span_kind=CHAIN
            )
            span.set_attribute("question", question)
            span.set_attribute("input", question)
            self._active_root_span = span

        return trace_id

    @beartype
    def start_retrieval_span(
        self,
        trace_id: str,
        query_text: str,
        chunks: list[dict[str, Any]],
        k: int,
        timing_ms: int | float,
    ) -> None:
        """
        Create retrieval span (RETRIEVER kind).

        If called within rag_query_span context, becomes a child span automatically.

        Args:
            trace_id: Parent trace ID (for tracking).
            query_text: The query text used for retrieval.
            chunks: Retrieved chunks with doc_id, text, score.
            k: Number of chunks retrieved.
            timing_ms: Retrieval time.

        """
        if not self._tracer:
            return

        with self._tracer.start_as_current_span(
            name="retrieval", openinference_span_kind=RETRIEVER
        ) as span:
            span.set_attribute("input.value", query_text)
            span.set_attribute("retrieval.k", k)
            span.set_attribute("retrieval.timing_ms", timing_ms)

            # OpenInference retrieval attributes
            for i, chunk in enumerate(chunks[:k]):
                text = chunk.get("text", "")
                doc_id = chunk.get("doc_id", f"doc_{i}")
                score = chunk.get("score", 0.0)

                span.set_attribute(f"retrieval.documents.{i}.document.content", text)
                span.set_attribute(f"retrieval.documents.{i}.document.id", doc_id)
                span.set_attribute(f"retrieval.documents.{i}.score", float(score))

    @beartype
    def start_generation_span(
        self,
        trace_id: str,
        model: str,
        prompt: str,
        tokens: int = 0,
        timing_ms: int | float = 0,
    ) -> None:
        """
        Create generation span (CLIENT kind - calls LLM API).

        If called within rag_query_span context, becomes a child span automatically.

        Args:
            trace_id: Parent trace ID (for tracking).
            model: LLM model name.
            prompt: The prompt text.
            tokens: Token count (total = prompt + completion).
            timing_ms: Generation time.

        """
        if not self._tracer:
            return

        with self._tracer.start_as_current_span(
            name="generation", openinference_span_kind=LLM
        ) as span:
            span.set_attribute("llm.model_name", model)
            span.set_attribute("llm.token_count.total", tokens)
            span.set_attribute("llm.timing_ms", timing_ms)
            span.set_attribute("llm.prompt_length", len(prompt))
            invocation_params = json.dumps({"model": model, "temperature": 0})
            span.set_attribute("llm.invocation_parameters", invocation_params)

    @beartype
    def start_evaluation_span(
        self,
        trace_id: str,
        evaluation_metrics: dict[str, float],
        verdict: str | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        """
        Create evaluation span (EVALUATOR kind - LLM judge via DeepEval).

        If called within rag_query_span context, becomes a child span automatically.

        Args:
            trace_id: Parent trace ID (for tracking).
            evaluation_metrics: Dictionary of metric scores.
            verdict: Optional verdict string (PASS/NEEDS_REVIEW/ERROR).
            reasoning: Optional full reasoning from DeepEval with keys:
                - metric_name.reason: L1 overall explanation
                - metric_name.verdicts: L2 per-chunk judgments
                - metric_name.claims/truths: L3 detailed breakdown

        """
        # Store for export
        self._evaluations.append(
            {
                "trace_id": trace_id,
                "metrics": evaluation_metrics,
            }
        )

        if not self._tracer:
            return

        with self._tracer.start_as_current_span(
            name="evaluator", openinference_span_kind=EVALUATOR
        ) as span:
            # Add verdict if provided
            if verdict:
                span.set_attribute("evaluation.verdict", verdict)

            # Build output summary for Phoenix UI
            metrics_summary = json.dumps(
                {k: round(v, 4) for k, v in evaluation_metrics.items()}
            )
            span.set_attribute("output.value", metrics_summary)

            # Individual metric scores
            for metric_name, score in evaluation_metrics.items():
                span.set_attribute(f"evaluation.{metric_name}", score)

            # Add reasoning data if provided (DeepEval full reasoning)
            if reasoning:
                for metric_name, metric_reasoning in reasoning.items():
                    # Add L1: overall reason
                    if reason := metric_reasoning.get("reason"):
                        span.set_attribute(f"evaluation.{metric_name}.reason", reason)

                    # Add L2: verdicts (per-chunk judgments)
                    if verdicts := metric_reasoning.get("verdicts"):
                        # Store as JSON for Phoenix UI display
                        span.set_attribute(
                            f"evaluation.{metric_name}.verdicts",
                            json.dumps(verdicts),
                        )
                        # Also store count for quick filtering
                        span.set_attribute(
                            f"evaluation.{metric_name}.verdicts_count",
                            len(verdicts),
                        )

                    # Add L3: claims/truths (for Faithfulness)
                    for attr in ("claims", "truths", "statements"):
                        if items := metric_reasoning.get(attr):
                            span.set_attribute(
                                f"evaluation.{metric_name}.{attr}",
                                json.dumps(items),
                            )

    @beartype
    def export_traces(self) -> dict[str, Any]:
        """
        Export traces to Phoenix.

        Ends any active root span (from non-context-manager usage) and flushes.

        Returns:
            Dictionary with export results.

        """
        # End active root span if it's still open (from start_rag_query_span)
        if self._active_root_span:
            self._active_root_span.end()
            self._active_root_span = None

        # Force flush
        if self._tracer_provider:
            self._tracer_provider.force_flush()

        result: dict[str, Any] = {
            "trace_count": len(self._evaluations),
            "mode": "phoenix" if self.is_connected() else "parquet",
            "path": None,
        }

        # Export evaluations to Parquet if needed
        if self._evaluations and not self.is_connected():
            result["path"] = self._export_to_parquet()

        count = len(self._evaluations)
        self._evaluations = []
        result["trace_count"] = count

        return result

    def _export_to_parquet(self) -> str:
        """Export evaluations to Parquet file."""
        import polars as pl

        export_dir = self._export_path or Path("/tmp/phoenix_traces")
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        parquet_path = export_dir / f"evaluations_{timestamp}.parquet"

        rows = []
        for eval_data in self._evaluations:
            row = {"trace_id": eval_data["trace_id"]}
            row.update(eval_data["metrics"])
            rows.append(row)

        df = pl.DataFrame(rows)
        df.write_parquet(parquet_path)

        return str(parquet_path)

    @beartype
    def upload_parquet_to_s3(
        self,
        parquet_path: Path,
        bucket: str,
        key_prefix: str = "phoenix-traces",
    ) -> bool:
        """Upload buffered Parquet traces to S3."""
        try:
            import boto3

            s3_client = boto3.client("s3")
            key = f"{key_prefix}/{parquet_path.name}"
            s3_client.upload_file(str(parquet_path), bucket, key)
            return True
        except Exception:
            import sys

            print(f"[WARN] S3 upload failed for {parquet_path}", file=sys.stderr)
            return False
