"""
CLI runner for replay evaluation against generated spans.

Usage:
    uv run eval-replay --candidate=stub-local --baseline=stub-local

Runs replay evaluation against synthetic spans to compare candidate
versus baseline adapters using real RAG execution and DeepEval metrics.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from eval_harness.adapters.rag_adapter import RagAdapter
from eval_harness.config import load_config
from eval_harness.replay.comparison import paired_comparison
from eval_harness.replay.phoenix_client import PhoenixClient

# Disable DeepEval telemetry
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
load_dotenv()

# Global tracer (initialized once, reused for all questions)
_tracer_provider: Any = None
_tracer: Any = None


def _get_tracer(phoenix_endpoint: str, project_name: str) -> Any:
    """
    Get or initialize global tracer for replay spans.

    Args:
        phoenix_endpoint: Phoenix server endpoint.
        project_name: Phoenix project name.

    Returns:
        Tracer instance or None if initialization fails.

    """
    global _tracer_provider, _tracer

    if _tracer is not None:
        return _tracer

    try:

        from eval_harness.stubs.span_generator.tracer import setup_tracer

        _tracer_provider, _tracer = setup_tracer(
            phoenix_endpoint=phoenix_endpoint,
            project_name=project_name,
            batch=True,
            auto_instrument=False,
        )
        return _tracer
    except Exception:
        return None


def _extract_question_from_span(span: dict[str, Any]) -> str | None:
    """
    Extract question text from a span.

    Args:
        span: Span dictionary.

    Returns:
        Question text or None if not found.

    """
    # Try common attribute names
    for key in ["attributes.input.value", "input.value", "input"]:
        if key in span and span[key]:
            return str(span[key])

    # Check nested attributes
    attrs = span.get("attributes", {})
    if isinstance(attrs, dict):
        if "input.value" in attrs:
            return str(attrs["input.value"])
        if "input" in attrs:
            input_val = attrs["input"]
            if isinstance(input_val, dict) and "value" in input_val:
                return str(input_val["value"])

    return None


def _extract_ground_truth_from_span(span: dict[str, Any]) -> str | None:
    """
    Extract expected answer from span metadata.

    Args:
        span: Span dictionary.

    Returns:
        Expected answer text or None if not found.

    """
    attrs = span.get("attributes", {})
    if not isinstance(attrs, dict):
        return None

    # Check metadata JSON
    metadata = attrs.get("metadata")
    if metadata:
        try:
            if isinstance(metadata, str):
                meta_dict = json.loads(metadata)
            else:
                meta_dict = metadata

            return meta_dict.get("expected_answer")
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _extract_baseline_score_from_span(span: dict[str, Any]) -> float | None:
    """
    Extract baseline faithfulness score from span attributes.

    Args:
        span: Span dictionary (from PhoenixClient with flattened keys).

    Returns:
        Baseline score or None if not found.

    """
    # PhoenixClient returns flattened keys like "attributes.rag_faithfulness"
    # Check these first (new format)
    faithfulness_keys = [
        "attributes.rag_faithfulness",
        "attributes.faithfulness",
        "attributes.eval_harness.faithfulness",
    ]
    for key in faithfulness_keys:
        if key in span:
            val = span[key]
            if val is not None and not (isinstance(val, float) and str(val) == 'nan'):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass

    # Check nested attributes dict (some formats)
    attrs = span.get("attributes", {})
    if isinstance(attrs, dict):
        # Direct keys in attributes dict
        for key in ["rag_faithfulness", "faithfulness"]:
            if key in attrs:
                val = attrs[key]
                is_valid = val is not None and not (
                    isinstance(val, float) and str(val) == 'nan'
                )
                if is_valid:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass

        # Check nested eval_harness dict (old format)
        eval_harness = attrs.get("eval_harness")
        if isinstance(eval_harness, dict):
            faithfulness = eval_harness.get("faithfulness")
            if faithfulness is not None:
                try:
                    return float(faithfulness)
                except (ValueError, TypeError):
                    pass

    return None


def _extract_all_baseline_scores(span: dict[str, Any]) -> dict[str, float]:
    """
    Extract all baseline scores from span attributes.

    Args:
        span: Span dictionary (from PhoenixClient with flattened keys).

    Returns:
        Dict of metric names to scores.

    """
    metrics = {}
    metric_map = {
        "attributes.rag_faithfulness": "faithfulness",
        "attributes.rag_context_precision": "context_precision",
        "attributes.rag_context_recall": "context_recall",
        "attributes.rag_answer_relevancy": "answer_relevancy",
        "attributes.rag_latency_total_ms": "latency_total_ms",
    }

    for key, metric_name in metric_map.items():
        if key in span:
            val = span[key]
            if val is not None and not (isinstance(val, float) and str(val) == 'nan'):
                try:
                    metrics[metric_name] = float(val)
                except (ValueError, TypeError):
                    pass

    return metrics


def _run_adapter_on_questions(
    questions: list[str],
    adapter: RagAdapter,
    corpus_dir: Path,
    expected_answers: list[str] | None = None,
    tracer: Any = None,
    candidate_name: str = "candidate",
) -> list[dict[str, float]]:
    """
    Run RAG adapter on questions and return all scores.

    Args:
        questions: List of question strings.
        adapter: RAG adapter to run.
        corpus_dir: Path to document corpus.
        expected_answers: Optional list of expected answers for evaluation.
        tracer: Optional tracer for creating parent spans.
        candidate_name: Name for candidate (used in span naming).

    Returns:
        List of dicts with metric scores per question.

    """
    from openinference.semconv.trace import (
        OpenInferenceSpanKindValues,
        SpanAttributes,
    )

    from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

    # Initialize evaluator
    evaluator = DeepEvalEvaluator(
        llm_provider="openai",
        judge_model="gpt-4o",
        temperature=0.0,
        max_concurrent=4,
    )

    all_scores = []
    for i, question in enumerate(questions):
        span_context = None
        try:
            # Create parent span for replay query
            if tracer is not None:
                span_context = tracer.start_as_current_span(
                    name=f"replay_{candidate_name}",
                    openinference_span_kind=OpenInferenceSpanKindValues.CHAIN,
                )
                span = span_context.__enter__()
                span.set_attribute(SpanAttributes.INPUT_VALUE, question)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, "")

            # Query RAG system
            output = adapter.query(question, corpus_dir)

            # Compute all metrics
            expected = (
                expected_answers[i]
                if expected_answers and i < len(expected_answers)
                else ""
            )
            metric_result = evaluator.compute_metrics_with_reasoning(
                output, expected
            )
            scores = metric_result["scores"]
            # Add latency
            timings = output.get("timings_ms", {})
            scores["rag_latency_total_ms"] = timings.get("total", 0.0)
            scores["rag_latency_retrieval_ms"] = timings.get("retrieval", 0.0)
            scores["rag_latency_generation_ms"] = timings.get("generation", 0.0)

            # Update span with output and metrics
            if span_context is not None:
                answer_text = output.get("answer", {}).get("text", "")
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer_text)

                # Store evaluation metrics as span attributes
                # (same format as synthetic spans)
                span.set_attribute("rag_faithfulness", scores.get("faithfulness", 0.0))
                span.set_attribute(
                    "rag_context_precision", scores.get("context_precision", 0.0)
                )
                span.set_attribute(
                    "rag_context_recall", scores.get("context_recall", 0.0)
                )
                span.set_attribute(
                    "rag_answer_relevancy", scores.get("answer_relevancy", 0.0)
                )
                span.set_attribute(
                    "rag_latency_total_ms", scores.get("rag_latency_total_ms", 0.0)
                )
                span.set_attribute(
                    "rag_latency_retrieval_ms",
                    scores.get("rag_latency_retrieval_ms", 0.0),
                )
                span.set_attribute(
                    "rag_latency_generation_ms",
                    scores.get("rag_latency_generation_ms", 0.0),
                )

            all_scores.append(scores)

            if span_context is not None:
                span_context.__exit__(None, None, None)
        except Exception as e:
            if span_context is not None:
                span_context.__exit__(None, None, None)
            click.echo(f"WARNING: Error processing question: {e}", err=True)
            all_scores.append({"rag_faithfulness": 0.0, "rag_context_precision": 0.0,
                              "rag_context_recall": 0.0, "rag_answer_relevancy": 0.0,
                              "rag_latency_total_ms": 0.0})

    return all_scores


def _get_rag_adapter(
    rag_name: str,
    top_k: int = 5,
    embedder: Any = None,
) -> RagAdapter:
    """
    Get RAG adapter by name.

    Parses candidate name to extract configuration. Supported formats:
    - stub-local: Default stub with config chunking
    - stub-chunks-{size}-overlap-{overlap}: Configurable chunking
    - Future: stub-reranker-{model}, stub-llm-{model}, etc.

    Args:
        rag_name: Name of RAG system (encodes configuration).
        top_k: Number of chunks to retrieve.
        embedder: Optional shared embedder instance.

    Returns:
        RagAdapter instance.

    """
    from eval_harness.stubs.rag.chromadb_query import query as chromadb_query

    # Parse candidate name for chunking config
    chunk_size = None
    chunk_overlap = None

    if "chunks-" in rag_name and "-overlap-" in rag_name:
        # Parse: stub-chunks-512-overlap-150
        parts = rag_name.split("-")
        try:
            chunks_idx = parts.index("chunks")
            overlap_idx = parts.index("overlap")
            chunk_size = int(parts[chunks_idx + 1])
            chunk_overlap = int(parts[overlap_idx + 1])
            click.echo(
                f"  Configured: chunk_size={chunk_size}, overlap={chunk_overlap}"
            )
        except (ValueError, IndexError):
            click.echo(
                f"  WARNING: Invalid chunking format in '{rag_name}', using defaults",
                err=True,
            )

    def chromadb_wrapper(
        question: str, corpus_dir: Path, embedder: Any = None
    ) -> dict[str, Any]:
        return chromadb_query(
            question=question,
            corpus_dir=corpus_dir,
            top_k=top_k,
            force_reingest=False,
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return RagAdapter(query_callable=chromadb_wrapper, embedder=embedder)


@click.command()
@click.option(
    "--candidate",
    type=str,
    default="stub-local",
    help="Candidate adapter name (default: stub-local)",
)
@click.option(
    "--baseline",
    type=str,
    default="stub-local",
    help="Baseline adapter name (default: stub-local)",
)
@click.option(
    "--project",
    type=str,
    default="case-assistant-synthetic",
    help="Phoenix project name",
)
@click.option(
    "--output",
    type=Path,
    default=None,
    help="Output path for results JSON",
)
@click.option(
    "--phoenix-endpoint",
    type=str,
    default=None,
    help="Phoenix server endpoint",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of spans to evaluate (default: 10)",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of chunks to retrieve (default: 5)",
)
@click.option(
    "--production-baseline",
    is_flag=True,
    default=False,
    help=(
        "Use production baseline scores from span attributes "
        "instead of re-running baseline adapter."
    ),
)
def main(
    candidate: str,
    baseline: str,
    project: str,
    output: Path | None,
    phoenix_endpoint: str | None,
    limit: int,
    top_k: int,
    production_baseline: bool,
) -> None:
    """
    Run replay evaluation against generated spans.

    This CLI queries synthetic spans from Phoenix, runs candidate and
    baseline adapters, and performs paired statistical comparison.

    Example:
        eval-replay --candidate=stub-local --baseline=stub-local --limit 10

    """
    global _tracer_provider

    try:
        # Initialize Phoenix client
        endpoint = phoenix_endpoint or "http://localhost:6006"
        client = PhoenixClient(endpoint=endpoint, project_name=project)

        if not client.is_connected():
            click.echo(
                f"ERROR: Could not connect to Phoenix at {endpoint}",
                err=True,
            )
            click.echo("Hint: Start Phoenix with: python -m phoenix.server.main serve")
            sys.exit(1)

        # Query root spans
        click.echo(f"Querying spans from project: {project}")
        root_spans = client.query_root_spans(limit=limit)
        click.echo(f"Found {len(root_spans)} synthetic spans")

        if len(root_spans) == 0:
            click.echo("ERROR: No spans found in Phoenix.")
            click.echo("Hint: Run 'uv run generate-spans --limit N' first")
            sys.exit(1)

        # Extract questions and expected answers from spans
        questions = []
        expected_answers = []
        for span in root_spans:
            question = _extract_question_from_span(span)
            expected = _extract_ground_truth_from_span(span)
            if question:
                questions.append(question)
                expected_answers.append(expected or "")

        if len(questions) == 0:
            click.echo("ERROR: Could not extract questions from spans")
            sys.exit(1)

        click.echo(f"Extracted {len(questions)} questions from spans")

        # Load config for corpus directory
        config = load_config(Path("eval_config.yaml"))
        dataset_config = config["datasets"].get("legal_rag_bench", {})
        corpus_dir = Path(dataset_config.get("path", "data/rag/legal_rag_bench"))

        # Initialize shared embedder
        try:
            from eval_harness.adapters.embeddings import get_embedder

            embedder = get_embedder(
                provider="huggingface",
                model="sentence-transformers/all-MiniLM-L6-v2",
            )
        except Exception as e:
            click.echo(f"WARNING: Could not initialize embedder: {e}", err=True)
            embedder = None

        # Get candidate adapter
        click.echo(f"Running candidate: {candidate}")
        candidate_adapter = _get_rag_adapter(candidate, top_k, embedder)
        candidate_scores = _run_adapter_on_questions(
            questions,
            candidate_adapter,
            corpus_dir,
            expected_answers,
            tracer=_get_tracer(endpoint, project),
            candidate_name=candidate,
        )

        # Calculate candidate averages
        # Note: DeepEval returns keys without "rag_" prefix
        # (faithfulness, context_precision, etc.)
        # We add latency_total_ms separately from timings
        candidate_avgs = {}
        metric_key_map = {
            "rag_faithfulness": "faithfulness",
            "rag_context_precision": "context_precision",
            "rag_context_recall": "context_recall",
            "rag_answer_relevancy": "answer_relevancy",
            # Keep this for custom added key
            "rag_latency_total_ms": "rag_latency_total_ms",
        }
        for rag_metric, score_key in metric_key_map.items():
            values = [s.get(score_key, 0.0) for s in candidate_scores]
            candidate_avgs[rag_metric] = sum(values) / len(values) if values else 0.0

        click.echo(
            f"Candidate avg faithfulness: {candidate_avgs['rag_faithfulness']:.4f}"
        )
        click.echo(
            f"  context_precision: {candidate_avgs['rag_context_precision']:.4f}"
        )
        click.echo(f"  context_recall: {candidate_avgs['rag_context_recall']:.4f}")
        click.echo(
            f"  answer_relevancy: {candidate_avgs['rag_answer_relevancy']:.4f}"
        )
        click.echo(f"  latency_total_ms: {candidate_avgs['rag_latency_total_ms']:.0f}")

        # Metric key mapping for consistent lookups
        # DeepEval returns: faithfulness, context_precision, context_recall,
        #                   answer_relevancy
        # We add: rag_latency_total_ms (from timings)
        # Spans have: faithfulness, context_precision, context_recall,
        #            answer_relevancy, latency_total_ms
        metric_key_map = {
            "rag_faithfulness": "faithfulness",
            "rag_context_precision": "context_precision",
            "rag_context_recall": "context_recall",
            "rag_answer_relevancy": "answer_relevancy",
            "rag_latency_total_ms": "rag_latency_total_ms",
        }

        # Get baseline scores (from production spans or re-run)
        if production_baseline:
            # Extract all baseline scores from span attributes (production results)
            baseline_all_scores = []
            baseline_faithfulness = []
            for span in root_spans:
                scores = _extract_all_baseline_scores(span)
                if scores:
                    baseline_all_scores.append(scores)
                    if "faithfulness" in scores:
                        baseline_faithfulness.append(scores["faithfulness"])

            if len(baseline_faithfulness) == 0:
                click.echo(
                    "ERROR: No baseline scores found in span attributes",
                    err=True,
                )
                click.echo(
                    "Hint: Spans must contain 'faithfulness' attribute "
                    "in span output"
                )
                sys.exit(1)

            # Calculate baseline averages
            # Note: Span scores have keys without "rag_" prefix
            #       (faithfulness, context_precision, latency_total_ms, etc.)
            baseline_avgs = {}
            for rag_metric, score_key in metric_key_map.items():
                key = score_key.replace("rag_", "")  # Strip rag_ for baseline lookups
                values = [s.get(key, 0.0) for s in baseline_all_scores]
                baseline_avgs[rag_metric] = sum(values) / len(values) if values else 0.0

            click.echo(
                f"Baseline (production) avg faithfulness: "
                f"{baseline_avgs['rag_faithfulness']:.4f}"
            )
            click.echo(
                f"  context_precision: {baseline_avgs['rag_context_precision']:.4f}"
            )
            click.echo(f"  context_recall: {baseline_avgs['rag_context_recall']:.4f}")
            click.echo(
                f"  answer_relevancy: {baseline_avgs['rag_answer_relevancy']:.4f}"
            )
            click.echo(
                f"  latency_total_ms: {baseline_avgs['rag_latency_total_ms']:.0f}"
            )

            # Use faithfulness for statistical comparison
            baseline_scores = baseline_faithfulness
            candidate_faithfulness = [
                s.get("faithfulness", 0.0) for s in candidate_scores
            ]
        else:
            # Re-run baseline adapter
            if baseline == candidate:
                baseline_scores = [s.get("faithfulness", 0.0) for s in candidate_scores]
                baseline_avgs = candidate_avgs
                baseline_all_scores = candidate_scores  # For comparison
                click.echo("Baseline same as candidate, using same scores")
            else:
                click.echo(f"Running baseline adapter: {baseline}")
                baseline_adapter = _get_rag_adapter(baseline, top_k, embedder)
                baseline_results = _run_adapter_on_questions(
                    questions,
                    baseline_adapter,
                    corpus_dir,
                    expected_answers,
                    tracer=_get_tracer(endpoint, project),
                    candidate_name=baseline,
                )
                baseline_scores = [
                    s.get("faithfulness", 0.0) for s in baseline_results
                ]
                baseline_avgs = {}
                for rag_metric, score_key in metric_key_map.items():
                    values = [s.get(score_key, 0.0) for s in baseline_results]
                    baseline_avgs[rag_metric] = (
                        sum(values) / len(values) if values else 0.0
                    )
                baseline_all_scores = baseline_results  # For comparison
                click.echo(
                    f"Baseline avg faithfulness: "
                    f"{sum(baseline_scores)/len(baseline_scores):.4f}"
                )
            candidate_faithfulness = [
                s.get("faithfulness", 0.0) for s in candidate_scores
            ]

        # Perform paired comparison on all metrics
        # Note: candidate_scores and baseline_all_scores have different key formats
        # candidate_scores: faithfulness, context_precision, rag_latency_total_ms
        # baseline_all_scores (from spans): faithfulness, latency_total_ms
        candidate_context_precision = [
            s.get("context_precision", 0.0) for s in candidate_scores
        ]
        baseline_context_precision = [
            s.get("context_precision", 0.0) for s in baseline_all_scores
        ]
        candidate_context_recall = [
            s.get("context_recall", 0.0) for s in candidate_scores
        ]
        baseline_context_recall = [
            s.get("context_recall", 0.0) for s in baseline_all_scores
        ]
        candidate_answer_relevancy = [
            s.get("answer_relevancy", 0.0) for s in candidate_scores
        ]
        baseline_answer_relevancy = [
            s.get("answer_relevancy", 0.0) for s in baseline_all_scores
        ]
        candidate_latency = [
            s.get("rag_latency_total_ms", 0.0) for s in candidate_scores
        ]
        baseline_latency = [
            s.get("latency_total_ms", 0.0) for s in baseline_all_scores
        ]

        metrics_to_compare = [
            ("faithfulness", candidate_faithfulness, baseline_scores),
            (
                "context_precision",
                candidate_context_precision,
                baseline_context_precision,
            ),
            ("context_recall", candidate_context_recall, baseline_context_recall),
            (
                "answer_relevancy",
                candidate_answer_relevancy,
                baseline_answer_relevancy,
            ),
            ("latency_total_ms", candidate_latency, baseline_latency),
        ]

        results = {}
        for metric_name, cand_vals, base_vals in metrics_to_compare:
            try:
                results[metric_name] = paired_comparison(cand_vals, base_vals)
            except Exception as e:
                click.echo(f"WARNING: Could not compare {metric_name}: {e}", err=True)
                results[metric_name] = None

        # Print results
        click.echo("\n" + "=" * 50)
        click.echo("Replay Evaluation Results")
        click.echo("=" * 50)
        click.echo(f"Candidate: {candidate}")
        click.echo(f"Baseline: {baseline}")
        click.echo(f"Questions evaluated: {len(questions)}")

        header = (
            f"\n{'Metric':<20} {'Candidate':>10} {'Baseline':>10} "
            f"{'P-value':>10} {'Effect':>10} {'Winner':>10}"
        )
        click.echo(header)
        click.echo("-" * 70)

        for metric_name, cand_vals, base_vals in metrics_to_compare:
            cand_avg = sum(cand_vals) / len(cand_vals) if cand_vals else 0.0
            base_avg = sum(base_vals) / len(base_vals) if base_vals else 0.0
            result = results.get(metric_name)

            if result:
                sig = "YES" if result.p_value < 0.05 else "NO"
                click.echo(
                    f"{metric_name:<20} {cand_avg:>10.4f} {base_avg:>10.4f} "
                    f"{result.p_value:>10.4f} {result.effect_size:>10.4f} "
                    f"{result.winner.upper():>10} ({sig})"
                )
            else:
                na_line = (
                    f"{metric_name:<20} {cand_avg:>10.4f} {base_avg:>10.4f} "
                    f"{'N/A':>10} {'N/A':>10} {'N/A':>10}"
                )
                click.echo(na_line)

        # Use faithfulness as primary metric for pass/fail
        primary_result = results.get("faithfulness")
        if primary_result:
            click.echo(
                f"\nPrimary Metric (faithfulness): {primary_result.winner.upper()}"
            )
            click.echo(
                f"Pass/Fail: {'PASS' if primary_result.pass_fail else 'FAIL'}"
            )

        # Export results if requested
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)

            # Build comparison results
            comparison_results = {}
            for metric_name, result in results.items():
                if result:
                    comparison_results[metric_name] = {
                        "p_value": result.p_value,
                        "effect_size": result.effect_size,
                        "winner": result.winner,
                        "pass_fail": result.pass_fail,
                    }

            # Build per-question details
            per_question_details = []
            for i, question in enumerate(questions):
                detail = {
                    "question": question,
                    "expected_answer": expected_answers[i] if expected_answers else "",
                    "baseline": {},
                    "candidate": {},
                }

                # Baseline scores
                if i < len(baseline_all_scores):
                    for metric_name, score_key in metric_key_map.items():
                        key = score_key.replace("rag_", "")
                        detail["baseline"][metric_name] = baseline_all_scores[i].get(
                            key, 0.0
                        )

                # Candidate scores
                if i < len(candidate_scores):
                    for rag_metric in metric_key_map.keys():
                        detail["candidate"][rag_metric] = candidate_scores[i].get(
                            metric_key_map[rag_metric], 0.0
                        )

                per_question_details.append(detail)

            # Summary JSON
            summary_path = output.parent / f"{output.stem}_summary.json"
            summary_dict = {
                "candidate": candidate,
                "baseline": baseline,
                "project": project,
                "num_questions": len(questions),
                "num_spans": len(root_spans),
                "averages": {
                    "candidate": {
                        "faithfulness": candidate_avgs['rag_faithfulness'],
                        "context_precision": candidate_avgs['rag_context_precision'],
                        "context_recall": candidate_avgs['rag_context_recall'],
                        "answer_relevancy": candidate_avgs['rag_answer_relevancy'],
                        "latency_total_ms": candidate_avgs['rag_latency_total_ms'],
                    },
                    "baseline": {
                        "faithfulness": baseline_avgs['rag_faithfulness'],
                        "context_precision": baseline_avgs['rag_context_precision'],
                        "context_recall": baseline_avgs['rag_context_recall'],
                        "answer_relevancy": baseline_avgs['rag_answer_relevancy'],
                        "latency_total_ms": baseline_avgs['rag_latency_total_ms'],
                    },
                },
                "statistical_tests": comparison_results,
            }

            with open(summary_path, "w") as f:
                json.dump(summary_dict, f, indent=2)

            click.echo(f"\nSummary JSON: {summary_path}")

            # Details JSON (per-question baseline + candidate scores)
            details_path = output.parent / f"{output.stem}_details.json"
            details_dict = {
                "candidate": candidate,
                "baseline": baseline,
                "project": project,
                "num_questions": len(questions),
                "questions": per_question_details,
            }

            with open(details_path, "w") as f:
                json.dump(details_dict, f, indent=2)

            click.echo(f"Details JSON: {details_path}")

            # CSV output (metrics summary)
            csv_path = output.parent / f"{output.stem}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow([
                    "metric",
                    "candidate_avg",
                    "baseline_avg",
                    "p_value",
                    "effect_size",
                    "winner",
                ])
                # Rows
                for metric_name, result in results.items():
                    if result:
                        cand_avg = sum(cand_vals) / len(cand_vals) if cand_vals else 0.0
                        base_avg = sum(base_vals) / len(base_vals) if base_vals else 0.0
                        writer.writerow([
                            metric_name,
                            f"{cand_avg:.4f}",
                            f"{base_avg:.4f}",
                            f"{result.p_value:.4f}",
                            f"{result.effect_size:.4f}",
                            result.winner,
                        ])

            click.echo(f"CSV summary: {csv_path}")

        # Exit with appropriate code (based on primary faithfulness metric)
        if primary_result:
            exit_code = 0 if primary_result.pass_fail else 1
        else:
            exit_code = 1

        # Force flush spans before exit
        if _tracer_provider is not None:
            _tracer_provider.force_flush()

        sys.exit(exit_code)

    except ConnectionError as e:
        click.echo(f"ERROR: {e}", err=True)
        click.echo(
            "Hint: Ensure Phoenix is running or use --phoenix-endpoint", err=True
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"ERROR: Unexpected error: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure spans are flushed
        if _tracer_provider is not None:
            _tracer_provider.force_flush()


if __name__ == "__main__":
    main()
