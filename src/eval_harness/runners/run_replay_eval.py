"""
CLI runner for replay evaluation against generated spans.

Usage:
    uv run eval-replay --candidate=stub-local --baseline=stub-local

Runs replay evaluation against synthetic spans to compare candidate
versus baseline adapters using real RAG execution and DeepEval metrics.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

import click
from dotenv import load_dotenv

from eval_harness.adapters.rag_adapter import RagAdapter
from eval_harness.config import load_config
from eval_harness.replay.comparison import paired_comparison
from eval_harness.replay.phoenix_client import PhoenixClient

# Disable DeepEval telemetry
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
load_dotenv()


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


def _run_adapter_on_questions(
    questions: list[str],
    adapter: RagAdapter,
    corpus_dir: Path,
) -> list[float]:
    """
    Run RAG adapter on questions and return scores.

    Args:
        questions: List of question strings.
        adapter: RAG adapter to run.
        corpus_dir: Path to document corpus.

    Returns:
        List of faithfulness scores.
    """
    from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

    # Initialize evaluator
    evaluator = DeepEvalEvaluator(
        llm_provider="openai",
        judge_model="gpt-4o",
        temperature=0.0,
        max_concurrent=4,
    )

    scores = []
    for question in questions:
        try:
            # Query RAG system
            output = adapter.query(question, corpus_dir)
            generated_answer = output.get("answer", {}).get("text", "")

            # Compute faithfulness score
            metric_result = evaluator.compute_metrics_with_reasoning(
                output, ""  # No gold answer needed for faithfulness
            )
            score = metric_result["scores"].get("faithfulness", 0.0)
            scores.append(score)
        except Exception as e:
            click.echo(f"WARNING: Error processing question: {e}", err=True)
            scores.append(0.0)

    return scores


def _get_rag_adapter(
    rag_name: str,
    top_k: int = 5,
    embedder: Any = None,
) -> RagAdapter:
    """
    Get RAG adapter by name.

    Args:
        rag_name: Name of RAG system.
        top_k: Number of chunks to retrieve.
        embedder: Optional shared embedder instance.

    Returns:
        RagAdapter instance.
    """
    if rag_name == "stub-local":
        from eval_harness.stubs.rag.chromadb_query import query as chromadb_query

        def chromadb_wrapper(
            question: str, corpus_dir: Path, embedder: Any = None
        ) -> dict[str, Any]:
            return chromadb_query(
                question=question,
                corpus_dir=corpus_dir,
                top_k=top_k,
                force_reingest=False,
                embedder=embedder,
            )

        return RagAdapter(query_callable=chromadb_wrapper, embedder=embedder)
    else:
        click.echo(f"WARNING: RAG '{rag_name}' not found, using stub-local", err=True)
        from eval_harness.stubs.rag.chromadb_query import query as chromadb_query

        def chromadb_wrapper(
            question: str, corpus_dir: Path, embedder: Any = None
        ) -> dict[str, Any]:
            return chromadb_query(
                question=question,
                corpus_dir=corpus_dir,
                top_k=top_k,
                force_reingest=False,
                embedder=embedder,
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
def main(
    candidate: str,
    baseline: str,
    project: str,
    output: Path | None,
    phoenix_endpoint: str | None,
    limit: int,
    top_k: int,
) -> None:
    """
    Run replay evaluation against generated spans.

    This CLI queries synthetic spans from Phoenix, runs candidate and
    baseline adapters, and performs paired statistical comparison.

    Example:
        eval-replay --candidate=stub-local --baseline=stub-local --limit 10

    """
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

        # Extract questions from spans
        questions = []
        for span in root_spans:
            question = _extract_question_from_span(span)
            if question:
                questions.append(question)

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
        click.echo(f"Running candidate adapter: {candidate}")
        candidate_adapter = _get_rag_adapter(candidate, top_k, embedder)
        candidate_scores = _run_adapter_on_questions(questions, candidate_adapter, corpus_dir)
        click.echo(f"Candidate avg faithfulness: {sum(candidate_scores)/len(candidate_scores):.4f}")

        # Get baseline adapter (if different from candidate)
        if baseline == candidate:
            baseline_scores = candidate_scores
            click.echo(f"Baseline same as candidate, using same scores")
        else:
            click.echo(f"Running baseline adapter: {baseline}")
            baseline_adapter = _get_rag_adapter(baseline, top_k, embedder)
            baseline_scores = _run_adapter_on_questions(questions, baseline_adapter, corpus_dir)
            click.echo(f"Baseline avg faithfulness: {sum(baseline_scores)/len(baseline_scores):.4f}")

        # Perform paired comparison
        result = paired_comparison(candidate_scores, baseline_scores)

        # Print results
        click.echo("\n" + "=" * 50)
        click.echo("Replay Evaluation Results")
        click.echo("=" * 50)
        click.echo(f"Candidate: {candidate}")
        click.echo(f"Baseline: {baseline}")
        click.echo(f"Questions evaluated: {len(questions)}")
        click.echo(f"\nCandidate avg faithfulness: {sum(candidate_scores)/len(candidate_scores):.4f}")
        click.echo(f"Baseline avg faithfulness: {sum(baseline_scores)/len(baseline_scores):.4f}")
        click.echo(f"\nTest Statistic: {result.statistic:.4f}")
        click.echo(f"P-value: {result.p_value:.4f}")
        click.echo(f"Effect Size (Cliff's Delta): {result.effect_size:.4f}")
        click.echo(f"Winner: {result.winner.upper()}")
        click.echo(f"Pass/Fail: {'PASS' if result.pass_fail else 'FAIL'}")

        # Export results if requested
        if output:
            results_dict = {
                "candidate": candidate,
                "baseline": baseline,
                "project": project,
                "num_questions": len(questions),
                "num_spans": len(root_spans),
                "candidate_avg_faithfulness": sum(candidate_scores) / len(candidate_scores),
                "baseline_avg_faithfulness": sum(baseline_scores) / len(baseline_scores),
                "candidate_scores": candidate_scores,
                "baseline_scores": baseline_scores,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "winner": result.winner,
                "pass_fail": result.pass_fail,
            }

            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(results_dict, f, indent=2)

            click.echo(f"\nResults exported to: {output}")

        # Exit with appropriate code
        sys.exit(0 if result.pass_fail else 1)

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


if __name__ == "__main__":
    main()
