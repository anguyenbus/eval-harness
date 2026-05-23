"""
CLI runner for demo showcase - RAG chunking comparison.

Usage:
    uv run demo-showcase

Demonstrates paired statistical testing for comparing RAG chunking strategies:
- Baseline: 512-token chunks, 0 overlap
- Candidate: 512-token chunks, 150 overlap
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from eval_harness.replay.comparison import paired_comparison
from eval_harness.stubs.rag.chunking import ConfigurableChunker
from eval_harness.stubs.span_generator.loader import sample_questions

# Disable DeepEval telemetry
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
load_dotenv()

console = Console()


def _parse_approach_name(name: str) -> tuple[int, int]:
    """
    Parse approach name to extract chunk size and overlap.

    Expected format: stub-chunks-{size}-overlap-{overlap}
    Example: stub-chunks-512-overlap-150 -> (512, 150)

    Args:
        name: Approach name string.

    Returns:
        Tuple of (chunk_size, chunk_overlap).

    Raises:
        click.ClickException: If name format is invalid.
    """
    parts = name.split("-")
    try:
        chunks_idx = parts.index("chunks")
        overlap_idx = parts.index("overlap")
        chunk_size = int(parts[chunks_idx + 1])
        chunk_overlap = int(parts[overlap_idx + 1])
        return chunk_size, chunk_overlap
    except (ValueError, IndexError):
        raise click.ClickException(
            f"Invalid approach name '{name}'. Expected format: stub-chunks-SIZE-overlap-OVERLAP\n"
            f"Example: stub-chunks-512-overlap-0"
        )


@click.command()
@click.option(
    "--baseline",
    type=str,
    default="stub-chunks-512-overlap-0",
    help="Baseline approach name (default: stub-chunks-512-overlap-0)",
)
@click.option(
    "--candidate",
    type=str,
    default="stub-chunks-512-overlap-150",
    help="Candidate approach name (default: stub-chunks-512-overlap-150)",
)
@click.option(
    "--questions",
    type=int,
    default=50,
    help="Number of questions to evaluate (default: 50)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
@click.option(
    "--naive-mode",
    is_flag=True,
    default=False,
    help="Run each approach on different random samples (demonstrates wrong way)",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Export results to JSON file",
)
def main(
    baseline: str,
    candidate: str,
    questions: int,
    seed: int,
    naive_mode: bool,
    output: str | None,
) -> None:
    """
    Run demo showcase comparing RAG chunking strategies.

    Example:
        demo-showcase --baseline stub-chunks-512-overlap-0 --candidate stub-chunks-512-overlap-150

    Approach name format: stub-chunks-{size}-overlap-{overlap}
    """
    # Parse approach names to extract chunk size and overlap
    baseline_chunk_size, baseline_overlap = _parse_approach_name(baseline)
    candidate_chunk_size, candidate_overlap = _parse_approach_name(candidate)

    console.print(f"[bold blue]RAG Chunking Comparison Demo[/bold blue]")
    console.print("=" * 50)

    # Sample questions
    console.print(f"\nLoading {questions} questions from legal-rag-bench...")
    if naive_mode:
        baseline_questions = sample_questions(limit=questions, seed=seed)
        candidate_questions = sample_questions(limit=questions, seed=seed + 1)
        console.print(f"[dim]Mode: NAIVE (different questions for each approach)[/dim]")
    else:
        all_questions = sample_questions(limit=questions, seed=seed)
        baseline_questions = all_questions
        candidate_questions = all_questions
        console.print(f"[dim]Mode: PAIRED (same questions for both approaches)[/dim]")

    # Create chunkers
    baseline_chunker = ConfigurableChunker(
        chunk_size=baseline_chunk_size, chunk_overlap=baseline_overlap
    )
    candidate_chunker = ConfigurableChunker(
        chunk_size=candidate_chunk_size, chunk_overlap=candidate_overlap
    )

    # TODO: Run actual RAG pipeline with both chunkers
    # For now, generate mock results for demonstration
    console.print(f"\nRunning baseline: {baseline}...")
    baseline_scores = [0.7 + (i * 0.01) % 0.3 for i in range(len(baseline_questions))]
    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    console.print(f"  Baseline avg faithfulness: {baseline_avg:.4f}")

    console.print(f"\nRunning candidate: {candidate}...")
    candidate_scores = [0.75 + (i * 0.01) % 0.25 for i in range(len(candidate_questions))]
    candidate_avg = sum(candidate_scores) / len(candidate_scores)
    console.print(f"  Candidate avg faithfulness: {candidate_avg:.4f}")

    # Run statistical comparison
    result = paired_comparison(
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
    )

    # Print results
    console.print("\n" + "=" * 50)
    console.print("[bold]Results[/bold]")
    console.print("=" * 50)
    console.print(f"Baseline:   {baseline}")
    console.print(f"Candidate:  {candidate}")
    console.print(f"Questions:  {len(baseline_questions)}")
    console.print(f"Seed:       {seed}")
    console.print(f"\nBaseline avg faithfulness:   {baseline_avg:.4f}")
    console.print(f"Candidate avg faithfulness:  {candidate_avg:.4f}")
    console.print(f"\nTest Statistic:  {result.statistic:.4f}")
    console.print(f"P-value:         {result.p_value:.6f}")
    console.print(f"Effect Size:     {result.effect_size:.4f}")
    console.print(f"Winner:          {result.winner.upper()}")
    console.print(f"Significant:     {'YES' if result.p_value < 0.05 else 'NO'}")

    # Convert ComparisonResult to dict for easier handling
    result_dict = {
        "statistic": result.statistic,
        "p_value": result.p_value,
        "effect_size": result.effect_size,
        "cliffs_delta": result.effect_size,
        "pass_fail": result.pass_fail,
        "winner": result.winner,
        "is_significant": result.p_value < 0.05,
        "mean_baseline": sum(baseline_scores) / len(baseline_scores),
        "mean_candidate": sum(candidate_scores) / len(candidate_scores),
        "mean_delta": (sum(candidate_scores) - sum(baseline_scores)) / len(baseline_scores),
        "effect_size_label": _interpret_effect_size(result.effect_size),
    }

    # Display per-question results
    _display_per_question_results(baseline_questions, baseline_scores, candidate_scores)

    # Display statistical summary
    _display_statistical_summary(result_dict)

    # Export results if requested
    if output:
        _export_results(
            output=output,
            baseline_questions=baseline_questions,
            candidate_questions=candidate_questions,
            baseline_scores=baseline_scores,
            candidate_scores=candidate_scores,
            result=result_dict,
            config={
                "baseline_chunk_size": baseline_chunk_size,
                "baseline_overlap": baseline_overlap,
                "candidate_chunk_size": candidate_chunk_size,
                "candidate_overlap": candidate_overlap,
                "seed": seed,
                "naive_mode": naive_mode,
            },
        )
        console.print(f"\n[green]Results exported to {output}[/green]")


def _display_per_question_results(
    baseline_questions: list[Any],
    baseline_scores: list[float],
    candidate_scores: list[float],
) -> None:
    """Display per-question comparison table."""
    table = Table(title="Per-Question Results")
    table.add_column("Q#", style="dim", width=4)
    table.add_column("Baseline", justify="right", width=8)
    table.add_column("Candidate", justify="right", width=8)
    table.add_column("Delta", justify="right", width=8)

    for i, (base, cand) in enumerate(zip(baseline_scores, candidate_scores)):
        delta = cand - base
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "dim"
        table.add_row(str(i + 1), f"{base:.3f}", f"{cand:.3f}", f"[{delta_style}]{delta:+.3f}[/{delta_style}]")

    console.print(table)


def _interpret_effect_size(effect_size: float) -> str:
    """Interpret Cliff's Delta effect size."""
    abs_es = abs(effect_size)
    if abs_es < 0.147:
        return "negligible"
    if abs_es < 0.33:
        return "small"
    if abs_es < 0.474:
        return "medium"
    return "large"


def _display_statistical_summary(result: dict[str, Any]) -> None:
    """Display statistical summary."""
    console.print("\n[bold]Statistical Analysis:[/bold]")
    console.print(f"  Baseline mean score:  {result['mean_baseline']:.3f}")
    console.print(f"  Candidate mean score: {result['mean_candidate']:.3f}")
    console.print(f"  Average improvement:  {result['mean_delta']:+.3f}")

    console.print(f"\n  [dim]Wilcoxon signed-rank test (paired comparison):[/dim]")
    console.print(f"    P-value: {result['p_value']:.6f} {'(significant)' if result['is_significant'] else '(not significant)'}")
    console.print(f"    Cliff's Delta: {result['cliffs_delta']:.3f} ({result['effect_size_label']} effect size)")

    console.print(f"\n  [bold]Conclusion:[/bold] {result['winner'].upper()} wins")
    if result['is_significant']:
        console.print(f"    The difference is statistically significant (p < 0.05)")
    else:
        console.print(f"    Cannot conclude the difference is real (could be random variation)")


def _export_results(
    output: str,
    baseline_questions: list[Any],
    candidate_questions: list[Any],
    baseline_scores: list[float],
    candidate_scores: list[float],
    result: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Export results to JSON file."""
    output_path = Path(output)

    per_question = []
    for i, (q, base, cand) in enumerate(zip(baseline_questions, baseline_scores, candidate_scores)):
        per_question.append(
            {
                "question_id": str(i),
                "question": q.question if hasattr(q, "question") else str(q),
                "baseline_score": base,
                "candidate_score": cand,
                "delta": cand - base,
            }
        )

    export_data = {
        "config": config,
        "per_question": per_question,
        "statistical_summary": result,
    }

    output_path.write_text(json.dumps(export_data, indent=2))


if __name__ == "__main__":
    main()
