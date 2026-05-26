"""
CLI runner for synthetic span generation.

Usage:
    uv run generate-spans --limit=100 --seed=42
    uv run generate-spans --candidate-spec \
        configs/candidates/stub-chunking-512.yaml --limit=3

Generates synthetic OpenInference-compliant spans from the stub RAG pipeline
for testing evaluation harnesses before production traffic exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from eval_harness.config import load_config
from eval_harness.replay.candidate_config import CandidateConfig
from eval_harness.replay.http_client import HTTPClient
from eval_harness.stubs.span_generator.config import load_generator_config
from eval_harness.stubs.span_generator.loader import GeneratorQuestion, iter_questions
from eval_harness.stubs.span_generator.runner import run_generator


@click.command()
@click.option(
    "--candidate-spec",
    type=Path,
    default=None,
    help="Path to candidate spec YAML for HTTP-based generation",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of questions to process (default: from config)",
)
@click.option(
    "--phoenix-endpoint",
    type=str,
    default=None,
    help="Phoenix server endpoint (default: from config or PHOENIX_ENDPOINT env var)",
)
@click.option(
    "--project",
    type=str,
    default=None,
    help="Phoenix project name (default: case-assistant-synthetic)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility (default: from config or 42)",
)
@click.option(
    "--config",
    type=Path,
    default=Path("eval_config.yaml"),
    help="Path to eval_config.yaml",
)
@click.option(
    "--corpus-dir",
    type=Path,
    default=None,
    help="Path to document corpus directory (default: from config)",
)
def main(
    candidate_spec: Path | None,
    limit: int | None,
    phoenix_endpoint: str | None,
    project: str | None,
    seed: int | None,
    config: Path,
    corpus_dir: Path | None,
) -> None:
    """
    Generate synthetic OpenInference spans for replay evaluation.

    This CLI generates synthetic traces from the stub RAG pipeline and
    exports them to Phoenix for replay evaluation.

    Two modes:
        1. HTTP mode: --candidate-spec (calls external HTTP service)
        2. Import mode: default (uses direct imports, backward compatible)

    Example HTTP mode:
        generate-spans --candidate-spec \
            configs/candidates/stub-chunking-512.yaml --limit 3

    Example import mode:
        generate-spans --limit=5 --seed=42

    """
    try:
        # HTTP-based path
        if candidate_spec is not None:
            return _generate_via_http(
                candidate_spec=candidate_spec,
                limit=limit,
                config=config,
            )

        # Original import-based path (backward compatible)
        return _generate_via_import(
            limit=limit,
            phoenix_endpoint=phoenix_endpoint,
            project=project,
            seed=seed,
            config=config,
            corpus_dir=corpus_dir,
        )

    except FileNotFoundError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"ERROR: Unexpected error: {e}", err=True)
        sys.exit(1)


def _generate_via_http(
    candidate_spec: Path,
    limit: int | None,
    config: Path,
) -> None:
    """Generate spans by calling HTTP service."""
    from dataclasses import dataclass

    # Load candidate spec
    click.echo(f"Loading candidate spec from: {candidate_spec}")
    candidate_config = CandidateConfig.from_yaml_file(candidate_spec)

    # Create HTTP client
    client = HTTPClient(candidate_config)

    # Check health
    click.echo(f"Checking service health at: {candidate_config.candidate.service_url}")
    if not client.check_health():
        click.echo("ERROR: Service health check failed", err=True)
        sys.exit(1)
    click.echo("Service is healthy")

    # Load questions
    questions: list[GeneratorQuestion] = list(iter_questions(limit=limit))

    click.echo(f"Generating {len(questions)} spans via HTTP")
    click.echo(f"Service: {candidate_config.name}")
    click.echo(f"URL: {candidate_config.candidate.service_url}")

    # Generate spans via HTTP
    @dataclass
    class Result:
        successes: int
        failures: int
        run_id: str

    successes = 0
    failures = 0
    run_id = f"http-{candidate_config.name}"

    for idx, question in enumerate(questions, 1):
        try:
            payload = {
                "question": question.question,
                "top_k": 5,
            }
            client.query(payload)
            successes += 1
            click.echo(f"  [{idx}/{len(questions)}] {question.question[:50]}...")
        except Exception as e:
            failures += 1
            click.echo(f"  [{idx}/{len(questions)}] FAILED: {e}", err=True)

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Span generation complete (HTTP mode)")
    click.echo("=" * 50)
    click.echo(f"Successes: {successes}")
    click.echo(f"Failures: {failures}")
    click.echo(f"Run ID: {run_id}")

    if failures > 0:
        sys.exit(1)


def _generate_via_import(
    limit: int | None,
    phoenix_endpoint: str | None,
    project: str | None,
    seed: int | None,
    config: Path,
    corpus_dir: Path | None,
) -> None:
    """Generate spans using direct imports (original behavior)."""
    # Load config
    eval_config = load_config(config)
    gen_config = load_generator_config(config)

    # Apply CLI overrides
    if phoenix_endpoint:
        from dataclasses import replace

        gen_config = replace(gen_config, phoenix_endpoint=phoenix_endpoint)

    if project is not None:
        from dataclasses import replace

        gen_config = replace(gen_config, project_name=project)

    if seed is not None:
        from dataclasses import replace

        gen_config = replace(gen_config, seed=seed)

    # Determine corpus directory
    if corpus_dir is None:
        corpus_dir = Path(
            eval_config["datasets"]["legal_rag_bench"].get("path", "data/corpus")
        )

    # Print configuration
    click.echo(f"Generating synthetic spans for project: {gen_config.project_name}")
    click.echo(f"Phoenix endpoint: {gen_config.phoenix_endpoint}")
    click.echo(f"Corpus directory: {corpus_dir}")
    if limit:
        click.echo(f"Limit: {limit} questions")
    click.echo(f"Seed: {gen_config.seed}")

    # Run generator
    result = run_generator(
        config=gen_config,
        corpus_dir=corpus_dir,
        limit=limit,
    )

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Span generation complete (import mode)")
    click.echo("=" * 50)
    click.echo(f"Successes: {result.successes}")
    click.echo(f"Failures: {result.failures}")
    click.echo(f"Run ID: {result.run_id}")

    if result.failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
