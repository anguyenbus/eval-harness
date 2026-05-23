"""
CLI runner for synthetic span generation.

Usage:
    uv run generate-spans --limit=100 --seed=42

Generates synthetic OpenInference-compliant spans from the stub RAG pipeline
for testing evaluation harnesses before production traffic exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from eval_harness.config import load_config
from eval_harness.stubs.span_generator.config import load_generator_config
from eval_harness.stubs.span_generator.runner import run_generator


@click.command()
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

    Example:
        generate-spans --limit=5 --seed=42

    """
    try:
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
        click.echo("Span generation complete")
        click.echo("=" * 50)
        click.echo(f"Successes: {result.successes}")
        click.echo(f"Failures: {result.failures}")
        click.echo(f"Run ID: {result.run_id}")

        if result.failures > 0:
            sys.exit(1)

    except FileNotFoundError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"ERROR: Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
