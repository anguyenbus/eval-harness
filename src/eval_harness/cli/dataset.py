"""
Phoenix dataset management CLI commands.

This module provides CLI commands for managing Phoenix datasets:
- list: List all datasets with versions
- upload: Upload dataset from spans or file
- download: Download dataset to file
- validate: Validate dataset schema and completeness

PHOENIX NATIVE MIGRATION: Phase 2.3 - CLI Dataset Management Commands
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import click
import pandas as pd

from eval_harness.replay.phoenix_client_datasets import PhoenixClientWithDatasets
from eval_harness.replay.phoenix_datasets import (
    create_phoenix_dataset,
    extract_dataset_from_spans,
)

# Constants
DEFAULT_ENDPOINT: Final[str] = "http://localhost:6006"
DEFAULT_PROJECT_NAME: Final[str] = "case-assistant-synthetic"


@click.group()
@click.option(
    "--endpoint",
    envvar="PHOENIX_ENDPOINT",
    default=DEFAULT_ENDPOINT,
    help="Phoenix server endpoint (default: $PHOENIX_ENDPOINT or http://localhost:6006)",
)
@click.option(
    "--project",
    default=DEFAULT_PROJECT_NAME,
    help="Phoenix project name (default: case-assistant-synthetic)",
)
@click.pass_context
def dataset(ctx: click.Context, endpoint: str, project: str) -> None:
    """
    Manage Phoenix datasets.

    Dataset management commands for extracting, uploading, downloading,
    and validating Phoenix datasets used in RAG evaluation.

    Examples:
        eval-dataset list
        eval-dataset upload --name my-dataset --from-spans
        eval-dataset download --dataset-id abc123 --output dataset.csv
        eval-dataset validate --file dataset.csv
    """
    ctx.ensure_object(dict)
    ctx.obj["endpoint"] = endpoint
    ctx.obj["project"] = project


@dataset.command("list")
@click.option(
    "--dataset-id",
    default=None,
    help="Filter by specific dataset ID (shows versions)",
)
@click.pass_context
def list_datasets(ctx: click.Context, dataset_id: str | None) -> None:
    """
    List all datasets or versions of a specific dataset.

    Lists all available datasets in Phoenix or shows version history
    for a specific dataset.

    Examples:
        eval-dataset list
        eval-dataset list --dataset-id my-dataset-id
    """
    endpoint = ctx.obj["endpoint"]

    client = PhoenixClientWithDatasets(endpoint=endpoint)

    if not client.is_connected():
        click.echo(
            click.style("ERROR: Phoenix client is not connected", fg="red"),
            err=True,
        )
        raise SystemExit(1)

    if dataset_id:
        # Show versions for specific dataset
        versions = client.list_dataset_versions(dataset_id)

        if not versions:
            click.echo(f"No versions found for dataset: {dataset_id}")
            return

        click.echo(f"Dataset versions for: {click.style(dataset_id, fg='cyan', bold=True)}")
        click.echo()

        for version in versions:
            version_id = version.get("version_id", "unknown")
            created_at = version.get("created_at", "unknown")
            click.echo(f"  Version: {click.style(version_id, fg='green')}")
            click.echo(f"    Created: {created_at}")
    else:
        # List all datasets (using dataset versions endpoint)
        # NOTE: Phoenix doesn't have a direct list_datasets endpoint
        # So we provide guidance on how to view datasets
        click.echo("Phoenix Dataset Management")
        click.echo()
        click.echo("To view datasets in Phoenix UI:")
        click.echo(f"  1. Open {endpoint} in your browser")
        click.echo(f"  2. Navigate to the Datasets section")
        click.echo()
        click.echo("To get dataset versions, use --dataset-id:")
        click.echo("  eval-dataset list --dataset-id <dataset-id>")


@dataset.command("upload")
@click.option(
    "--name",
    required=True,
    help="Dataset name (required)",
)
@click.option(
    "--from-spans",
    is_flag=True,
    default=False,
    help="Extract dataset from existing Phoenix spans",
)
@click.option(
    "--span-name",
    default="rag_query",
    help="Span name to extract (default: rag_query)",
)
@click.option(
    "--file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Upload dataset from CSV file",
)
@click.option(
    "--input-keys",
    default="question",
    help="Input column names (comma-separated, default: question)",
)
@click.option(
    "--output-keys",
    default="expected_answer",
    help="Output column names (comma-separated, default: expected_answer)",
)
@click.pass_context
def upload_dataset(
    ctx: click.Context,
    name: str,
    from_spans: bool,
    span_name: str,
    file: Path | None,
    input_keys: str,
    output_keys: str,
) -> None:
    """
    Upload a dataset to Phoenix.

    Upload a dataset either by extracting from existing Phoenix spans
    or from a local CSV file.

    Examples:
        eval-dataset upload --name my-dataset --from-spans
        eval-dataset upload --name my-dataset --file local_dataset.csv
        eval-dataset upload --name my-dataset --file data.csv --input-keys question --output-keys answer
    """
    endpoint = ctx.obj["endpoint"]
    project = ctx.obj["project"]

    if not from_spans and file is None:
        click.echo(
            click.style(
                "ERROR: Must specify either --from-spans or --file",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    if from_spans and file is not None:
        click.echo(
            click.style(
                "ERROR: Cannot use both --from-spans and --file",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    client = PhoenixClientWithDatasets(endpoint=endpoint)

    if not client.is_connected():
        click.echo(
            click.style("ERROR: Phoenix client is not connected", fg="red"),
            err=True,
        )
        raise SystemExit(1)

    # Parse column keys
    input_keys_list = [k.strip() for k in input_keys.split(",")]
    output_keys_list = [k.strip() for k in output_keys.split(",")]

    click.echo(f"Creating dataset: {click.style(name, fg='cyan', bold=True)}")

    if from_spans:
        # Extract dataset from spans
        click.echo(f"Extracting from spans (name: {span_name})...")

        df = extract_dataset_from_spans(
            client=client._client,
            project_name=project,
            span_name=span_name,
        )

        if df.empty:
            click.echo(
                click.style(
                    "WARNING: No spans found. Dataset will be empty.",
                    fg="yellow",
                )
            )
        else:
            click.echo(f"Extracted {len(df)} rows from spans")
    else:
        # Load from file
        click.echo(f"Loading from file: {file}")

        try:
            df = pd.read_csv(file)
            click.echo(f"Loaded {len(df)} rows from file")
        except Exception as e:
            click.echo(
                click.style(f"ERROR: Failed to read file: {e}", fg="red"),
                err=True,
            )
            raise SystemExit(1)

    # Validate required columns exist
    missing_input = [k for k in input_keys_list if k not in df.columns]
    missing_output = [k for k in output_keys_list if k not in df.columns]

    if missing_input or missing_output:
        click.echo(
            click.style(
                f"ERROR: Missing required columns: "
                f"input={missing_input}, output={missing_output}",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    # Create dataset in Phoenix
    result = create_phoenix_dataset(
        client=client._client,
        name=name,
        dataframe=df,
        input_keys=input_keys_list,
        output_keys=output_keys_list,
    )

    dataset_id = result.get("dataset_id")
    version = result.get("version")

    if dataset_id:
        click.echo(
            click.style("SUCCESS", fg="green", bold=True) +
            f": Dataset created"
        )
        click.echo(f"  Dataset ID: {click.style(dataset_id, fg='cyan')}")
        click.echo(f"  Version: {click.style(version, fg='cyan')}")
    else:
        error = result.get("error", "Unknown error")
        click.echo(
            click.style(f"ERROR: Failed to create dataset: {error}", fg="red"),
            err=True,
        )
        raise SystemExit(1)


@dataset.command("download")
@click.option(
    "--dataset-id",
    required=True,
    help="Dataset ID to download (required)",
)
@click.option(
    "--version",
    default=None,
    help="Dataset version (default: latest)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path (required)",
)
@click.option(
    "--format",
    type=click.Choice(["csv", "json"], case_sensitive=False),
    default="csv",
    help="Output format (default: csv)",
)
@click.pass_context
def download_dataset(
    ctx: click.Context,
    dataset_id: str,
    version: str | None,
    output: Path,
    format: str,
) -> None:
    """
    Download a dataset from Phoenix.

    Download a dataset from Phoenix and save it to a local file
    in CSV or JSON format.

    Examples:
        eval-dataset download --dataset-id abc123 --output dataset.csv
        eval-dataset download --dataset-id abc123 --version v1 --output data.json --format json
    """
    endpoint = ctx.obj["endpoint"]

    client = PhoenixClientWithDatasets(endpoint=endpoint)

    if not client.is_connected():
        click.echo(
            click.style("ERROR: Phoenix client is not connected", fg="red"),
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Downloading dataset: {click.style(dataset_id, fg='cyan')}")

    if version:
        click.echo(f"Version: {version}")

    # Get dataset from Phoenix
    dataset_obj = client.get_dataset(dataset_id, version=version)

    if dataset_obj is None:
        click.echo(
            click.style(
                f"ERROR: Dataset not found: {dataset_id}",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    # Convert to DataFrame if needed
    if not isinstance(dataset_obj, pd.DataFrame):
        # If it's a Phoenix dataset object, convert to DataFrame
        try:
            df = pd.DataFrame(dataset_obj.data)
        except (AttributeError, TypeError):
            try:
                # Try alternative conversion methods
                df = dataset_obj.to_pandas()
            except AttributeError:
                click.echo(
                    click.style(
                        "ERROR: Cannot convert dataset to DataFrame",
                        fg="red",
                    ),
                    err=True,
                )
                raise SystemExit(1)
    else:
        df = dataset_obj

    click.echo(f"Downloaded {len(df)} rows")

    # Write to file
    try:
        if format == "csv":
            df.to_csv(output, index=False)
        else:  # json
            df.to_json(output, orient="records", indent=2)

        click.echo(
            click.style("SUCCESS", fg="green", bold=True) +
            f": Dataset saved to {output}"
        )
    except Exception as e:
        click.echo(
            click.style(f"ERROR: Failed to write file: {e}", fg="red"),
            err=True,
        )
        raise SystemExit(1)


@dataset.command("validate")
@click.option(
    "--file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Dataset file to validate (required)",
)
@click.option(
    "--input-keys",
    default="question",
    help="Required input column names (comma-separated, default: question)",
)
@click.option(
    "--output-keys",
    default="expected_answer",
    help="Required output column names (comma-separated, default: expected_answer)",
)
def validate_dataset(
    file: Path,
    input_keys: str,
    output_keys: str,
) -> None:
    """
    Validate a dataset file.

    Validate that a dataset file has the required schema and
    check for completeness (no missing values).

    Examples:
        eval-dataset validate --file dataset.csv
        eval-dataset validate --file dataset.csv --input-keys question --output-keys answer
    """
    click.echo(f"Validating dataset: {click.style(str(file), fg='cyan')}")
    click.echo()

    # Load file
    try:
        df = pd.read_csv(file)
        click.echo(f"Loaded {len(df)} rows")
    except Exception as e:
        click.echo(
            click.style(f"ERROR: Failed to read file: {e}", fg="red"),
            err=True,
        )
        raise SystemExit(1)

    # Parse column keys
    input_keys_list = [k.strip() for k in input_keys.split(",")]
    output_keys_list = [k.strip() for k in output_keys.split(",")]
    required_keys = input_keys_list + output_keys_list

    # Check for required columns
    missing_columns = [k for k in required_keys if k not in df.columns]

    if missing_columns:
        click.echo(
            click.style(
                f"FAILED: Missing required columns: {missing_columns}",
                fg="red",
                bold=True,
            )
        )
        click.echo(f"  Found columns: {list(df.columns)}")
        raise SystemExit(1)

    click.echo(
        click.style(
            f"PASSED: All required columns present: {required_keys}",
            fg="green",
        )
    )

    # Check for empty values
    empty_count = 0
    empty_details = []

    for col in required_keys:
        nulls = df[col].isna().sum()
        empty_strings = (df[col] == "").sum()
        total_empty = nulls + empty_strings

        if total_empty > 0:
            empty_count += total_empty
            empty_details.append(f"  {col}: {total_empty} empty values")

    if empty_count > 0:
        click.echo()
        click.echo(
            click.style(
                f"WARNING: Found {empty_count} empty values:",
                fg="yellow",
            )
        )
        for detail in empty_details:
            click.echo(detail)

    # Final result
    click.echo()
    if empty_count > len(df) * 0.1:  # More than 10% empty
        click.echo(
            click.style(
                f"WARNING: Dataset has {empty_count/len(df)*100:.1f}% empty values",
                fg="yellow",
                bold=True,
            )
        )
        click.echo("  Consider cleaning the data before use")
    else:
        click.echo(
            click.style(
                "SUCCESS: Dataset validation passed",
                fg="green",
                bold=True,
            )
        )
