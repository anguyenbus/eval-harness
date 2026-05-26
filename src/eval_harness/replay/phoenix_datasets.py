"""
Phoenix dataset extraction and management utilities.

This module provides utility functions for extracting datasets from Phoenix spans
and managing Phoenix datasets via the datasets API.

PHOENIX NATIVE MIGRATION: Phase 2 - Dataset Migration
"""

from __future__ import annotations

from typing import Any, Final

from beartype import beartype
from beartype.typing import Dict, List

# Constants
DEFAULT_PROJECT_NAME: Final[str] = "eval-harness"


@beartype
def extract_dataset_from_spans(
    client: Any,
    project_name: str = DEFAULT_PROJECT_NAME,
    span_name: str = "rag_query",
) -> Any:
    """
    Extract question/expected answer pairs from Phoenix spans.

    Uses client.spans.get_spans_dataframe() to extract existing spans
    and filters for question/expected answer pairs.

    Args:
        client: Phoenix client instance.
        project_name: Phoenix project name.
        span_name: Name of spans to extract (default: "rag_query").

    Returns:
        pandas DataFrame with columns: question, expected_answer.

    Example:
        >>> from phoenix.client import Client
        >>> client = Client()
        >>> df = extract_dataset_from_spans(client, project_name="my-project")
        >>> print(df.head())

    """
    import pandas as pd

    # Get spans DataFrame
    try:
        spans_df = client.spans.get_spans_dataframe(project_name=project_name)
    except Exception as e:
        import sys

        print(f"[WARN] Failed to get spans DataFrame: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["question", "expected_answer"])

    # Filter for root spans (parent_id is NaN or empty)
    root_mask = spans_df["parent_id"].isna() | (spans_df["parent_id"] == "")
    root_spans = spans_df[root_mask]

    # Filter for specified span name (rag_query by default)
    if "name" in root_spans.columns:
        root_spans = root_spans[root_spans["name"] == span_name]

    # Extract question and expected answer
    questions = []
    expected_answers = []

    for _, span in root_spans.iterrows():
        # Extract question from input.value attribute
        question = None
        if "attributes.input.value" in span and pd.notna(span["attributes.input.value"]):
            question = span["attributes.input.value"]
        elif "input.value" in span and pd.notna(span["input.value"]):
            question = span["input.value"]
        elif "attributes.question" in span and pd.notna(span["attributes.question"]):
            question = span["attributes.question"]

        # Extract expected answer from output.value attribute
        expected_answer = None
        if "attributes.output.value" in span and pd.notna(span["attributes.output.value"]):
            expected_answer = span["attributes.output.value"]
        elif "output.value" in span and pd.notna(span["output.value"]):
            expected_answer = span["output.value"]
        elif "attributes.expected_answer" in span and pd.notna(span["attributes.expected_answer"]):
            expected_answer = span["attributes.expected_answer"]

        if question and expected_answer:
            questions.append(str(question))
            expected_answers.append(str(expected_answer))

    # Create DataFrame
    return pd.DataFrame({
        "question": questions,
        "expected_answer": expected_answers,
    })


@beartype
def create_phoenix_dataset(
    client: Any,
    name: str,
    dataframe: Any,
    input_keys: List[str],
    output_keys: List[str],
) -> Dict[str, Any]:
    """
    Create a Phoenix dataset from a pandas DataFrame.

    Uses client.datasets.create_dataset() with proper schema.

    Args:
        client: Phoenix client instance.
        name: Dataset name.
        dataframe: pandas DataFrame with dataset data.
        input_keys: List of input column names (e.g., ["question"]).
        output_keys: List of output column names (e.g., ["expected_answer"]).

    Returns:
        Dictionary with dataset_id and version.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "question": ["What is contract law?"],
        ...     "expected_answer": ["Contract law governs..."]
        ... })
        >>> result = create_phoenix_dataset(
        ...     client,
        ...     name="my-dataset",
        ...     dataframe=df,
        ...     input_keys=["question"],
        ...     output_keys=["expected_answer"]
        ... )
        >>> print(result["dataset_id"])

    """
    try:
        # Create dataset using Phoenix API
        dataset = client.datasets.create_dataset(
            dataset_name=name,
            input_keys=input_keys,
            output_keys=output_keys,
            data=dataframe,
        )

        return {
            "dataset_id": dataset.dataset_id,
            "version": getattr(dataset, "version", "1"),
        }
    except Exception as e:
        import sys

        print(f"[WARN] Failed to create dataset: {e}", file=sys.stderr)
        return {
            "dataset_id": None,
            "version": None,
            "error": str(e),
        }


@beartype
def get_dataset_versions(
    client: Any,
    dataset_id: str,
) -> List[Dict[str, Any]]:
    """
    Get versions of a Phoenix dataset.

    Uses client.datasets.get_dataset_versions() for versioning.

    Args:
        client: Phoenix client instance.
        dataset_id: Dataset ID.

    Returns:
        List of dataset version dictionaries.

    Example:
        >>> versions = get_dataset_versions(client, dataset_id="my-dataset-id")
        >>> for version in versions:
        ...     print(f"Version {version['version_id']}: {version['created_at']}")

    """
    try:
        versions = client.datasets.get_dataset_versions(dataset_id)

        result = []
        for version in versions:
            result.append({
                "version_id": getattr(version, "version_id", "unknown"),
                "created_at": getattr(version, "created_at", None),
            })

        return result
    except Exception as e:
        import sys

        print(f"[WARN] Failed to get dataset versions: {e}", file=sys.stderr)
        return []


@beartype
def get_dataset(
    client: Any,
    dataset_id: str,
    version: str | None = None,
) -> Any:
    """
    Get a Phoenix dataset by ID and optionally version.

    Uses client.datasets.get_dataset().

    Args:
        client: Phoenix client instance.
        dataset_id: Dataset ID.
        version: Optional version identifier.

    Returns:
        Dataset object or DataFrame.

    Example:
        >>> dataset = get_dataset(client, dataset_id="my-dataset-id")
        >>> print(dataset.head())

    """
    try:
        if version:
            dataset = client.datasets.get_dataset(dataset_id, version=version)
        else:
            dataset = client.datasets.get_dataset(dataset_id)

        return dataset
    except Exception as e:
        import sys

        print(f"[WARN] Failed to get dataset: {e}", file=sys.stderr)
        return None
