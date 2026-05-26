"""
Phoenix client wrapper with datasets API extension.

This module extends PhoenixClient with dataset management capabilities.

PHOENIX NATIVE MIGRATION: Phase 2 - Dataset Migration
"""

from __future__ import annotations

from typing import Any, Final

from beartype import beartype
from beartype.typing import Dict, List

# Constants
DEFAULT_ENDPOINT: Final[str] = "http://localhost:6006"
DEFAULT_PROJECT_NAME: Final[str] = "case-assistant-synthetic"


@beartype
class PhoenixClientWithDatasets:
    """
    Phoenix client with datasets API extension.

    Extends Phoenix client functionality with dataset management methods:
    - create_dataset()
    - get_dataset()
    - list_dataset_versions()

    This wrapper provides a consistent API for dataset operations while
    maintaining compatibility with existing PhoenixClient patterns.

    Attributes:
        _endpoint: Phoenix server endpoint.
        _project_name: Phoenix project name.
        _client: Phoenix client instance (None if not connected).

    Example:
        >>> client = PhoenixClientWithDatasets()
        >>> dataset_id = client.create_dataset(
        ...     name="my-dataset",
        ...     dataframe=df,
        ...     input_keys=["question"],
        ...     output_keys=["expected_answer"]
        ... )
        >>> versions = client.list_dataset_versions(dataset_id)

    """

    __slots__ = ("_endpoint", "_project_name", "_client")

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        project_name: str = DEFAULT_PROJECT_NAME,
        base_client: Any = None,
    ) -> None:
        """
        Initialize Phoenix client with datasets extension.

        Args:
            endpoint: Phoenix server endpoint.
            project_name: Phoenix project name.
            base_client: Optional pre-initialized Phoenix client instance.

        """
        self._endpoint = endpoint
        self._project_name = project_name

        if base_client is not None:
            self._client = base_client
        else:
            self._client = self._initialize_client()

    def _initialize_client(self) -> Any | None:
        """
        Initialize Phoenix client.

        Returns:
            Phoenix client instance or None if initialization fails.

        """
        try:
            from phoenix.client import Client as PhoenixClientClass

            return PhoenixClientClass(base_url=self._endpoint)
        except (ImportError, AttributeError):
            try:
                # Try legacy import path
                from arize_phoenix.client import Client as PhoenixClientClass

                # Legacy might use endpoint or base_url
                try:
                    return PhoenixClientClass(base_url=self._endpoint)
                except TypeError:
                    return PhoenixClientClass(endpoint=self._endpoint)
            except (ImportError, AttributeError):
                # Phoenix not available - return None
                return None

    @beartype
    def is_connected(self) -> bool:
        """
        Check if Phoenix client is connected.

        Returns:
            True if connected, False otherwise.

        """
        return self._client is not None

    @beartype
    def create_dataset(
        self,
        name: str,
        dataframe: Any,
        input_keys: List[str],
        output_keys: List[str],
    ) -> Dict[str, Any]:
        """
        Create a Phoenix dataset from a pandas DataFrame.

        Uses client.datasets.create_dataset() with proper schema.

        Args:
            name: Dataset name.
            dataframe: pandas DataFrame with dataset data.
            input_keys: List of input column names (e.g., ["question"]).
            output_keys: List of output column names (e.g., ["expected_answer"]).

        Returns:
            Dictionary with dataset_id and version.

        Raises:
            ConnectionError: If Phoenix client is not connected.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     "question": ["What is contract law?"],
            ...     "expected_answer": ["Contract law governs..."]
            ... })
            >>> result = client.create_dataset(
            ...     name="my-dataset",
            ...     dataframe=df,
            ...     input_keys=["question"],
            ...     output_keys=["expected_answer"]
            ... )
            >>> print(result["dataset_id"])

        """
        if not self.is_connected():
            raise ConnectionError("Phoenix client is not connected")

        try:
            # Create dataset using Phoenix API
            dataset = self._client.datasets.create_dataset(
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
    def get_dataset(
        self,
        dataset_id: str,
        version: str | None = None,
    ) -> Any:
        """
        Get a Phoenix dataset by ID and optionally version.

        Args:
            dataset_id: Dataset ID.
            version: Optional version identifier.

        Returns:
            Dataset object or DataFrame.

        Raises:
            ConnectionError: If Phoenix client is not connected.

        Example:
            >>> dataset = client.get_dataset(dataset_id="my-dataset-id")
            >>> print(dataset.head())

        """
        if not self.is_connected():
            raise ConnectionError("Phoenix client is not connected")

        try:
            if version:
                dataset = self._client.datasets.get_dataset(dataset_id, version=version)
            else:
                dataset = self._client.datasets.get_dataset(dataset_id)

            return dataset
        except Exception as e:
            import sys

            print(f"[WARN] Failed to get dataset: {e}", file=sys.stderr)
            return None

    @beartype
    def list_dataset_versions(
        self,
        dataset_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get versions of a Phoenix dataset.

        Args:
            dataset_id: Dataset ID.

        Returns:
            List of dataset version dictionaries.

        Raises:
            ConnectionError: If Phoenix client is not connected.

        Example:
            >>> versions = client.list_dataset_versions(dataset_id="my-dataset-id")
            >>> for version in versions:
            ...     print(f"Version {version['version_id']}: {version['created_at']}")

        """
        if not self.is_connected():
            raise ConnectionError("Phoenix client is not connected")

        try:
            versions = self._client.datasets.get_dataset_versions(dataset_id)

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

    # Preserve existing PhoenixClient methods for compatibility
    @beartype
    def query_root_spans(
        self,
        limit: int = 100,
        case_id: str | None = None,
        tenant_id_hashed: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query root spans with synthetic marker.

        Preserved from existing PhoenixClient for compatibility.

        Args:
            limit: Maximum number of spans to return.
            case_id: Optional filter by case_id.
            tenant_id_hashed: Optional filter by tenant_id_hashed.

        Returns:
            List of root span dictionaries.

        Raises:
            ConnectionError: If Phoenix client is not connected.

        """
        if not self.is_connected():
            raise ConnectionError("Phoenix client is not connected")

        # Query spans using Phoenix client
        try:
            spans_df = self._client.spans.get_spans_dataframe(
                project_name=self._project_name
            )
        except Exception:
            return []

        # Filter for root spans (parent_id is NaN or empty)
        root_mask = spans_df["parent_id"].isna() | (spans_df["parent_id"] == "")
        root_spans = spans_df[root_mask]

        # Filter for synthetic spans by name (synthetic_rag_query)
        if "name" in root_spans.columns:
            root_spans = root_spans[
                root_spans["name"] == "synthetic_rag_query"
            ]

        # Apply additional filters
        if case_id is not None:
            case_col = "attributes.eval_harness.case_id"
            if case_col in root_spans.columns:
                root_spans = root_spans[root_spans[case_col] == case_id]

        if tenant_id_hashed is not None:
            tenant_col = "attributes.eval_harness.tenant_id_hashed"
            if tenant_col in root_spans.columns:
                root_spans = root_spans[root_spans[tenant_col] == tenant_id_hashed]

        # Convert to list of dicts and limit
        return root_spans.head(limit).to_dict("records")

    @beartype
    def extract_child_spans(
        self,
        parent_span: dict[str, Any],
        span_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract child spans from a parent span.

        Preserved from existing PhoenixClient for compatibility.

        Args:
            parent_span: Parent span dictionary.
            span_kind: Optional filter by span_kind (EMBEDDING, RETRIEVER, LLM).

        Returns:
            List of child span dictionaries.

        """
        if not self.is_connected():
            return []

        parent_span_id = parent_span.get("span_id") or parent_span.get("id")
        if not parent_span_id:
            return []

        try:
            # Try to use query_spans method
            if hasattr(self._client, "query_spans"):
                filters = {"parent_id": parent_span_id}
                if span_kind is not None:
                    filters["span_kind"] = span_kind.upper()
                return self._client.query_spans(filters, limit=100)
            # Try to use spans() method
            elif hasattr(self._client, "spans"):
                spans_df = self._client.spans(project_name=self._project_name)
                # Filter for child spans
                child_spans = spans_df[spans_df.get("parent_id") == parent_span_id]
                # Filter by span_kind if specified
                if span_kind is not None:
                    child_spans = child_spans[
                        child_spans.get("span_kind").str.upper() == span_kind.upper()
                    ]
                return child_spans.to_dict("records")
            else:
                return []
        except Exception:
            return []
