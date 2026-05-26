"""
Phoenix client wrapper with server-side query support.

PHOENIX NATIVE MIGRATION: Phase 5.2 - Server-Side Query Implementation
Enhanced Phoenix client with server-side filtering to reduce memory usage.
"""

from __future__ import annotations

from beartype import beartype
from beartype.typing import Any, Final

# Constants
DEFAULT_ENDPOINT: Final[str] = "http://localhost:6006"
DEFAULT_PROJECT_NAME: Final[str] = "case-assistant-synthetic"


@beartype
class PhoenixClientServerSide:
    """
    Phoenix client with server-side query support.

    Uses Phoenix query_spans() API for server-side filtering instead of
    loading all spans into memory with get_spans_dataframe().

    Attributes:
        _endpoint: Phoenix server endpoint.
        _project_name: Phoenix project name.
        _client: Phoenix client instance (None if not connected).

    Example:
        >>> client = PhoenixClientServerSide()
        >>> root_spans = client.query_root_spans(limit=10)
        >>> for span in root_spans:
        ...     children = client.extract_child_spans(span)

    """

    __slots__ = ("_endpoint", "_project_name", "_client")

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        project_name: str = DEFAULT_PROJECT_NAME,
    ) -> None:
        """
        Initialize Phoenix client.

        Args:
            endpoint: Phoenix server endpoint.
            project_name: Phoenix project name.

        """
        self._endpoint = endpoint
        self._project_name = project_name
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
                from arize_phoenix.client import Client as PhoenixClientClass

                try:
                    return PhoenixClientClass(base_url=self._endpoint)
                except TypeError:
                    return PhoenixClientClass(endpoint=self._endpoint)
            except (ImportError, AttributeError):
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
    def query_root_spans(
        self,
        limit: int = 100,
        case_id: str | None = None,
        tenant_id_hashed: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query root spans with server-side filtering.

        Uses query_spans() API for server-side filtering instead of loading
        all spans into memory.

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

        # Try server-side query first
        if hasattr(self._client, "query_spans"):
            try:
                return self._query_root_spans_server_side(
                    limit=limit,
                    case_id=case_id,
                    tenant_id_hashed=tenant_id_hashed,
                )
            except Exception:
                # Fall back to DataFrame query
                pass

        # Fallback to DataFrame query
        return self._query_root_spans_dataframe(
            limit=limit,
            case_id=case_id,
            tenant_id_hashed=tenant_id_hashed,
        )

    @beartype
    def _query_root_spans_server_side(
        self,
        limit: int,
        case_id: str | None,
        tenant_id_hashed: str | None,
    ) -> list[dict[str, Any]]:
        """
        Query root spans using server-side filtering.

        Builds filters for server-side query to reduce memory usage.

        Args:
            limit: Maximum number of spans to return.
            case_id: Optional filter by case_id.
            tenant_id_hashed: Optional filter by tenant_id_hashed.

        Returns:
            List of root span dictionaries.

        """
        # Build filters for server-side query
        filters: dict[str, Any] = {
            "name": "synthetic_rag_query",
        }

        # Add case_id filter if provided
        if case_id is not None:
            filters["attributes.eval_harness.case_id"] = case_id

        # Add tenant_id filter if provided
        if tenant_id_hashed is not None:
            filters["attributes.eval_harness.tenant_id_hashed"] = tenant_id_hashed

        # Execute server-side query
        results = self._client.query_spans(filters, limit=limit)

        # Filter for root spans (those without parent_id)
        root_spans = [
            span for span in results
            if not span.get("parent_id") and not span.get("parent_id") == ""
        ]

        return root_spans

    @beartype
    def _query_root_spans_dataframe(
        self,
        limit: int,
        case_id: str | None,
        tenant_id_hashed: str | None,
    ) -> list[dict[str, Any]]:
        """
        Query root spans using DataFrame (fallback).

        This is the original implementation that loads all spans into memory.
        Used as fallback when server-side query is not available.

        Args:
            limit: Maximum number of spans to return.
            case_id: Optional filter by case_id.
            tenant_id_hashed: Optional filter by tenant_id_hashed.

        Returns:
            List of root span dictionaries.

        """
        import pandas as pd

        try:
            spans_df = self._client.spans.get_spans_dataframe(
                project_name=self._project_name
            )
        except Exception:
            return []

        # Filter for root spans
        root_mask = spans_df["parent_id"].isna() | (spans_df["parent_id"] == "")
        root_spans = spans_df[root_mask]

        # Filter for synthetic spans
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
        Extract child spans using server-side filtering.

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

        # Try server-side query first
        if hasattr(self._client, "query_spans"):
            try:
                filters = {"parent_id": parent_span_id}
                if span_kind is not None:
                    filters["span_kind"] = span_kind.upper()
                return self._client.query_spans(filters, limit=100)
            except Exception:
                # Fall back to DataFrame query
                pass

        # Fallback to DataFrame query
        return self._extract_child_spans_dataframe(parent_span_id, span_kind)

    @beartype
    def _extract_child_spans_dataframe(
        self,
        parent_span_id: str,
        span_kind: str | None,
    ) -> list[dict[str, Any]]:
        """
        Extract child spans using DataFrame (fallback).

        Args:
            parent_span_id: Parent span ID.
            span_kind: Optional filter by span_kind.

        Returns:
            List of child span dictionaries.

        """
        import pandas as pd

        try:
            spans_df = self._client.spans(project_name=self._project_name)
            child_spans = spans_df[spans_df.get("parent_id") == parent_span_id]

            if span_kind is not None:
                child_spans = child_spans[
                    child_spans.get("span_kind").str.upper() == span_kind.upper()
                ]

            return child_spans.to_dict("records")
        except Exception:
            return []
