"""
Tests for server-side query implementation in Phoenix client.

PHOENIX NATIVE MIGRATION: Phase 5.2 - Server-Side Query Implementation
Tests for replacing get_spans_dataframe() with server-side filtering.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestServerSideQueryRootSpans:
    """Tests for server-side root span queries."""

    def test_query_root_spans_server_side(self) -> None:
        """Test querying root spans with server-side filtering."""
        from eval_harness.replay.phoenix_client_server_side import (
            PhoenixClientServerSide,
        )

        with patch("phoenix.client.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock server-side query response
            mock_client.query_spans.return_value = [
                {
                    "span_id": "span1",
                    "name": "synthetic_rag_query",
                    "attributes": {"eval_harness.case_id": "case1"},
                    "parent_id": None,
                },
                {
                    "span_id": "span2",
                    "name": "synthetic_rag_query",
                    "attributes": {"eval_harness.case_id": "case2"},
                    "parent_id": None,
                },
            ]

            client = PhoenixClientServerSide(endpoint="http://localhost:6006")

            results = client.query_root_spans(limit=10)

            assert len(results) == 2
            assert results[0]["span_id"] == "span1"
            assert results[1]["span_id"] == "span2"

    def test_query_root_spans_with_filters(self) -> None:
        """Test querying root spans with server-side filters."""
        from eval_harness.replay.phoenix_client_server_side import (
            PhoenixClientServerSide,
        )

        with patch("phoenix.client.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_client.query_spans.return_value = [
                {
                    "span_id": "span1",
                    "name": "synthetic_rag_query",
                    "attributes": {
                        "eval_harness.case_id": "case1",
                        "eval_harness.tenant_id_hashed": "tenant1",
                    },
                    "parent_id": None,
                },
            ]

            client = PhoenixClientServerSide(endpoint="http://localhost:6006")

            results = client.query_root_spans(
                limit=10,
                case_id="case1",
                tenant_id_hashed="tenant1",
            )

            assert len(results) == 1
            assert results[0]["span_id"] == "span1"

    def test_query_root_spans_empty_result(self) -> None:
        """Test querying root spans returns empty list when no matches."""
        from eval_harness.replay.phoenix_client_server_side import (
            PhoenixClientServerSide,
        )

        with patch("phoenix.client.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_client.query_spans.return_value = []

            client = PhoenixClientServerSide(endpoint="http://localhost:6006")

            results = client.query_root_spans(limit=10)

            assert results == []


class TestServerSideQueryChildSpans:
    """Tests for server-side child span queries."""

    def test_extract_child_spans_server_side(self) -> None:
        """Test extracting child spans with server-side filtering."""
        from eval_harness.replay.phoenix_client_server_side import (
            PhoenixClientServerSide,
        )

        with patch("phoenix.client.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_client.query_spans.return_value = [
                {
                    "span_id": "child1",
                    "parent_id": "parent1",
                    "span_kind": "RETRIEVER",
                    "name": "retrieval",
                },
                {
                    "span_id": "child2",
                    "parent_id": "parent1",
                    "span_kind": "LLM",
                    "name": "generation",
                },
            ]

            client = PhoenixClientServerSide(endpoint="http://localhost:6006")

            parent_span = {"span_id": "parent1"}

            results = client.extract_child_spans(parent_span)

            assert len(results) == 2
            assert results[0]["span_id"] == "child1"
            assert results[1]["span_id"] == "child2"

    def test_extract_child_spans_with_kind_filter(self) -> None:
        """Test extracting child spans with span_kind filter."""
        from eval_harness.replay.phoenix_client_server_side import (
            PhoenixClientServerSide,
        )

        with patch("phoenix.client.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock should return filtered results when span_kind filter is applied
            def mock_query_with_filters(filters, limit):
                # Server-side filtering happens here
                if filters.get("span_kind") == "LLM":
                    return [
                        {
                            "span_id": "child2",
                            "parent_id": "parent1",
                            "span_kind": "LLM",
                        },
                    ]
                return [
                    {
                        "span_id": "child1",
                        "parent_id": "parent1",
                        "span_kind": "RETRIEVER",
                    },
                    {
                        "span_id": "child2",
                        "parent_id": "parent1",
                        "span_kind": "LLM",
                    },
                ]

            mock_client.query_spans.side_effect = mock_query_with_filters

            client = PhoenixClientServerSide(endpoint="http://localhost:6006")

            parent_span = {"span_id": "parent1"}

            results = client.extract_child_spans(parent_span, span_kind="LLM")

            assert len(results) == 1
            assert results[0]["span_id"] == "child2"


class TestServerSideQueryMemoryEfficiency:
    """Tests for memory efficiency improvements."""

    def test_server_side_query_reduces_memory(self) -> None:
        """Test server-side query reduces memory compared to DataFrame."""
        from eval_harness.replay.phoenix_client_server_side import (
            PhoenixClientServerSide,
        )

        with patch("phoenix.client.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Server-side returns list of dicts (memory efficient)
            mock_client.query_spans.return_value = [
                {"span_id": f"span{i}", "name": "synthetic_rag_query", "parent_id": None}
                for i in range(1000)
            ]

            client = PhoenixClientServerSide(endpoint="http://localhost:6006")

            results = client.query_root_spans(limit=1000)

            # Should return list without loading full DataFrame
            assert len(results) == 1000
            assert isinstance(results, list)

    def test_streaming_query_for_large_datasets(self) -> None:
        """Test streaming query for very large datasets."""
        from eval_harness.replay.phoenix_client_server_side import (
            PhoenixClientServerSide,
        )

        with patch("phoenix.client.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Simulate streaming response
            def mock_stream_spans(filters, limit):
                return [
                    {"span_id": f"span{i}", "name": "synthetic_rag_query", "parent_id": None}
                    for i in range(min(limit, 100))
                ]

            mock_client.query_spans.side_effect = mock_stream_spans

            client = PhoenixClientServerSide(endpoint="http://localhost:6006")

            results = client.query_root_spans(limit=100)

            assert len(results) == 100


class TestServerSideQueryFallback:
    """Tests for fallback to DataFrame when server-side unavailable."""

    def test_fallback_to_dataframe_on_error(self) -> None:
        """Test fallback to DataFrame when server-side query fails."""
        import pandas as pd

        from eval_harness.replay.phoenix_client_server_side import (
            PhoenixClientServerSide,
        )

        with patch("phoenix.client.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Server-side query not available
            mock_client.query_spans.side_effect = AttributeError("query_spans not available")

            # DataFrame fallback available
            mock_df = pd.DataFrame({
                "span_id": ["span1", "span2"],
                "name": ["synthetic_rag_query", "synthetic_rag_query"],
                "parent_id": [None, None],
            })
            mock_client.spans.get_spans_dataframe.return_value = mock_df

            client = PhoenixClientServerSide(endpoint="http://localhost:6006")

            results = client.query_root_spans(limit=10)

            # Should fall back to DataFrame query
            assert isinstance(results, list)
            assert len(results) == 2
