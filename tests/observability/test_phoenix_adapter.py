"""Tests for PhoenixAdapter class."""

from __future__ import annotations

import pytest

from eval_harness.observability.phoenix_adapter import (
    DEFAULT_ENDPOINT,
    DEFAULT_PROJECT_NAME,
    PhoenixAdapter,
)


class TestPhoenixAdapter:
    """Test PhoenixAdapter implementation."""

    def test_init_with_valid_endpoint(self):
        """Test PhoenixAdapter initialization with valid endpoint."""
        adapter = PhoenixAdapter(
            endpoint="http://localhost:6006",
            project_name="test-project",
            enabled=True,
        )

        assert adapter._endpoint == "http://localhost:6006"
        assert adapter._project_name == "test-project"
        assert adapter._enabled is True

    def test_init_with_default_values(self):
        """Test PhoenixAdapter initialization with default values."""
        adapter = PhoenixAdapter()

        assert adapter._endpoint == DEFAULT_ENDPOINT
        assert adapter._project_name == DEFAULT_PROJECT_NAME
        assert adapter._enabled is True

    def test_init_with_invalid_endpoint(self):
        """Test that invalid endpoint URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Phoenix endpoint URL"):
            PhoenixAdapter(endpoint="invalid-url")

        with pytest.raises(ValueError, match="Invalid Phoenix endpoint URL"):
            PhoenixAdapter(endpoint="ftp://localhost:6006")

    def test_init_with_https_endpoint(self):
        """Test that https:// endpoints are accepted."""
        adapter = PhoenixAdapter(endpoint="https://phoenix.example.com")

        assert adapter._endpoint == "https://phoenix.example.com"

    def test_is_connected_returns_bool(self):
        """Test is_connected returns a boolean."""
        adapter = PhoenixAdapter(enabled=False)

        # When disabled, should return False
        result = adapter.is_connected()
        assert isinstance(result, bool)

    def test_export_traces_in_buffering_mode(self):
        """Test export_traces in buffering mode returns parquet path."""
        adapter = PhoenixAdapter(enabled=False)

        result = adapter.export_traces()

        assert result["mode"] == "parquet"
        assert "path" in result
        assert "trace_count" in result
        assert isinstance(result["trace_count"], int)

    def test_start_rag_query_span_returns_trace_id(self):
        """Test that start_rag_query_span returns a valid trace_id."""
        adapter = PhoenixAdapter(enabled=False)

        trace_id = adapter.start_rag_query_span("What is the termination clause?")

        assert isinstance(trace_id, str)
        assert len(trace_id) > 0

    def test_span_methods_do_not_fail(self):
        """Test that span creation methods do not raise exceptions."""
        adapter = PhoenixAdapter(enabled=False)
        trace_id = adapter.start_rag_query_span("Test question")

        # These should not raise exceptions
        adapter.start_retrieval_span(
            trace_id=trace_id,
            embeddings=[0.1, 0.2, 0.3],
            chunks=[{"text": "chunk1"}],
            k=5,
            timing_ms=100.0,
        )

        adapter.start_generation_span(
            trace_id=trace_id,
            model="claude-3-opus",
            prompt="Test prompt",
            tokens=100,
            timing_ms=200.0,
        )

        adapter.start_evaluation_span(
            trace_id=trace_id,
            ragas_metrics={
                "faithfulness": 0.9,
                "context_precision": 0.8,
                "context_recall": 0.7,
                "answer_relevancy": 0.85,
            },
        )

    def test_slots_defined(self):
        """Test that PhoenixAdapter uses __slots__."""
        assert hasattr(PhoenixAdapter, "__slots__")
        assert "_endpoint" in PhoenixAdapter.__slots__
        assert "_project_name" in PhoenixAdapter.__slots__
        assert "_enabled" in PhoenixAdapter.__slots__
        assert "_export_path" in PhoenixAdapter.__slots__
        assert "_client" in PhoenixAdapter.__slots__
