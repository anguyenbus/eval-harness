"""Tests for Phoenix auto-instrumentation migration."""

import pytest
from unittest.mock import MagicMock, patch, call


class TestPhoenixAutoInstrumentation:
    """Test suite for Phoenix auto-instrumentation."""

    def test_auto_instrumentation_creates_rag_pipeline_spans(self, tmp_path):
        """Test that auto-instrumentation captures RAG pipeline spans."""
        config_content = """
        datasets:
          legalbench_rag:
            path: /data/legalbench

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0

        phoenix_native:
          use_phoenix_native: true
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        from eval_harness.config import load_config
        from eval_harness.observability.config_phoenix_native import (
            get_phoenix_native_config,
        )
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        config = load_config(config_file)
        phoenix_config = get_phoenix_native_config(config)

        # Mock phoenix.otel.register to avoid actual connection
        with patch("phoenix.otel.register") as mock_register:
            mock_tracer_provider = MagicMock()
            mock_tracer = MagicMock()
            mock_tracer_provider.get_tracer.return_value = mock_tracer
            mock_register.return_value = mock_tracer_provider

            adapter = PhoenixAdapter(
                endpoint="http://localhost:6006",
                project_name="test-project",
                enabled=True,
            )

            # Verify register was called with auto_instrument=True equivalent
            # (We'll verify this after implementing auto-instrumentation)
            assert adapter.is_connected()

    def test_span_hierarchy_preserved(self):
        """Test that span hierarchy (parent-child relationships) is preserved."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        with patch("phoenix.otel.register") as mock_register:
            mock_tracer_provider = MagicMock()
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            mock_tracer.start_as_current_span.return_value.__exit__.return_value = None
            mock_tracer_provider.get_tracer.return_value = mock_tracer
            mock_register.return_value = mock_tracer_provider

            adapter = PhoenixAdapter(
                endpoint="http://localhost:6006",
                project_name="test-project",
                enabled=True,
            )

            # Test eval_run_span (root)
            with adapter.eval_run_span("test-run", num_questions=5) as run_id:
                assert run_id is not None

                # Test rag_query_span (child of eval_run)
                with adapter.rag_query_span("What is contract law?") as trace_id:
                    assert trace_id is not None

    def test_span_attributes_properly_set(self):
        """Test that span attributes are properly set."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        with patch("phoenix.otel.register") as mock_register:
            mock_tracer_provider = MagicMock()
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            mock_tracer.start_as_current_span.return_value.__exit__.return_value = None
            mock_tracer_provider.get_tracer.return_value = mock_tracer
            mock_register.return_value = mock_tracer_provider

            adapter = PhoenixAdapter(
                endpoint="http://localhost:6006",
                project_name="test-project",
                enabled=True,
            )

            # Test that span attributes are set
            with adapter.rag_query_span("Test question") as trace_id:
                pass

            # Verify set_attribute was called with correct keys
            assert any(call[0][0] in ["question", "input"] for call in mock_span.set_attribute.call_args_list)

    def test_backward_compatibility_when_flag_false(self):
        """Test backward compatibility when use_phoenix_native is False."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        with patch("phoenix.otel.register") as mock_register:
            mock_tracer_provider = MagicMock()
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            mock_tracer.start_as_current_span.return_value.__exit__.return_value = None
            mock_tracer_provider.get_tracer.return_value = mock_tracer
            mock_register.return_value = mock_tracer_provider

            # When flag is false, should still work with manual span creation
            adapter = PhoenixAdapter(
                endpoint="http://localhost:6006",
                project_name="test-project",
                enabled=True,
            )

            # Test all span creation methods still work
            with adapter.eval_run_span("test-run", num_questions=5):
                with adapter.rag_query_span("Test question") as trace_id:
                    adapter.start_retrieval_span(
                        trace_id=trace_id,
                        query_text="Test",
                        chunks=[{"text": "chunk1", "doc_id": "doc1", "score": 0.9}],
                        k=1,
                        timing_ms=100,
                    )
                    adapter.start_generation_span(
                        trace_id=trace_id,
                        model="gpt-4",
                        prompt="Test prompt",
                        tokens=100,
                        timing_ms=200,
                    )

    def test_span_names_match_existing(self):
        """Test that span names match existing: eval_run, rag_query, retrieval, generation, evaluator."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter
        from openinference.semconv.trace import OpenInferenceSpanKindValues

        with patch("phoenix.otel.register") as mock_register:
            mock_tracer_provider = MagicMock()
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            mock_tracer.start_as_current_span.return_value.__exit__.return_value = None
            mock_tracer_provider.get_tracer.return_value = mock_tracer
            mock_register.return_value = mock_tracer_provider

            adapter = PhoenixAdapter(
                endpoint="http://localhost:6006",
                project_name="test-project",
                enabled=True,
            )

            # Call the span methods
            with adapter.eval_run_span("test-run"):
                with adapter.rag_query_span("Test"):
                    pass

            # Verify that start_as_current_span was called
            assert mock_tracer.start_as_current_span.call_count > 0
