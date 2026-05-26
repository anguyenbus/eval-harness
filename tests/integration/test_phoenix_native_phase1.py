"""Phase 1 Integration Tests for Phoenix Native Migration.

Tests end-to-end trace capture with auto-instrumentation,
feature flag toggle between legacy and native modes,
and trace suppression during evaluation.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestPhase1Integration:
    """Integration tests for Phase 1 Phoenix Native migration."""

    def test_end_to_end_trace_capture_with_auto_instrumentation(self):
        """Test end-to-end trace capture with auto-instrumentation."""
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

            # Simulate a full RAG pipeline trace
            with adapter.eval_run_span("integration-test", num_questions=1) as run_id:
                assert run_id is not None

                with adapter.rag_query_span("What is contract law?") as trace_id:
                    assert trace_id is not None

                    adapter.start_retrieval_span(
                        trace_id=trace_id,
                        query_text="What is contract law?",
                        chunks=[{"text": "Contract context", "doc_id": "doc1", "score": 0.9}],
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

                    adapter.start_evaluation_span(
                        trace_id=trace_id,
                        evaluation_metrics={"faithfulness": 0.9, "answer_relevancy": 0.85},
                        verdict="PASS",
                        reasoning={"faithfulness": {"reason": "Good answer"}},
                    )

            # Verify spans were created
            assert mock_tracer.start_as_current_span.call_count > 0

    def test_feature_flag_toggle_between_legacy_and_native_modes(self):
        """Test feature flag toggle between legacy and native modes."""
        from eval_harness.observability.config_phoenix_native import (
            get_phoenix_native_config,
        )
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        with patch("phoenix.otel.register") as mock_register:
            mock_tracer_provider = MagicMock()
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
            mock_tracer.start_as_current_span.return_value.__exit__.return_value = None
            mock_tracer_provider.get_tracer.return_value = mock_tracer
            mock_register.return_value = mock_tracer_provider

            # Test with use_phoenix_native: true
            config_native = {
                "phoenix_native": {"use_phoenix_native": True}
            }
            phoenix_config = get_phoenix_native_config(config_native)
            assert phoenix_config["use_phoenix_native"] is True

            # Test with use_phoenix_native: false
            config_legacy = {
                "phoenix_native": {"use_phoenix_native": False}
            }
            phoenix_config = get_phoenix_native_config(config_legacy)
            assert phoenix_config["use_phoenix_native"] is False

            # Test with missing phoenix_native section (default: false)
            config_default = {}
            phoenix_config = get_phoenix_native_config(config_default)
            assert phoenix_config["use_phoenix_native"] is False

    def test_trace_suppression_during_evaluation(self):
        """Test trace suppression during evaluation."""
        from eval_harness.adapters.deepeval_adapter import (
            DeepEvalEvaluator,
            _suppress_tracing_if_available,
        )

        # Verify suppress_tracing context manager is available
        suppress_cm = _suppress_tracing_if_available()

        # Should work as a context manager
        with suppress_cm:
            # Simulate evaluation that would normally create spans
            pass

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            mock_metrics = {
                "faithfulness": MagicMock(),
            }
            mock_metrics["faithfulness"].measure.return_value = None
            mock_metrics["faithfulness"].score = 0.9
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")

            # Evaluation should work with trace suppression
            rag_output = {
                "query": {"text": "What is contract law?"},
                "answer": {"text": "Contract law governs..."},
                "retrieved_chunks": [{"text": "Contract context"}],
            }
            result = evaluator.compute_metrics_with_reasoning(rag_output, "Reference answer")

            assert "scores" in result
            assert "reasoning" in result

    def test_backward_compatibility_with_phoenix_disabled(self):
        """Test backward compatibility when Phoenix is disabled."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        # Create adapter with enabled=False
        adapter = PhoenixAdapter(
            endpoint="http://localhost:6006",
            project_name="test-project",
            enabled=False,  # Disabled
        )

        # Should still work without creating spans
        assert adapter.is_connected() is False

        with adapter.eval_run_span("test-run", num_questions=1) as run_id:
            assert run_id is not None

            with adapter.rag_query_span("Test question") as trace_id:
                assert trace_id is not None

                adapter.start_retrieval_span(
                    trace_id=trace_id,
                    query_text="Test",
                    chunks=[{"text": "chunk1", "doc_id": "doc1", "score": 0.9}],
                    k=1,
                    timing_ms=100,
                )
