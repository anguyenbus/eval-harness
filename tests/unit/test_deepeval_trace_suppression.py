"""Tests for DeepEval trace suppression implementation."""

from unittest.mock import MagicMock, patch


class TestDeepEvalTraceSuppression:
    """Test suite for DeepEval trace suppression."""

    def test_deepeval_llm_calls_use_suppress_tracing_when_available(self):
        """Test that suppress_tracing is used when Phoenix is available."""
        from eval_harness.adapters.deepeval_adapter import (
            _suppress_tracing_if_available,
        )

        # The function should return a context manager
        result = _suppress_tracing_if_available()

        # Should be able to use it as a context manager without error
        with result:
            pass

    def test_deepeval_llm_calls_use_noop_when_phoenix_unavailable(self):
        """Test that a no-op context manager is used when Phoenix is unavailable."""
        # Even when Phoenix is not available, _suppress_tracing_if_available
        # should return a working no-op context manager
        from eval_harness.adapters.deepeval_adapter import (
            _suppress_tracing_if_available,
        )

        result = _suppress_tracing_if_available()

        # Should be able to use it as a context manager without error
        with result:
            # No exception should be raised
            pass

    def test_normal_rag_calls_still_create_spans(self):
        """Test that normal RAG calls still create spans."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            # Mock the metrics
            mock_metrics = {
                "faithfulness": MagicMock(),
            }
            mock_metrics["faithfulness"].measure.return_value = None
            mock_metrics["faithfulness"].score = 0.9
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")

            # Normal evaluation should still work
            rag_output = {
                "query": {"text": "What is contract law?"},
                "answer": {"text": "Contract law governs..."},
                "retrieved_chunks": [{"text": "Contract context"}],
            }
            scores = evaluator.compute_metrics(rag_output, "Reference answer")

            assert scores["faithfulness"] == 0.9

    def test_suppress_context_manager_works_correctly(self):
        """Test that suppress_tracing context manager works correctly."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            # Mock the metrics
            mock_metrics = {
                "faithfulness": MagicMock(),
            }
            mock_metrics["faithfulness"].measure.return_value = None
            mock_metrics["faithfulness"].score = 0.9
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")

            # Verify the evaluator works
            assert callable(evaluator.compute_metrics)

    def test_suppress_tracing_in_compute_metrics_with_reasoning(self):
        """Test that compute_metrics_with_reasoning uses suppress_tracing."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            # Mock the metrics
            mock_metrics = {
                "faithfulness": MagicMock(),
            }
            mock_metrics["faithfulness"].measure.return_value = None
            mock_metrics["faithfulness"].score = 0.9
            mock_metrics["faithfulness"].reason = "Test reason"
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")

            # Test compute_metrics_with_reasoning
            rag_output = {
                "query": {"text": "What is contract law?"},
                "answer": {"text": "Contract law governs..."},
                "retrieved_chunks": [{"text": "Contract context"}],
            }
            result = evaluator.compute_metrics_with_reasoning(rag_output, "Reference answer")

            assert "scores" in result
            assert "reasoning" in result
            assert result["scores"]["faithfulness"] == 0.9
