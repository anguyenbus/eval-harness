"""
Tests for evaluator comparison between DeepEval and Phoenix.

PHOENIX NATIVE MIGRATION: Phase 3.3 - Evaluator Comparison Tests
Tests to verify Phoenix and DeepEval produce equivalent results on test data.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestEvaluatorComparisonInterface:
    """Tests for interface compatibility between DeepEval and Phoenix evaluators."""

    def test_deepeval_interface(self) -> None:
        """Test DeepEval adapter has expected interface."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            mock_metrics = {
                "faithfulness": MagicMock(),
                "context_precision": MagicMock(),
                "context_recall": MagicMock(),
                "answer_relevancy": MagicMock(),
            }
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator()

            # Check interface exists
            assert hasattr(evaluator, "compute_metrics")
            assert hasattr(evaluator, "async_batch_compute_metrics")
            assert hasattr(evaluator, "compute_metrics_with_reasoning")

    def test_phoenix_interface(self) -> None:
        """Test Phoenix adapter has expected interface."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        with patch("phoenix.evals.LLM"):
            adapter = PhoenixEvalAdapter()

            # Check interface exists
            assert hasattr(adapter, "compute_metrics")
            assert hasattr(adapter, "batch_compute_metrics")
            assert hasattr(adapter, "compute_metrics_with_reasoning")

    def test_interfaces_compatible(self) -> None:
        """Test both adapters have compatible signatures."""
        import inspect

        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            mock_metrics = {
                "faithfulness": MagicMock(),
                "context_precision": MagicMock(),
                "context_recall": MagicMock(),
                "answer_relevancy": MagicMock(),
            }
            mock_create.return_value = mock_metrics

            with patch("phoenix.evals.LLM"):
                deepeval = DeepEvalEvaluator()
                phoenix = PhoenixEvalAdapter()

                # Compare compute_metrics signature
                deepeval_sig = inspect.signature(deepeval.compute_metrics)
                phoenix_sig = inspect.signature(phoenix.compute_metrics)

                # Both should accept rag_output and reference_answer
                deepeval_params = list(deepeval_sig.parameters.keys())
                phoenix_params = list(phoenix_sig.parameters.keys())

                assert "rag_output" in deepeval_params
                assert "rag_output" in phoenix_params
                assert "reference_answer" in deepeval_params
                assert "reference_answer" in phoenix_params


class TestEvaluatorOutputFormat:
    """Tests for output format compatibility."""

    def test_deepeval_output_format(self) -> None:
        """Test DeepEval output format."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        rag_output = {
            "query": {"text": "Test question"},
            "answer": {"text": "Test answer"},
            "retrieved_chunks": [{"text": "Test context"}],
        }
        reference_answer = "Test reference"

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            # Mock metrics that return valid scores
            mock_faithfulness = MagicMock()
            mock_faithfulness.score = 0.8

            mock_precision = MagicMock()
            mock_precision.score = 0.75

            mock_recall = MagicMock()
            mock_recall.score = 0.85

            mock_relevancy = MagicMock()
            mock_relevancy.score = 0.9

            mock_metrics = {
                "faithfulness": mock_faithfulness,
                "context_precision": mock_precision,
                "context_recall": mock_recall,
                "answer_relevancy": mock_relevancy,
            }
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator()

            with patch("eval_harness.adapters.deepeval_adapter._suppress_tracing_if_available"):
                result = evaluator.compute_metrics(rag_output, reference_answer)

            # Check output format
            assert isinstance(result, dict)
            assert "faithfulness" in result
            assert isinstance(result["faithfulness"], (int, float))

    def test_phoenix_output_format(self) -> None:
        """Test Phoenix output format matches DeepEval."""
        import pandas as pd

        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "Test question"},
            "answer": {"text": "Test answer"},
            "retrieved_chunks": [{"text": "Test context"}],
        }
        reference_answer = "Test reference"

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            mock_result = pd.DataFrame({
                "faithfulness_score": [0.8],
                "correctness_score": [0.9],
                "relevance_score": [0.75],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            result = adapter.compute_metrics(rag_output, reference_answer)

            # Check output format
            assert isinstance(result, dict)
            assert "faithfulness" in result
            assert isinstance(result["faithfulness"], (int, float))


class TestEvaluatorErrorHandling:
    """Tests for error handling compatibility."""

    def test_deepeval_error_handling(self) -> None:
        """Test DeepEval handles errors gracefully."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        rag_output = {
            "query": {"text": "Test"},
            "answer": {"text": "Test"},
            "retrieved_chunks": [],
        }
        reference_answer = "Test"

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            # Mock metric that raises error
            mock_metric = MagicMock()
            mock_metric.measure.side_effect = Exception("Evaluation failed")

            mock_metrics = {
                "faithfulness": mock_metric,
                "context_precision": mock_metric,
                "context_recall": mock_metric,
                "answer_relevancy": mock_metric,
            }
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator()

            with patch("eval_harness.adapters.deepeval_adapter._suppress_tracing_if_available"):
                result = evaluator.compute_metrics(rag_output, reference_answer)

            # Should still return a result with default scores
            assert isinstance(result, dict)
            assert "faithfulness" in result
            assert result["faithfulness"] == 0.0

    def test_phoenix_error_handling(self) -> None:
        """Test Phoenix handles errors gracefully."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "Test"},
            "answer": {"text": "Test"},
            "retrieved_chunks": [],
        }
        reference_answer = "Test"

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            mock_evaluate.side_effect = Exception("Evaluation failed")

            adapter = PhoenixEvalAdapter()

            result = adapter.compute_metrics(rag_output, reference_answer)

            # Should return default scores
            assert isinstance(result, dict)
            assert "faithfulness" in result
            assert result["faithfulness"] == 0.0


class TestEvaluatorReasoningExtraction:
    """Tests for reasoning extraction compatibility."""

    def test_deepeval_reasoning_format(self) -> None:
        """Test DeepEval reasoning format."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        rag_output = {
            "query": {"text": "Test question"},
            "answer": {"text": "Test answer"},
            "retrieved_chunks": [{"text": "Test context"}],
        }
        reference_answer = "Test reference"

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            # Mock metrics with reasoning
            mock_faithfulness = MagicMock()
            mock_faithfulness.score = 0.8
            mock_faithfulness.reason = "The answer is supported"
            mock_faithfulness.verdicts = []

            mock_metrics = {
                "faithfulness": mock_faithfulness,
                "context_precision": MagicMock(),
                "context_recall": MagicMock(),
                "answer_relevancy": MagicMock(),
            }
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator()

            with patch("eval_harness.adapters.deepeval_adapter._suppress_tracing_if_available"):
                result = evaluator.compute_metrics_with_reasoning(rag_output, reference_answer)

            # Check reasoning format
            assert "scores" in result
            assert "reasoning" in result
            assert "faithfulness" in result["scores"]
            assert "faithfulness" in result["reasoning"]

    def test_phoenix_reasoning_format(self) -> None:
        """Test Phoenix reasoning format matches DeepEval."""
        import pandas as pd

        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "Test question"},
            "answer": {"text": "Test answer"},
            "retrieved_chunks": [{"text": "Test context"}],
        }
        reference_answer = "Test reference"

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            mock_result = pd.DataFrame({
                "faithfulness_score": [0.8],
                "faithfulness_reason": ["The answer is supported"],
                "correctness_score": [0.9],
                "relevance_score": [0.75],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            result = adapter.compute_metrics_with_reasoning(rag_output, reference_answer)

            # Check reasoning format
            assert "scores" in result
            assert "reasoning" in result
            assert "faithfulness" in result["scores"]
            assert "faithfulness" in result["reasoning"]
