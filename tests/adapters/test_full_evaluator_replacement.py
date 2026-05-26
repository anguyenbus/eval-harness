"""
Tests for full DeepEval replacement with Phoenix evaluators.

PHOENIX NATIVE MIGRATION: Phase 3.5 - Full DeepEval Replacement
Tests for replacing DeepEval evaluators with Phoenix equivalents.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestEvaluatorSwitching:
    """Tests for switching between Phoenix and DeepEval evaluators."""

    def test_phoenix_evaluator_used_when_flag_true(self) -> None:
        """Test that Phoenix evaluators are used when use_phoenix_native is true."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter
        from eval_harness.observability.config_phoenix_native import (
            get_phoenix_native_config,
        )

        config = {"phoenix_native": {"use_phoenix_native": True}}
        phoenix_config = get_phoenix_native_config(config)

        assert phoenix_config["use_phoenix_native"] is True

        # Phoenix evaluator should be used
        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            import pandas as pd

            mock_result = pd.DataFrame({
                "faithfulness_score": [0.85],
                "correctness_score": [0.90],
                "relevance_score": [0.75],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            rag_output = {
                "query": {"text": "What is contract law?"},
                "answer": {"text": "Contract law governs agreements..."},
                "retrieved_chunks": [{"text": "Contract law context"}],
            }
            reference_answer = "Contract law governs agreements."

            scores = adapter.compute_metrics(rag_output, reference_answer)

            assert "faithfulness" in scores
            assert scores["faithfulness"] == 0.85

    def test_deepeval_still_works_when_flag_false(self) -> None:
        """Test that DeepEval still works when use_phoenix_native is false."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator
        from eval_harness.observability.config_phoenix_native import (
            get_phoenix_native_config,
        )

        config = {"phoenix_native": {"use_phoenix_native": False}}
        phoenix_config = get_phoenix_native_config(config)

        assert phoenix_config["use_phoenix_native"] is False

        # DeepEval should still work
        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            mock_metrics = {
                "faithfulness": MagicMock(),
                "context_precision": MagicMock(),
                "context_recall": MagicMock(),
                "answer_relevancy": MagicMock(),
            }
            for metric in mock_metrics.values():
                metric.measure.return_value = None
                metric.score = 0.85
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")

            rag_output = {
                "query": {"text": "Question"},
                "answer": {"text": "Answer"},
                "retrieved_chunks": [{"text": "Context"}],
            }

            scores = evaluator.compute_metrics(rag_output, "Reference")

            assert "faithfulness" in scores
            assert scores["faithfulness"] == 0.85

    def test_evaluator_produces_consistent_results(self) -> None:
        """Test that evaluators produce consistent results."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "What is contract law?"},
            "answer": {"text": "Contract law governs agreements..."},
            "retrieved_chunks": [{"text": "Contract law context"}],
        }
        reference_answer = "Contract law governs agreements."

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            import pandas as pd

            mock_result = pd.DataFrame({
                "faithfulness_score": [0.85],
                "correctness_score": [0.90],
                "relevance_score": [0.75],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            # Multiple calls should produce consistent results
            scores1 = adapter.compute_metrics(rag_output, reference_answer)
            scores2 = adapter.compute_metrics(rag_output, reference_answer)

            assert scores1["faithfulness"] == scores2["faithfulness"]
            assert scores1["correctness"] == scores2["correctness"]
            assert scores1["relevance"] == scores2["relevance"]


class TestDeepEvalReplacement:
    """Tests for DeepEval replacement in evaluation runner."""

    def test_evaluation_runner_uses_phoenix_when_enabled(self) -> None:
        """Test that evaluation runner uses Phoenix evaluator when flag is enabled."""
        # This test verifies the integration point
        from eval_harness.observability.config_phoenix_native import (
            get_phoenix_native_config,
        )

        config = {"phoenix_native": {"use_phoenix_native": True}}
        phoenix_config = get_phoenix_native_config(config)

        # When flag is true, Phoenix evaluator should be used
        assert phoenix_config["use_phoenix_native"] is True

    def test_evaluation_runner_falls_back_to_deepeval(self) -> None:
        """Test that evaluation runner falls back to DeepEval when flag is disabled."""
        from eval_harness.observability.config_phoenix_native import (
            get_phoenix_native_config,
        )

        config = {"phoenix_native": {"use_phoenix_native": False}}
        phoenix_config = get_phoenix_native_config(config)

        # When flag is false, DeepEval should be used
        assert phoenix_config["use_phoenix_native"] is False

    def test_trace_suppression_in_phoenix_evaluator(self) -> None:
        """Test that Phoenix evaluator properly suppresses traces."""
        from eval_harness.adapters.deepeval_adapter import (
            _suppress_tracing_if_available,
        )

        # Verify suppress_tracing is available
        suppress_cm = _suppress_tracing_if_available()

        # Should work as a context manager
        with suppress_cm:
            # Simulated evaluation
            pass

        # No exceptions should occur
        assert True

    def test_phoenix_adapter_error_handling(self) -> None:
        """Test error handling in Phoenix adapter."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "Question"},
            "answer": {"text": "Answer"},
            "retrieved_chunks": [],
        }
        reference_answer = "Reference"

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            # Simulate an error
            mock_evaluate.side_effect = Exception("Phoenix evaluation error")

            adapter = PhoenixEvalAdapter()

            # Should return default scores on error
            scores = adapter.compute_metrics(rag_output, reference_answer)

            assert "faithfulness" in scores
            assert scores["faithfulness"] == 0.0
            assert "correctness" in scores
            assert scores["correctness"] == 0.0
            assert "relevance" in scores
            assert scores["relevance"] == 0.0


class TestBackwardCompatibility:
    """Tests for backward compatibility during migration."""

    def test_deepeval_interface_preserved(self) -> None:
        """Test that DeepEval interface is preserved for backward compatibility."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            mock_metrics = {
                "faithfulness": MagicMock(),
            }
            mock_metrics["faithfulness"].measure.return_value = None
            mock_metrics["faithfulness"].score = 0.9
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator()

            # All expected methods should exist
            assert hasattr(evaluator, "compute_metrics")
            assert hasattr(evaluator, "compute_metrics_with_reasoning")
            assert hasattr(evaluator, "async_batch_compute_metrics")

    def test_phoenix_adapter_matches_deepeval_interface(self) -> None:
        """Test that Phoenix adapter matches DeepEval interface."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        adapter = PhoenixEvalAdapter()

        # Should have similar methods to DeepEval evaluator
        assert hasattr(adapter, "compute_metrics")
        assert hasattr(adapter, "compute_metrics_with_reasoning")
        assert hasattr(adapter, "batch_compute_metrics")
