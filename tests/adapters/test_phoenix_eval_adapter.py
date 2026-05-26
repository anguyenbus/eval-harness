"""
Tests for Phoenix evaluator adapter.

PHOENIX NATIVE MIGRATION: Phase 3.2 - Phoenix Evaluator Adapter
Tests for PhoenixEvalAdapter wrapping Phoenix evaluators with existing interface.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd


class TestPhoenixEvalAdapterInitialization:
    """Tests for PhoenixEvalAdapter initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        with patch("phoenix.evals.LLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            adapter = PhoenixEvalAdapter()

            assert adapter._llm_provider == "openai"
            assert adapter._judge_model == "gpt-4o-mini"
            assert adapter._temperature == 0.0

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        with patch("phoenix.evals.LLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            adapter = PhoenixEvalAdapter(
                llm_provider="openai",
                judge_model="gpt-4o",
                temperature=0.1,
            )

            assert adapter._judge_model == "gpt-4o"
            assert adapter._temperature == 0.1

    def test_init_creates_evaluators(self) -> None:
        """Test that initialization creates required evaluators."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        with patch("phoenix.evals.LLM") as mock_llm_class:
            with patch("phoenix.evals.metrics.FaithfulnessEvaluator") as mock_faith_class:
                with patch("phoenix.evals.metrics.CorrectnessEvaluator") as mock_correct_class:
                    with patch("phoenix.evals.metrics.DocumentRelevanceEvaluator") as mock_rel_class:
                        mock_llm = MagicMock()
                        mock_llm_class.return_value = mock_llm

                        mock_faith = MagicMock()
                        mock_faith_class.return_value = mock_faith
                        mock_correct = MagicMock()
                        mock_correct_class.return_value = mock_correct
                        mock_rel = MagicMock()
                        mock_rel_class.return_value = mock_rel

                        adapter = PhoenixEvalAdapter()

                        # Check that evaluators were created
                        assert adapter._faithfulness_evaluator is not None
                        assert adapter._correctness_evaluator is not None
                        assert adapter._relevance_evaluator is not None


class TestPhoenixEvalAdapterFaithfulness:
    """Tests for faithfulness evaluation."""

    def test_compute_faithfulness(self) -> None:
        """Test computing faithfulness score."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "What is contract law?"},
            "answer": {"text": "Contract law governs agreements..."},
            "retrieved_chunks": [{"text": "Contract law context"}],
        }
        reference_answer = "Contract law governs agreements between parties."

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            # Mock the evaluate_dataframe response
            mock_result = pd.DataFrame({
                "faithfulness_score": [0.85],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            scores = adapter.compute_metrics(rag_output, reference_answer)

            assert "faithfulness" in scores
            assert scores["faithfulness"] == 0.85

    def test_compute_faithfulness_error_handling(self) -> None:
        """Test faithfulness evaluation with errors."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "Question"},
            "answer": {"text": "Answer"},
            "retrieved_chunks": [],
        }
        reference_answer = "Reference"

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            # Simulate an error
            mock_evaluate.side_effect = Exception("Phoenix error")

            adapter = PhoenixEvalAdapter()

            # Should return default scores on error
            scores = adapter.compute_metrics(rag_output, reference_answer)

            # Should return default scores
            assert "faithfulness" in scores
            assert scores["faithfulness"] == 0.0


class TestPhoenixEvalAdapterCorrectness:
    """Tests for correctness evaluation."""

    def test_compute_correctness(self) -> None:
        """Test computing correctness score."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "What is tort law?"},
            "answer": {"text": "Tort law deals with civil wrongs..."},
            "retrieved_chunks": [{"text": "Tort law context"}],
        }
        reference_answer = "Tort law addresses civil wrongs and remedies."

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            mock_result = pd.DataFrame({
                "correctness_score": [0.90],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            scores = adapter.compute_metrics(rag_output, reference_answer)

            assert "correctness" in scores
            assert scores["correctness"] == 0.90


class TestPhoenixEvalAdapterRelevance:
    """Tests for document relevance evaluation."""

    def test_compute_relevance(self) -> None:
        """Test computing document relevance score."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "What is the termination clause?"},
            "answer": {"text": "The contract can be terminated with 30 days notice..."},
            "retrieved_chunks": [
                {"text": "Termination requires 30 days notice"},
                {"text": "Irrelevant context"},
            ],
        }
        reference_answer = "Termination requires 30 days written notice."

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            mock_result = pd.DataFrame({
                "relevance_score": [0.75],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            scores = adapter.compute_metrics(rag_output, reference_answer)

            assert "relevance" in scores
            assert scores["relevance"] == 0.75


class TestPhoenixEvalAdapterBatchEvaluation:
    """Tests for batch evaluation."""

    def test_batch_compute_metrics(self) -> None:
        """Test computing metrics for multiple samples."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_outputs = [
            {
                "query": {"text": "Question 1"},
                "answer": {"text": "Answer 1"},
                "retrieved_chunks": [{"text": "Context 1"}],
            },
            {
                "query": {"text": "Question 2"},
                "answer": {"text": "Answer 2"},
                "retrieved_chunks": [{"text": "Context 2"}],
            },
        ]
        reference_answers = ["Reference 1", "Reference 2"]

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            mock_result = pd.DataFrame({
                "faithfulness_score": [0.8, 0.9],
                "correctness_score": [0.85, 0.95],
                "relevance_score": [0.75, 0.88],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            scores_list = adapter.batch_compute_metrics(rag_outputs, reference_answers)

            assert len(scores_list) == 2
            assert "faithfulness" in scores_list[0]
            assert "correctness" in scores_list[0]
            assert "relevance" in scores_list[0]


class TestPhoenixEvalAdapterReasoning:
    """Tests for reasoning extraction."""

    def test_compute_metrics_with_reasoning(self) -> None:
        """Test computing metrics with full reasoning extraction."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "What is contract law?"},
            "answer": {"text": "Contract law governs agreements..."},
            "retrieved_chunks": [{"text": "Contract law context"}],
        }
        reference_answer = "Contract law governs agreements."

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            # Mock results with reasoning columns
            mock_result = pd.DataFrame({
                "faithfulness_score": [0.85],
                "faithfulness_reason": ["The answer is fully supported by context"],
                "correctness_score": [0.90],
                "correctness_reason": ["Answer matches reference"],
                "relevance_score": [0.75],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            result = adapter.compute_metrics_with_reasoning(rag_output, reference_answer)

            assert "scores" in result
            assert "reasoning" in result
            assert "faithfulness" in result["scores"]
            assert result["scores"]["faithfulness"] == 0.85
            assert "faithfulness" in result["reasoning"]
