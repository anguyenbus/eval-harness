"""
Tests for DeepEval metrics wrapped as Phoenix evaluators.
"""

from unittest.mock import MagicMock, patch

import pytest

from eval_harness.experiments.deepeval_evaluators import (
    create_answer_relevancy_evaluator,
    create_context_precision_evaluator,
    create_context_recall_evaluator,
    create_faithfulness_evaluator,
)


def test_create_faithfulness_evaluator_returns_callable():
    """Test that faithfulness evaluator factory returns a callable."""
    evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")
    assert callable(evaluator)
    # Should have name from decorator
    assert hasattr(evaluator, "name") or "faithfulness" in str(evaluator)


def test_create_context_precision_evaluator_returns_callable():
    """Test that context precision evaluator factory returns a callable."""
    evaluator = create_context_precision_evaluator()
    assert callable(evaluator)


def test_create_context_recall_evaluator_returns_callable():
    """Test that context recall evaluator factory returns a callable."""
    evaluator = create_context_recall_evaluator()
    assert callable(evaluator)


def test_create_answer_relevancy_evaluator_returns_callable():
    """Test that answer relevancy evaluator factory returns a callable."""
    evaluator = create_answer_relevancy_evaluator()
    assert callable(evaluator)


def test_faithfulness_evaluator_returns_dict_with_score():
    """Test faithfulness evaluator returns dict with score, label, explanation."""
    with (
        patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class,
        patch("deepeval.test_case.LLMTestCase") as mock_test_case_class,
    ):
        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.success = True
        mock_metric.reason = "Good answer."
        mock_metric.threshold = 0.5
        mock_metric.evaluation_model = "gpt-4o-mini"
        mock_metric_class.return_value = mock_metric

        mock_test_case = MagicMock()
        mock_test_case_class.return_value = mock_test_case

        evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

        result = evaluator(
            input="Test question?",
            output={
                "answer": "Test answer.",
                "retrieval_context": ["Context 1"],
            },
        )

        assert isinstance(result, dict)
        assert result["score"] == 0.85
        assert result["label"] == "faithful"
        assert "explanation" in result
        assert "metadata" in result


def test_context_precision_evaluator_returns_dict_with_score():
    """Test context precision evaluator returns dict with score, label, explanation."""
    with (
        patch("deepeval.metrics.ContextualPrecisionMetric") as mock_metric_class,
        patch("deepeval.test_case.LLMTestCase") as mock_test_case_class,
    ):
        mock_metric = MagicMock()
        mock_metric.score = 0.75
        mock_metric.success = True
        mock_metric.reason = "Good precision."
        mock_metric.threshold = 0.5
        mock_metric_class.return_value = mock_metric

        mock_test_case = MagicMock()
        mock_test_case_class.return_value = mock_test_case

        evaluator = create_context_precision_evaluator(judge_model="gpt-4o-mini")

        result = evaluator(
            input="Test question?",
            output={
                "answer": "Test answer.",
                "retrieval_context": ["Context 1", "Context 2"],
            },
            expected="Expected answer.",
        )

        assert isinstance(result, dict)
        assert result["score"] == 0.75
        assert result["label"] == "precise"
        assert "explanation" in result


def test_answer_relevancy_evaluator_returns_dict_with_score():
    """Test answer relevancy evaluator returns dict with score, label, explanation."""
    with (
        patch("deepeval.metrics.AnswerRelevancyMetric") as mock_metric_class,
        patch("deepeval.test_case.LLMTestCase") as mock_test_case_class,
        patch("eval_harness.adapters.embeddings.get_embedder") as mock_embedder,
    ):
        mock_embedder.return_value = MagicMock()

        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.success = True
        mock_metric.reason = "Relevant answer."
        mock_metric.threshold = 0.5
        mock_metric_class.return_value = mock_metric

        mock_test_case = MagicMock()
        mock_test_case_class.return_value = mock_test_case

        evaluator = create_answer_relevancy_evaluator(judge_model="gpt-4o-mini")

        result = evaluator(
            input="Test question?",
            output={"answer": "Test answer."},
        )

        assert isinstance(result, dict)
        assert result["score"] == 0.9
        assert result["label"] == "relevant"
        assert "explanation" in result
