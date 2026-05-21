"""Tests for DeepEval adapter module."""

from unittest.mock import MagicMock, patch

import pytest

from eval_harness.adapters.deepeval_adapter import (
    DeepEvalEvaluator,
    transform_to_deepeval_sample,
)


class TestTransformToDeepEvalSample:
    """Test suite for transform_to_deepeval_sample function."""

    def test_transform_question_to_input(self):
        """Test that question is mapped to input field."""
        rag_output = {
            "query": {"text": "What is the termination clause?"},
            "answer": {"text": "The contract can be terminated...", "citations": []},
            "retrieved_chunks": [],
        }
        reference_answer = "The contract allows termination with 30 days notice."

        sample = transform_to_deepeval_sample(rag_output, reference_answer)

        assert sample.input == "What is the termination clause?"

    def test_transform_retrieved_chunks_to_retrieval_context(self):
        """Test that retrieved_chunks are mapped to retrieval_context."""
        rag_output = {
            "query": {"text": "Question?"},
            "answer": {"text": "Answer", "citations": []},
            "retrieved_chunks": [
                {"text": "Context 1"},
                {"text": "Context 2"},
                {"text": "Context 3"},
            ],
        }
        reference_answer = "Reference answer"

        sample = transform_to_deepeval_sample(rag_output, reference_answer)

        assert len(sample.retrieval_context) == 3
        assert sample.retrieval_context[0] == "Context 1"
        assert sample.retrieval_context[1] == "Context 2"
        assert sample.retrieval_context[2] == "Context 3"

    def test_transform_answer_to_actual_output(self):
        """Test that answer text is mapped to actual_output."""
        rag_output = {
            "query": {"text": "Question?"},
            "answer": {"text": "This is the generated answer.", "citations": []},
            "retrieved_chunks": [],
        }
        reference_answer = "Reference"

        sample = transform_to_deepeval_sample(rag_output, reference_answer)

        assert sample.actual_output == "This is the generated answer."

    def test_transform_reference_to_expected_output(self):
        """Test that reference answer is mapped to expected_output."""
        rag_output = {
            "query": {"text": "Question?"},
            "answer": {"text": "Answer", "citations": []},
            "retrieved_chunks": [],
        }
        reference_answer = "This is the reference answer."

        sample = transform_to_deepeval_sample(rag_output, reference_answer)

        assert sample.expected_output == "This is the reference answer."


class TestDeepEvalEvaluator:
    """Test suite for DeepEvalEvaluator class."""

    def test_initialization_creates_all_metrics(self, monkeypatch):
        """Test that DeepEvalEvaluator initializes with all 4 metric instances."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        assert evaluator is not None
        assert evaluator._metrics is not None
        assert "faithfulness" in evaluator._metrics
        assert "context_precision" in evaluator._metrics
        assert "context_recall" in evaluator._metrics
        assert "answer_relevancy" in evaluator._metrics

    def test_initialization_with_custom_max_concurrent(self, monkeypatch):
        """Test that max_concurrent parameter is stored correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        evaluator = DeepEvalEvaluator(
            llm_provider="openai", judge_model="gpt-4o", max_concurrent=20
        )

        assert evaluator._max_concurrent == 20

    def test_initialization_with_embedder(self, monkeypatch):
        """Test that embedder can be passed to evaluator."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_embedder = MagicMock()

        evaluator = DeepEvalEvaluator(
            llm_provider="openai", judge_model="gpt-4o", embedder=mock_embedder
        )

        assert evaluator._embedder is mock_embedder

    def test_compute_metrics_returns_structure(self, monkeypatch):
        """Test that compute_metrics returns expected dictionary structure."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        rag_output = {
            "query": {"text": "Question?"},
            "answer": {"text": "Answer", "citations": []},
            "retrieved_chunks": [{"text": "Context"}],
        }
        reference_answer = "Reference"

        # Mock the metric.measure() method to avoid actual API calls
        with patch.object(evaluator._metrics["faithfulness"], "measure"):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(
                        evaluator._metrics["answer_relevancy"], "measure"
                    ):
                        # Set mock scores directly on the metrics
                        evaluator._metrics["faithfulness"].score = 0.95
                        evaluator._metrics["context_precision"].score = 0.85
                        evaluator._metrics["context_recall"].score = 0.90
                        evaluator._metrics["answer_relevancy"].score = 0.88

                        results = evaluator.compute_metrics(
                            rag_output, reference_answer
                        )

                        assert "faithfulness" in results
                        assert "context_precision" in results
                        assert "context_recall" in results
                        assert "answer_relevancy" in results

    def test_compute_metrics_handles_error_gracefully(self, monkeypatch):
        """Test that compute_metrics returns 0.0 on metric failure."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        rag_output = {
            "query": {"text": "Question?"},
            "answer": {"text": "Answer", "citations": []},
            "retrieved_chunks": [{"text": "Context"}],
        }
        reference_answer = "Reference"

        # Mock measure() to raise exception
        with patch.object(
            evaluator._metrics["faithfulness"],
            "measure",
            side_effect=Exception("Test error"),
        ):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(
                        evaluator._metrics["answer_relevancy"], "measure"
                    ):
                        results = evaluator.compute_metrics(
                            rag_output, reference_answer
                        )

                        # Should return 0.0 for failed metric
                        assert results["faithfulness"] == 0.0

    def test_compute_metrics_with_timing(self, monkeypatch):
        """Test that compute_metrics_with_timing includes timing info."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        rag_output = {
            "query": {"text": "Question?"},
            "answer": {"text": "Answer", "citations": []},
            "retrieved_chunks": [{"text": "Context"}],
        }
        reference_answer = "Reference"

        # Mock the metric.measure() method
        with patch.object(evaluator._metrics["faithfulness"], "measure"):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(
                        evaluator._metrics["answer_relevancy"], "measure"
                    ):
                        # Set mock scores directly on the metrics
                        evaluator._metrics["faithfulness"].score = 0.95
                        evaluator._metrics["context_precision"].score = 0.85
                        evaluator._metrics["context_recall"].score = 0.90
                        evaluator._metrics["answer_relevancy"].score = 0.88

                        results = evaluator.compute_metrics_with_timing(
                            rag_output, reference_answer
                        )

                        assert "metric_computation_time_ms" in results
                        assert results["metric_computation_time_ms"] >= 0

    def test_slots_prevent_dynamic_attributes(self, monkeypatch):
        """Test that __slots__ prevents dynamic attribute creation."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        # Attempt to set a non-slotted attribute should raise AttributeError
        with pytest.raises(AttributeError):
            evaluator.dynamic_attribute = "value"
