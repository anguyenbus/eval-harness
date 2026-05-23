"""Tests for RAGAS adapter module."""

from unittest.mock import MagicMock, patch

from eval_harness.adapters.ragas_adapter import (
    RagasEvaluator,
    transform_to_ragas_sample,
)


class TestTransformToRagasSample:
    """Test suite for transform_to_ragas_sample function."""

    def test_transform_question_to_user_input(self):
        """Test that question is mapped to user_input."""
        rag_output = {
            "query": {"text": "What is the termination clause?"},
            "answer": {"text": "The contract can be terminated...", "citations": []},
            "retrieved_chunks": [],
        }
        reference_answer = "The contract allows termination with 30 days notice."

        sample = transform_to_ragas_sample(rag_output, reference_answer)

        assert sample.user_input == "What is the termination clause?"

    def test_transform_retrieved_chunks_to_contexts(self):
        """Test that retrieved_chunks are mapped to retrieved_contexts."""
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

        sample = transform_to_ragas_sample(rag_output, reference_answer)

        assert len(sample.retrieved_contexts) == 3
        assert sample.retrieved_contexts[0] == "Context 1"
        assert sample.retrieved_contexts[1] == "Context 2"
        assert sample.retrieved_contexts[2] == "Context 3"

    def test_transform_response_to_response(self):
        """Test that answer text is mapped to response."""
        rag_output = {
            "query": {"text": "Question?"},
            "answer": {"text": "This is the generated answer.", "citations": []},
            "retrieved_chunks": [],
        }
        reference_answer = "Reference"

        sample = transform_to_ragas_sample(rag_output, reference_answer)

        assert sample.response == "This is the generated answer."

    def test_transform_reference_to_reference(self):
        """Test that reference answer is mapped to reference."""
        rag_output = {
            "query": {"text": "Question?"},
            "answer": {"text": "Answer", "citations": []},
            "retrieved_chunks": [],
        }
        reference_answer = "This is the reference answer."

        sample = transform_to_ragas_sample(rag_output, reference_answer)

        assert sample.reference == "This is the reference answer."

    def test_handle_relevant_passage_id(self):
        """Test that relevant_passage_id is handled correctly."""
        rag_output = {
            "query": {"text": "Question?"},
            "answer": {"text": "Answer", "citations": []},
            "retrieved_chunks": [
                {"doc_id": "passage_123", "text": "Relevant passage"},
            ],
            "relevant_passage_id": "passage_123",
        }
        reference_answer = "Reference"

        sample = transform_to_ragas_sample(rag_output, reference_answer)

        # Should still transform correctly
        assert sample.user_input == "Question?"


class TestRagasEvaluator:
    """Test suite for RagasEvaluator class."""

    def test_initialization(self, monkeypatch):
        """Test that RagasEvaluator initializes with metrics."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        evaluator = RagasEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")

        assert evaluator is not None
        assert evaluator._metrics is not None

    def test_compute_metrics(self, monkeypatch):
        """Test that compute_metrics calculates RAGAS scores."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        evaluator = RagasEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")

        # Mock the metric evaluation
        with patch.object(
            evaluator._metrics["faithfulness"],
            "score",
            return_value=MagicMock(faithfulness=0.95),
        ):
            with patch.object(
                evaluator._metrics["context_precision"],
                "score",
                return_value=MagicMock(context_precision=0.85),
            ):
                with patch.object(
                    evaluator._metrics["context_recall"],
                    "score",
                    return_value=MagicMock(context_recall=0.90),
                ):
                    with patch.object(
                        evaluator._metrics["answer_relevancy"],
                        "score",
                        return_value=MagicMock(answer_relevancy=0.88),
                    ):
                        rag_output = {
                            "query": {"text": "Question?"},
                            "answer": {"text": "Answer", "citations": []},
                            "retrieved_chunks": [{"text": "Context"}],
                        }
                        reference_answer = "Reference"

                        results = evaluator.compute_metrics(
                            rag_output, reference_answer
                        )

                        assert "faithfulness" in results
                        assert "context_precision" in results
                        assert "context_recall" in results
                        assert "answer_relevancy" in results

    def test_compute_metrics_returns_structured_scores(self, monkeypatch):
        """Test that compute_metrics returns properly structured scores."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        evaluator = RagasEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")

        # Create mock score result
        mock_score = MagicMock()
        mock_score.faithfulness = 0.95

        with patch.object(
            evaluator._metrics["faithfulness"], "score", return_value=mock_score
        ):
            rag_output = {
                "query": {"text": "Question?"},
                "answer": {"text": "Answer", "citations": []},
                "retrieved_chunks": [{"text": "Context"}],
            }
            reference_answer = "Reference"

            results = evaluator.compute_metrics(rag_output, reference_answer)

            assert isinstance(results, dict)

    def test_cost_tracking_initialization(self, monkeypatch):
        """Test that cost tracking can be initialized."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        evaluator = RagasEvaluator(
            llm_provider="openai", judge_model="gpt-4o-mini", track_costs=True
        )

        assert evaluator is not None
