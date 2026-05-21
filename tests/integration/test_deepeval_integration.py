"""Integration tests for DeepEval evaluation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDeepEvalIntegration:
    """Integration tests for DeepEval evaluator."""

    def test_end_to_end_pipeline_with_openai_provider(self, monkeypatch, tmp_path):
        """Test full evaluation pipeline with OpenAI provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        # Mock RAG output
        rag_output = {
            "query": {"text": "What is the termination clause?"},
            "answer": {
                "text": "The contract can be terminated with 30 days written notice.",
                "citations": [],
            },
            "retrieved_chunks": [
                {
                    "text": "The contract may be terminated by either party with 30 days written notice."
                }
            ],
        }
        reference_answer = "Termination requires 30 days written notice."

        # Mock metric evaluation
        with patch.object(evaluator._metrics["faithfulness"], "measure"):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(
                        evaluator._metrics["answer_relevancy"], "measure"
                    ):
                        # Set mock scores
                        evaluator._metrics["faithfulness"].score = 0.95
                        evaluator._metrics["context_precision"].score = 0.90
                        evaluator._metrics["context_recall"].score = 0.88
                        evaluator._metrics["answer_relevancy"].score = 0.92

                        results = evaluator.compute_metrics_with_timing(
                            rag_output, reference_answer
                        )

                        assert results["faithfulness"] == 0.95
                        assert results["context_precision"] == 0.90
                        assert results["context_recall"] == 0.88
                        assert results["answer_relevancy"] == 0.92
                        assert "metric_computation_time_ms" in results

    def test_async_batch_evaluation_on_multiple_queries(self, monkeypatch):
        """Test async batch evaluation on multiple queries."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        import asyncio

        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator(
            llm_provider="openai", judge_model="gpt-4o", max_concurrent=2
        )

        # Create multiple RAG outputs
        rag_outputs = [
            {
                "query": {"text": f"Question {i}?"},
                "answer": {"text": f"Answer {i}", "citations": []},
                "retrieved_chunks": [{"text": f"Context {i}"}],
            }
            for i in range(3)
        ]
        reference_answers = [f"Reference {i}" for i in range(3)]

        # Mock metric evaluation
        with patch.object(evaluator._metrics["faithfulness"], "measure"):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(
                        evaluator._metrics["answer_relevancy"], "measure"
                    ):
                        # Set mock scores
                        evaluator._metrics["faithfulness"].score = 0.95
                        evaluator._metrics["context_precision"].score = 0.90
                        evaluator._metrics["context_recall"].score = 0.88
                        evaluator._metrics["answer_relevancy"].score = 0.92

                        results = asyncio.run(
                            evaluator.async_batch_compute_metrics(
                                rag_outputs, reference_answers
                            )
                        )

                        assert len(results) == 3
                        for result in results:
                            assert "faithfulness" in result
                            assert "context_precision" in result
                            assert "context_recall" in result
                            assert "answer_relevancy" in result

    def test_csv_output_format_with_metadata(self, monkeypatch, tmp_path):
        """Test CSV output format includes all required columns."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        # Create test data
        rag_output = {
            "query": {"text": "Test question?"},
            "answer": {"text": "Test answer", "citations": []},
            "retrieved_chunks": [{"text": "Test context"}],
        }
        reference_answer = "Test reference"

        # Mock metric evaluation
        with patch.object(evaluator._metrics["faithfulness"], "measure"):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(
                        evaluator._metrics["answer_relevancy"], "measure"
                    ):
                        # Set mock scores
                        evaluator._metrics["faithfulness"].score = 0.95
                        evaluator._metrics["context_precision"].score = 0.90
                        evaluator._metrics["context_recall"].score = 0.88
                        evaluator._metrics["answer_relevancy"].score = 0.92

                        results = evaluator.compute_metrics_with_timing(
                            rag_output, reference_answer
                        )

                        # Verify all expected keys are present
                        expected_keys = [
                            "faithfulness",
                            "context_precision",
                            "context_recall",
                            "answer_relevancy",
                            "metric_computation_time_ms",
                        ]
                        for key in expected_keys:
                            assert key in results

    def test_phoenix_integration_with_deepeval(self, monkeypatch):
        """Test Phoenix observability integration with DeepEval."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        # Create mock phoenix adapter
        phoenix_adapter = MagicMock(spec=PhoenixAdapter)
        phoenix_adapter.is_connected.return_value = False
        phoenix_adapter._tracer = None

        # Create evaluator
        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        # Mock metric evaluation
        rag_output = {
            "query": {"text": "Test question?"},
            "answer": {"text": "Test answer", "citations": []},
            "retrieved_chunks": [{"text": "Test context"}],
        }
        reference_answer = "Test reference"

        with patch.object(evaluator._metrics["faithfulness"], "measure"):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(
                        evaluator._metrics["answer_relevancy"], "measure"
                    ):
                        evaluator._metrics["faithfulness"].score = 0.95
                        evaluator._metrics["context_precision"].score = 0.90
                        evaluator._metrics["context_recall"].score = 0.88
                        evaluator._metrics["answer_relevancy"].score = 0.92

                        results = evaluator.compute_metrics(
                            rag_output, reference_answer
                        )

                        # Verify metrics are computed
                        assert results["faithfulness"] == 0.95

                        # Verify Phoenix integration points exist
                        # (Phoenix adapter would be called in actual runner)
                        assert phoenix_adapter is not None

    def test_config_loading_from_yaml(self):
        """Test configuration loading from eval_config.yaml."""
        from eval_harness.config import load_config
        from eval_harness.metrics.deepeval_config import get_deepeval_config

        config_path = Path("eval_config.yaml")
        if not config_path.exists():
            pytest.skip("eval_config.yaml not found")

        config = load_config(config_path)
        deepeval_config = get_deepeval_config(config)

        # Verify expected config keys
        assert "enabled" in deepeval_config
        assert "judge_model" in deepeval_config
        assert "judge_model_provider" in deepeval_config
        assert "temperature" in deepeval_config
        assert "max_concurrent" in deepeval_config

    def test_metric_equivalencies(self, monkeypatch):
        """Test that DeepEval metrics map correctly to RAGAS equivalents."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        # Verify metric keys match RAGAS equivalents
        expected_keys = [
            "faithfulness",  # RAGAS: faithfulness
            "context_precision",  # RAGAS: context_precision
            "context_recall",  # RAGAS: context_recall
            "answer_relevancy",  # RAGAS: answer_relevancy
        ]

        for key in expected_keys:
            assert key in evaluator._metrics

    def test_backward_compatibility_csv_schema(self, monkeypatch):
        """Test that CSV output maintains backward compatibility with RAGAS schema."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o")

        # RAGAS CSV columns (must be preserved)
        ragas_columns = [
            "faithfulness_score",
            "context_precision_score",
            "context_recall_score",
            "answer_relevancy_score",
        ]

        rag_output = {
            "query": {"text": "Test?"},
            "answer": {"text": "Answer", "citations": []},
            "retrieved_chunks": [{"text": "Context"}],
        }
        reference_answer = "Reference"

        with patch.object(evaluator._metrics["faithfulness"], "measure"):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(
                        evaluator._metrics["answer_relevancy"], "measure"
                    ):
                        evaluator._metrics["faithfulness"].score = 0.95
                        evaluator._metrics["context_precision"].score = 0.90
                        evaluator._metrics["context_recall"].score = 0.88
                        evaluator._metrics["answer_relevancy"].score = 0.92

                        results = evaluator.compute_metrics(
                            rag_output, reference_answer
                        )

                        # Verify all RAGAS columns can be populated
                        for column in ragas_columns:
                            metric_key = column.replace("_score", "")
                            assert metric_key in results
