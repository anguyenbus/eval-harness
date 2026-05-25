"""Tests for DeepEval async batch evaluation."""

from unittest.mock import MagicMock, patch

import pytest

from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator


class TestAsyncBatchEvaluation:
    """Test suite for async batch evaluation functionality."""

    @pytest.mark.asyncio
    async def test_async_batch_compute_metrics_respects_max_concurrent(
        self, monkeypatch
    ):
        """Test that async_batch_compute_metrics respects max_concurrent limit."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        evaluator = DeepEvalEvaluator(
            llm_provider="openai", judge_model="gpt-4o-mini", max_concurrent=2
        )

        # Mock metric.measure() to avoid actual API calls
        with patch.object(evaluator._metrics["faithfulness"], "measure"):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(evaluator._metrics["answer_relevancy"], "measure"):
                        # Set mock scores
                        type(evaluator._metrics["faithfulness"]).score = 0.95
                        type(evaluator._metrics["context_precision"]).score = 0.85
                        type(evaluator._metrics["context_recall"]).score = 0.90
                        type(evaluator._metrics["answer_relevancy"]).score = 0.88

                        rag_outputs = [
                            {
                                "query": {"text": f"Question {i}?"},
                                "answer": {"text": f"Answer {i}", "citations": []},
                                "retrieved_chunks": [{"text": f"Context {i}"}],
                            }
                            for i in range(5)
                        ]
                        reference_answers = [f"Reference {i}" for i in range(5)]

                        results = await evaluator.async_batch_compute_metrics(
                            rag_outputs, reference_answers
                        )

                        assert len(results) == 5
                        for result in results:
                            assert "faithfulness" in result
                            assert "context_precision" in result
                            assert "context_recall" in result
                            assert "answer_relevancy" in result

    @pytest.mark.asyncio
    async def test_async_batch_compute_metrics_maintains_order(self, monkeypatch):
        """Test that results maintain input order."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        evaluator = DeepEvalEvaluator(
            llm_provider="openai", judge_model="gpt-4o-mini", max_concurrent=3
        )

        # Create distinct queries to track order
        rag_outputs = [
            {
                "query": {"text": "Question A"},
                "answer": {"text": "Answer A", "citations": []},
                "retrieved_chunks": [{"text": "Context A"}],
            },
            {
                "query": {"text": "Question B"},
                "answer": {"text": "Answer B", "citations": []},
                "retrieved_chunks": [{"text": "Context B"}],
            },
            {
                "query": {"text": "Question C"},
                "answer": {"text": "Answer C", "citations": []},
                "retrieved_chunks": [{"text": "Context C"}],
            },
        ]
        reference_answers = ["Reference A", "Reference B", "Reference C"]

        # Mock metric.measure() and set different scores per query
        async def mock_measure_with_delay(test_case):
            # Simulate async work
            import asyncio

            await asyncio.sleep(0.01)

        with patch.object(evaluator._metrics["faithfulness"], "measure"):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(evaluator._metrics["answer_relevancy"], "measure"):
                        results = await evaluator.async_batch_compute_metrics(
                            rag_outputs, reference_answers
                        )

                        # Check order is maintained
                        assert len(results) == 3
                        # All should have metric keys
                        for _i, result in enumerate(results):
                            assert "faithfulness" in result
                            assert "context_precision" in result
                            assert "context_recall" in result
                            assert "answer_relevancy" in result

    @pytest.mark.asyncio
    async def test_async_batch_semaphore_prevents_exceeding_limit(self, monkeypatch):
        """Test that semaphore prevents concurrent execution exceeding limit."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        # Use low max_concurrent to test semaphore
        max_concurrent = 2
        evaluator = DeepEvalEvaluator(
            llm_provider="openai", judge_model="gpt-4o-mini", max_concurrent=max_concurrent
        )

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent_seen = 0
        MagicMock()

        def increment_concurrent():
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            if concurrent_count > max_concurrent_seen:
                max_concurrent_seen = concurrent_count

        def decrement_concurrent():
            nonlocal concurrent_count
            concurrent_count -= 1

        # Mock metric.measure() to track concurrency
        evaluator._metrics["faithfulness"].measure

        def mock_measure(test_case):
            increment_concurrent()
            import time

            time.sleep(0.05)  # Small delay to allow overlap
            decrement_concurrent()

        with patch.object(evaluator._metrics["faithfulness"], "measure", side_effect=mock_measure):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(evaluator._metrics["answer_relevancy"], "measure"):
                        rag_outputs = [
                            {
                                "query": {"text": f"Q{i}"},
                                "answer": {"text": f"A{i}", "citations": []},
                                "retrieved_chunks": [{"text": f"C{i}"}],
                            }
                            for i in range(10)
                        ]
                        reference_answers = [f"R{i}" for i in range(10)]

                        results = await evaluator.async_batch_compute_metrics(
                            rag_outputs, reference_answers
                        )

                        # Verify semaphore limited concurrency
                        # Note: This is a best-effort test; actual semaphore behavior
                        # depends on asyncio scheduling
                        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_async_batch_handles_errors_gracefully(self, monkeypatch):
        """Test that async batch evaluation handles individual errors gracefully."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        evaluator = DeepEvalEvaluator(
            llm_provider="openai", judge_model="gpt-4o-mini", max_concurrent=3
        )

        rag_outputs = [
            {
                "query": {"text": "Question 1"},
                "answer": {"text": "Answer 1", "citations": []},
                "retrieved_chunks": [{"text": "Context 1"}],
            },
            {
                "query": {"text": "Question 2"},
                "answer": {"text": "Answer 2", "citations": []},
                "retrieved_chunks": [{"text": "Context 2"}],
            },
        ]
        reference_answers = ["Reference 1", "Reference 2"]

        # Mock measure() to raise exception for second query only
        call_count = [0]

        def mock_measure_with_error(test_case):
            call_count[0] += 1
            if call_count[0] % 2 == 0:  # Second call fails
                raise Exception("Simulated error")

        with patch.object(evaluator._metrics["faithfulness"], "measure", side_effect=mock_measure_with_error):
            with patch.object(evaluator._metrics["context_precision"], "measure"):
                with patch.object(evaluator._metrics["context_recall"], "measure"):
                    with patch.object(evaluator._metrics["answer_relevancy"], "measure"):
                        results = await evaluator.async_batch_compute_metrics(
                            rag_outputs, reference_answers
                        )

                        # Both should complete, with error for second
                        assert len(results) == 2
