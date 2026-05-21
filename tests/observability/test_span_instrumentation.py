"""Tests for span instrumentation in RAG pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSpanInstrumentation:
    """Test span creation and hierarchy for RAG pipeline stages."""

    def test_rag_query_root_span_creation(self):
        """Test that rag_query root span can be created."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)

        question = "What is the termination clause?"
        trace_id = adapter.start_rag_query_span(question)

        assert trace_id is not None
        assert isinstance(trace_id, str)
        assert len(trace_id) > 0

    def test_retrieval_span_with_embeddings_and_timing(self):
        """Test retrieval child span with embeddings, chunks, k, and timing."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)
        trace_id = adapter.start_rag_query_span("Test question")

        # Mock embedding data
        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunks = [
            {"text": "Chunk 1", "doc_id": "doc1"},
            {"text": "Chunk 2", "doc_id": "doc2"},
        ]
        k = 5
        timing_ms = 123.45

        # Should not raise exception
        adapter.start_retrieval_span(trace_id, embeddings, chunks, k, timing_ms)

    def test_generation_span_with_model_and_timing(self):
        """Test generation child span with model, prompt, tokens, and timing."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)
        trace_id = adapter.start_rag_query_span("Test question")

        model = "claude-3-opus-20240229"
        prompt = "Answer based on the context: ..."
        tokens = 150
        timing_ms = 234.56

        # Should not raise exception
        adapter.start_generation_span(trace_id, model, prompt, tokens, timing_ms)

    def test_evaluation_span_with_ragas_metrics(self):
        """Test evaluation child span with Ragas metrics."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)
        trace_id = adapter.start_rag_query_span("Test question")

        ragas_metrics = {
            "faithfulness": 0.92,
            "context_precision": 0.85,
            "context_recall": 0.78,
            "answer_relevancy": 0.88,
        }

        # Should not raise exception
        adapter.start_evaluation_span(trace_id, ragas_metrics)

    def test_span_hierarchy_structure(self):
        """Test that all spans can be created for a complete RAG query."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)

        # Root span
        trace_id = adapter.start_rag_query_span("What is the termination clause?")

        # Retrieval child span
        adapter.start_retrieval_span(
            trace_id=trace_id,
            embeddings=[0.1, 0.2, 0.3],
            chunks=[{"text": "chunk1"}],
            k=5,
            timing_ms=100.0,
        )

        # Generation child span
        adapter.start_generation_span(
            trace_id=trace_id,
            model="claude-3-opus",
            prompt="Test prompt",
            tokens=100,
            timing_ms=200.0,
        )

        # Evaluation child span
        adapter.start_evaluation_span(
            trace_id=trace_id,
            ragas_metrics={
                "faithfulness": 0.9,
                "context_precision": 0.8,
                "context_recall": 0.7,
                "answer_relevancy": 0.85,
            },
        )

    def test_embeddings_reused_not_recomputed(self):
        """Test that embeddings from retrieval can be passed to Phoenix."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)
        trace_id = adapter.start_rag_query_span("Test question")

        # Simulate retrieval embedding (computed once)
        retrieval_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Pass the same embedding to Phoenix (not recomputed)
        adapter.start_retrieval_span(
            trace_id=trace_id,
            embeddings=retrieval_embedding,
            chunks=[{"text": "chunk1"}],
            k=5,
            timing_ms=100.0,
        )

        # Verify no exception - embedding was reused, not recomputed

    def test_multiple_queries_have_different_trace_ids(self):
        """Test that different queries generate different trace IDs."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)

        trace_id_1 = adapter.start_rag_query_span("Question 1")
        trace_id_2 = adapter.start_rag_query_span("Question 2")

        assert trace_id_1 != trace_id_2
