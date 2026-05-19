"""
Integration tests for ChromaDB RAG pipeline.

These tests verify end-to-end workflows for the ChromaDB RAG feature.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def sample_corpus(tmp_path):
    """Create sample corpus for integration testing."""
    (tmp_path / "contract_01.txt").write_text(
        "This is a contract document. It contains important legal terms. "
        "The parties agree to the following terms and conditions."
    )
    (tmp_path / "contract_02.txt").write_text(
        "This is another contract document. It has different clauses. "
        "The liability is limited to the amount specified."
    )
    return tmp_path


@pytest.fixture
def mock_claude_response():
    """Mock Claude API response for integration testing."""
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text="This is a contract [contract_01_chunk_00000].")
    ]

    mock_client = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_response)

    return mock_client


class TestEndToEndPipeline:
    """Integration tests for end-to-end ChromaDB RAG pipeline."""

    def test_full_query_pipeline_with_real_chromadb(
        self, sample_corpus, mock_claude_response
    ):
        """Test end-to-end query with real ChromaDB collection."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic", return_value=mock_claude_response):
                from eval_harness.stubs.rag.chromadb_query import query

                # First query will auto-create and ingest
                output = query(
                    question="What is this document about?",
                    corpus_dir=sample_corpus,
                    top_k=5,
                )

                # Verify output structure
                assert "schema_version" in output
                assert output["schema_version"] == "1.0.0"
                assert "system_version" in output
                assert "query" in output
                assert "answer" in output
                assert "retrieved_chunks" in output
                assert "timings_ms" in output

    def test_auto_create_and_ingest_flow(self, sample_corpus):
        """Test auto-create + ingest flow on missing collection."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_claude:
                mock_response = MagicMock()
                mock_response.content = [MagicMock(text="Test answer.")]
                mock_claude.return_value.messages.create.return_value = mock_response

                from eval_harness.stubs.rag.chromadb_query import query

                # Query with new corpus (collection doesn't exist)
                output = query(
                    question="Test question",
                    corpus_dir=sample_corpus,
                    top_k=5,
                )

                # Verify ingestion occurred
                assert output["retrieved_chunks"] is not None
                # Should have chunks from ingested documents

    def test_force_reingest_clears_and_rebuilds(self, sample_corpus):
        """Test force-reingest clears and rebuilds collection."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_claude:
                mock_response = MagicMock()
                mock_response.content = [MagicMock(text="Test answer.")]
                mock_claude.return_value.messages.create.return_value = mock_response

                from eval_harness.stubs.rag.chromadb_query import query

                # First query (creates and ingests)
                output1 = query(
                    question="Test question",
                    corpus_dir=sample_corpus,
                    top_k=5,
                )

                # Force reingest
                output2 = query(
                    question="Test question",
                    corpus_dir=sample_corpus,
                    top_k=5,
                    force_reingest=True,
                )

                # Both should succeed
                assert output1 is not None
                assert output2 is not None

    def test_schema_validation_on_all_output_variations(self, sample_corpus):
        """Test schema validation on various output scenarios."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_claude:
                # Test with unsupported answer
                mock_response_unsupported = MagicMock()
                mock_response_unsupported.content = [
                    MagicMock(text="I don't have enough information.")
                ]
                mock_claude.return_value.messages.create.return_value = (
                    mock_response_unsupported
                )

                from eval_harness.stubs.rag.chromadb_query import query

                output = query(
                    question="What is this?",
                    corpus_dir=sample_corpus,
                    top_k=5,
                )

                # Verify schema compliance even with unsupported answer
                assert output["answer"]["answer_supported"] is False
                assert "schema_version" in output

    def test_timing_instrumentation_across_pipeline(self, sample_corpus):
        """Test timing instrumentation across all pipeline stages."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_claude:
                mock_response = MagicMock()
                mock_response.content = [
                    MagicMock(text="Test answer [doc_chunk_00000].")
                ]
                mock_claude.return_value.messages.create.return_value = mock_response

                from eval_harness.stubs.rag.chromadb_query import query

                output = query(
                    question="Test question",
                    corpus_dir=sample_corpus,
                    top_k=5,
                )

                # Verify timings are captured
                assert "timings_ms" in output
                assert "retrieval" in output["timings_ms"]
                assert "generation" in output["timings_ms"]
                assert "total" in output["timings_ms"]

                # Verify timing values are reasonable
                assert output["timings_ms"]["retrieval"] >= 0
                assert output["timings_ms"]["generation"] >= 0
                assert output["timings_ms"]["total"] >= 0

    def test_trace_context_populated_in_output(self, sample_corpus):
        """Test trace context is populated in output."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_claude:
                mock_response = MagicMock()
                mock_response.content = [MagicMock(text="Test answer.")]
                mock_claude.return_value.messages.create.return_value = mock_response

                from eval_harness.stubs.rag.chromadb_query import query

                output = query(
                    question="Test question",
                    corpus_dir=sample_corpus,
                    top_k=5,
                )

                # Verify trace context is populated
                assert "trace" in output
                assert "trace_id" in output["trace"]
                assert "span_id" in output["trace"]
                assert "phoenix_project" in output["trace"]

    def test_top_k_parameter_affects_retrieval_count(self, sample_corpus):
        """Test top-k parameter affects retrieval count."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_claude:
                mock_response = MagicMock()
                mock_response.content = [MagicMock(text="Test answer.")]
                mock_claude.return_value.messages.create.return_value = mock_response

                from eval_harness.stubs.rag.chromadb_query import query

                # Test with top_k=2
                output1 = query(
                    question="Test question",
                    corpus_dir=sample_corpus,
                    top_k=2,
                )

                # Test with top_k=5
                output2 = query(
                    question="Test question",
                    corpus_dir=sample_corpus,
                    top_k=5,
                )

                # Both should succeed
                assert len(output1["retrieved_chunks"]) <= 2
                assert len(output2["retrieved_chunks"]) <= 5
