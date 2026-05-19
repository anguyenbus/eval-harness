"""Tests for RagAdapter."""

from pathlib import Path

import pytest

from eval_harness.adapters.rag_adapter import RagAdapter
from eval_harness.adapters.schema_validator import SchemaValidationError
from eval_harness.stubs.stub_ingestion import query as stub_query


class TestRagAdapter:
    """Test suite for RagAdapter."""

    def test_adapter_invokes_query_callable(self, tmp_path):
        """Test that adapter invokes the provided query callable."""
        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        adapter = RagAdapter(query_callable=stub_query)
        output = adapter.query("What is this?", dummy_corpus)

        assert output is not None
        assert "schema_version" in output

    def test_adapter_validates_output(self, tmp_path):
        """Test that adapter validates output against schema."""
        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        adapter = RagAdapter(query_callable=stub_query)
        output = adapter.query("Test question?", dummy_corpus)

        # Should have validated and returned valid output
        assert output["schema_version"] == "1.0.0"
        assert "retrieved_chunks" in output

    def test_adapter_raises_on_invalid_output(self, tmp_path):
        """Test that adapter raises SchemaValidationError for invalid output."""

        def invalid_query(question: str, corpus_dir: Path) -> dict:
            # Returns invalid output missing required fields
            return {"invalid": "output"}

        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        adapter = RagAdapter(query_callable=invalid_query)

        with pytest.raises(SchemaValidationError):
            adapter.query("Test?", dummy_corpus)

    def test_adapter_default_is_stub(self, tmp_path):
        """Test that adapter defaults to stub RAG."""
        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        # Create adapter with default (should be stub)
        adapter = RagAdapter()
        output = adapter.query("Test question?", dummy_corpus)

        assert output["system_version"]["pipeline_version"] == "0.0.1-stub"
        assert output["answer"]["answer_supported"] is False

    def test_adapter_preserves_question(self, tmp_path):
        """Test that adapter correctly passes question to RAG."""
        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        test_question = "What is the capital of France?"
        adapter = RagAdapter(query_callable=stub_query)
        output = adapter.query(test_question, dummy_corpus)

        # Question should be preserved in output
        assert test_question in output["query"]["text"]
