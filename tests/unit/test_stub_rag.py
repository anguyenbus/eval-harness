"""Tests for stub RAG implementation."""

from pathlib import Path

from eval_harness.adapters.schema_validator import validate
from eval_harness.stubs.stub_ingestion import query


class TestStubRAG:
    """Test suite for stub RAG system."""

    def test_stub_output_validates_against_schema(self, tmp_path):
        """Test that stub RAG output validates against rag_query_output schema."""
        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        output = query("What is this document about?", dummy_corpus)

        schema_path = Path("contracts/rag_query_output.schema.json")
        validate(output, schema_path)

    def test_stub_includes_required_fields(self, tmp_path):
        """Test that stub output includes all required fields."""
        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        output = query("Test question?", dummy_corpus)

        # Check required top-level fields
        assert "schema_version" in output
        assert output["schema_version"] == "1.0.0"
        assert "system_version" in output
        assert "query" in output
        assert "answer" in output
        assert "retrieved_chunks" in output

    def test_stub_system_version_is_stub(self, tmp_path):
        """Test that stub has stub pipeline version."""
        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        output = query("Test?", dummy_corpus)

        assert output["system_version"]["pipeline_version"] == "0.0.1-stub"

    def test_stub_has_one_retrieved_chunk(self, tmp_path):
        """Test that stub returns exactly one retrieved chunk."""
        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        output = query("Test?", dummy_corpus)

        assert len(output["retrieved_chunks"]) == 1

        chunk = output["retrieved_chunks"][0]
        assert "chunk_id" in chunk
        assert "rank" in chunk
        assert chunk["rank"] == 0
        assert "score" in chunk
        assert "doc_id" in chunk
        assert "text" in chunk
        assert "char_span" in chunk

    def test_stub_answer_supported_is_false(self, tmp_path):
        """Test that stub answer indicates not supported."""
        dummy_corpus = tmp_path / "corpus"
        dummy_corpus.mkdir()

        output = query("Test?", dummy_corpus)

        assert "answer" in output
        assert output["answer"]["answer_supported"] is False
        assert output["answer"]["citations"] == []

    def test_stub_works_with_different_inputs(self, tmp_path):
        """Test that stub works with different questions and corpus paths."""
        corpus1 = tmp_path / "corpus1"
        corpus2 = tmp_path / "corpus2"
        corpus1.mkdir()
        corpus2.mkdir()

        output1 = query("Question 1?", corpus1)
        output2 = query("Question 2?", corpus2)

        # Both should be valid
        assert output1["schema_version"] == "1.0.0"
        assert output2["schema_version"] == "1.0.0"

        # Query text should be preserved
        assert "Question 1?" in output1["query"]["text"]
        assert "Question 2?" in output2["query"]["text"]
