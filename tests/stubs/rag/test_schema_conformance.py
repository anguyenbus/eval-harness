"""
Tests for schema validation and conformance.

These tests verify the validate_rag_output function and schema compliance.
"""

import pytest

from eval_harness.adapters.schema_validator import SchemaValidationError
from eval_harness.stubs.rag.chromadb_config import (
    CORPUS_LOADER_VERSION,
    EMBEDDING_MODEL,
    GENERATOR_MODEL,
    PIPELINE_VERSION,
)
from eval_harness.stubs.rag.schema_conformance import validate_rag_output


@pytest.fixture
def valid_output():
    """Create valid RAG output for testing."""
    return {
        "schema_version": "1.0.0",
        "system_version": {
            "pipeline_version": PIPELINE_VERSION,
            "corpus_loader_version": CORPUS_LOADER_VERSION,
            "embedder_model": EMBEDDING_MODEL,
            "generator_model": GENERATOR_MODEL,
        },
        "query": {
            "query_id": "test_001",
            "text": "What is this?",
            "metadata": {},
        },
        "answer": {
            "text": "This is the answer [doc1_chunk_00000].",
            "answer_supported": True,
            "citations": [
                {
                    "claim_span": [0, 30],
                    "chunk_ids": ["doc1_chunk_00000"],
                }
            ],
        },
        "retrieved_chunks": [
            {
                "chunk_id": "doc1_chunk_00000",
                "rank": 0,
                "score": 0.95,
                "retrieval_stage": "initial",
                "doc_id": "doc1",
                "text": "Test content",
                "char_span": [0, 12],
                "element_ids": [],
                "page_indices": [],
            }
        ],
    }


class TestSchemaConformance:
    """Test suite for schema validation."""

    def test_output_validates_against_schema(self, valid_output):
        """Test output validates against rag_query_output.schema.json."""
        # Should not raise error
        validate_rag_output(valid_output)

    def test_schema_version_is_1_0_0(self, valid_output):
        """Test schema_version is '1.0.0'."""
        assert valid_output["schema_version"] == "1.0.0"

    def test_pipeline_version_is_correct(self, valid_output):
        """Test pipeline_version is '0.1.0-chromadb'."""
        assert valid_output["system_version"]["pipeline_version"] == PIPELINE_VERSION
        assert PIPELINE_VERSION == "0.1.0-chromadb"

    def test_corpus_loader_version_is_correct(self, valid_output):
        """Test corpus_loader_version is '0.1.0'."""
        assert (
            valid_output["system_version"]["corpus_loader_version"]
            == CORPUS_LOADER_VERSION
        )
        assert CORPUS_LOADER_VERSION == "0.1.0"

    def test_embedder_model_is_correct(self, valid_output):
        """Test embedder_model is sentence-transformers/all-MiniLM-L6-v2."""
        assert valid_output["system_version"]["embedder_model"] == EMBEDDING_MODEL
        assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_generator_model_is_correct_or_from_env(self, valid_output):
        """Test generator_model is claude-opus-4-7 or from env."""
        assert valid_output["system_version"]["generator_model"] == GENERATOR_MODEL

    def test_all_retrieved_chunks_have_char_span(self, valid_output):
        """Test all retrieved_chunks have char_span."""
        for chunk in valid_output["retrieved_chunks"]:
            assert "char_span" in chunk

    def test_citations_reference_valid_chunk_ids(self, valid_output):
        """Test citations reference valid chunk_ids."""
        validate_rag_output(valid_output)

        # Verify chunk_ids are valid
        valid_chunk_ids = {c["chunk_id"] for c in valid_output["retrieved_chunks"]}
        for citation in valid_output["answer"]["citations"]:
            for chunk_id in citation["chunk_ids"]:
                assert chunk_id in valid_chunk_ids

    def test_invalid_schema_version_raises_error(self, valid_output):
        """Test invalid schema_version raises error."""
        valid_output["schema_version"] = "2.0.0"

        with pytest.raises(SchemaValidationError):
            validate_rag_output(valid_output)

    def test_invalid_pipeline_version_raises_error(self, valid_output):
        """Test invalid pipeline_version raises error."""
        valid_output["system_version"]["pipeline_version"] = "wrong-version"

        with pytest.raises(SchemaValidationError, match="pipeline_version mismatch"):
            validate_rag_output(valid_output)

    def test_invalid_embedder_model_raises_error(self, valid_output):
        """Test invalid embedder_model raises error."""
        valid_output["system_version"]["embedder_model"] = "wrong-model"

        with pytest.raises(SchemaValidationError, match="embedder_model mismatch"):
            validate_rag_output(valid_output)

    def test_invalid_generator_model_raises_error(self, valid_output):
        """Test invalid generator_model raises error."""
        valid_output["system_version"]["generator_model"] = "wrong-model"

        with pytest.raises(SchemaValidationError, match="generator_model mismatch"):
            validate_rag_output(valid_output)

    def test_citation_with_invalid_chunk_id_raises_error(self, valid_output):
        """Test citation with invalid chunk_id raises error."""
        valid_output["answer"]["citations"].append(
            {
                "claim_span": [0, 10],
                "chunk_ids": ["invalid_chunk_id"],
            }
        )

        with pytest.raises(SchemaValidationError, match="invalid chunk_id"):
            validate_rag_output(valid_output)

    def test_retrieved_chunk_without_char_span_raises_error(self, valid_output):
        """Test retrieved chunk without char_span raises error."""
        valid_output["retrieved_chunks"][0].pop("char_span")

        with pytest.raises(SchemaValidationError):
            validate_rag_output(valid_output)
