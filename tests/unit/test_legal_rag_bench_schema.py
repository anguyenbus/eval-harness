"""Tests for Legal RAG Bench schema validation."""

from pathlib import Path

import pytest

from eval_harness.adapters.schema_validator import (
    SchemaValidationError,
    validate,
)


class TestLegalRagBenchSchema:
    """Test suite for legal_rag_bench_query_output schema."""

    def test_valid_schema_instance_validates(self):
        """Test that valid Legal RAG Bench output passes validation."""
        schema_path = Path("contracts/legal_rag_bench_query_output.schema.json")

        valid_output = {
            "schema_version": "1.0.0",
            "system_version": {
                "pipeline_version": "0.1.0",
                "embedder_model": "text-embedding-3-large",
                "generator_model": "gpt-4o-mini",
            },
            "query": {
                "query_id": "q001",
                "text": "What is the termination clause?",
            },
            "answer": {
                "text": "The contract can be terminated with 30 days notice.",
                "citations": [],
            },
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk_001",
                    "rank": 0,
                    "score": 0.95,
                    "doc_id": "doc_001",
                    "text": "The contract may be terminated...",
                    "char_span": [0, 100],
                },
            ],
            "relevant_passage_id": "passage_123",
        }

        # Should not raise
        validate(valid_output, schema_path)

    def test_required_field_validation(self):
        """Test that required fields are validated."""
        schema_path = Path("contracts/legal_rag_bench_query_output.schema.json")

        # Missing required field: relevant_passage_id
        invalid_output = {
            "schema_version": "1.0.0",
            "system_version": {
                "pipeline_version": "0.1.0",
            },
            "query": {
                "query_id": "q001",
                "text": "What is the termination clause?",
            },
            "answer": {
                "text": "Answer text",
                "citations": [],
            },
            "retrieved_chunks": [],
            # Missing relevant_passage_id
        }

        with pytest.raises(SchemaValidationError):
            validate(invalid_output, schema_path)

    def test_single_relevant_passage_id_structure(self):
        """Test that relevant_passage_id accepts single string value."""
        schema_path = Path("contracts/legal_rag_bench_query_output.schema.json")

        output = {
            "schema_version": "1.0.0",
            "system_version": {
                "pipeline_version": "0.1.0",
            },
            "query": {
                "query_id": "q001",
                "text": "Question?",
            },
            "answer": {
                "text": "Answer",
                "citations": [],
            },
            "retrieved_chunks": [],
            "relevant_passage_id": "single_passage_id",  # Single string, not array
        }

        # Should not raise
        validate(output, schema_path)

    def test_schema_version_validation(self):
        """Test that schema_version is validated."""
        schema_path = Path("contracts/legal_rag_bench_query_output.schema.json")

        wrong_version = {
            "schema_version": "2.0.0",  # Wrong version
            "system_version": {
                "pipeline_version": "0.1.0",
            },
            "query": {
                "query_id": "q001",
                "text": "Question?",
            },
            "answer": {
                "text": "Answer",
                "citations": [],
            },
            "retrieved_chunks": [],
            "relevant_passage_id": "passage_123",
        }

        with pytest.raises(SchemaValidationError):
            validate(wrong_version, schema_path)

    def test_system_version_required_fields(self):
        """Test that system_version requires pipeline_version."""
        schema_path = Path("contracts/legal_rag_bench_query_output.schema.json")

        missing_pipeline_version = {
            "schema_version": "1.0.0",
            "system_version": {
                # Missing pipeline_version
                "embedder_model": "text-embedding-3-large",
            },
            "query": {
                "query_id": "q001",
                "text": "Question?",
            },
            "answer": {
                "text": "Answer",
                "citations": [],
            },
            "retrieved_chunks": [],
            "relevant_passage_id": "passage_123",
        }

        with pytest.raises(SchemaValidationError):
            validate(missing_pipeline_version, schema_path)

    def test_optional_metadata_in_query(self):
        """Test that query can have optional metadata field."""
        schema_path = Path("contracts/legal_rag_bench_query_output.schema.json")

        output_with_metadata = {
            "schema_version": "1.0.0",
            "system_version": {
                "pipeline_version": "0.1.0",
            },
            "query": {
                "query_id": "q001",
                "text": "Question?",
                "metadata": {
                    "category": "contract_law",
                    "difficulty": "easy",
                },
            },
            "answer": {
                "text": "Answer",
                "citations": [],
            },
            "retrieved_chunks": [],
            "relevant_passage_id": "passage_123",
        }

        # Should not raise
        validate(output_with_metadata, schema_path)
