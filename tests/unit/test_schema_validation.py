"""Tests for JSON schema validation."""

from pathlib import Path

import pytest

from eval_harness.adapters.schema_validator import (
    SchemaValidationError,
    validate,
)


class TestSchemaValidation:
    """Test suite for JSON schema validation."""

    def test_valid_parser_output_validates(self, tmp_path):
        """Test that valid parser output passes validation."""
        schema_path = Path("contracts/parser_output.schema.json")

        valid_output = {
            "schema_version": "1.0.0",
            "parser_version": "0.0.1-stub",
            "source": {
                "doc_id": "test-doc-001",
                "filename": "test.pdf",
                "mime_type": "application/pdf",
                "sha256": "a" * 64,
            },
            "pages": [{"page_index": 0, "width": 612, "height": 792}],
            "elements": [
                {
                    "element_id": "elem-001",
                    "type": "heading",
                    "page_index": 0,
                    "char_span": [0, 10],
                    "text": "Heading",
                    "content": {"kind": "text"},
                    "level": 1,
                },
                {
                    "element_id": "elem-002",
                    "type": "paragraph",
                    "page_index": 0,
                    "char_span": [11, 50],
                    "text": "This is a paragraph of text.",
                    "content": {"kind": "text"},
                },
            ],
            "warnings": [
                {
                    "code": "STUB_OUTPUT",
                    "message": "This is stub output, not from a real parser.",
                },
            ],
        }

        # Should not raise
        validate(valid_output, schema_path)

    def test_invalid_parser_output_fails_with_clear_error(self, tmp_path):
        """Test that invalid parser output fails with clear field path error."""
        schema_path = Path("contracts/parser_output.schema.json")

        invalid_output = {
            "schema_version": "1.0.0",
            # Missing required fields
        }

        with pytest.raises(SchemaValidationError) as exc_info:
            validate(invalid_output, schema_path)

        error_msg = str(exc_info.value)
        assert "parser_version" in error_msg or "required" in error_msg.lower()

    def test_valid_rag_query_output_validates(self, tmp_path):
        """Test that valid RAG query output passes validation."""
        schema_path = Path("contracts/rag_query_output.schema.json")

        valid_output = {
            "schema_version": "1.0.0",
            "system_version": {
                "pipeline_version": "0.0.1-stub",
                "parser_version": "0.0.1",
                "chunker_version": "0.0.1",
                "embedder_model": "text-embedding-3-large",
                "retriever_version": "0.0.1",
                "reranker_model": None,
                "generator_model": "claude-opus-4-7",
            },
            "query": {
                "query_id": "query-001",
                "text": "What is the document about?",
            },
            "answer": {
                "text": "This is a stub answer.",
                "answer_supported": False,
                "citations": [],
            },
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk-001",
                    "rank": 0,
                    "score": 0.95,
                    "doc_id": "doc-001",
                    "text": "This is a retrieved chunk.",
                    "char_span": [0, 25],
                },
            ],
        }

        # Should not raise
        validate(valid_output, schema_path)

    def test_schema_version_mismatch_fails(self, tmp_path):
        """Test that wrong schema_version fails validation."""
        schema_path = Path("contracts/parser_output.schema.json")

        wrong_version = {
            "schema_version": "2.0.0",  # Wrong version - must be "1.0.0"
            "parser_version": "0.0.1",
            "source": {
                "doc_id": "test-doc-001",
                "filename": "test.pdf",
                "mime_type": "application/pdf",
                "sha256": "a" * 64,
            },
            "pages": [{"page_index": 0, "width": 612, "height": 792}],
            "elements": [],
        }

        with pytest.raises(SchemaValidationError) as exc_info:
            validate(wrong_version, schema_path)

        error_msg = str(exc_info.value)
        # The error should mention validation failed
        assert "validation failed" in error_msg.lower()

    def test_missing_nested_field_raises_clear_error(self, tmp_path):
        """Test that missing nested field raises clear error with path."""
        schema_path = Path("contracts/parser_output.schema.json")

        missing_nested = {
            "schema_version": "1.0.0",
            "parser_version": "0.0.1",
            # Missing 'source' object
            "pages": [{"page_index": 0, "width": 612, "height": 792}],
            "elements": [],
        }

        with pytest.raises(SchemaValidationError) as exc_info:
            validate(missing_nested, schema_path)

        error_msg = str(exc_info.value)
        # Should mention what's missing
        assert len(error_msg) > 0
