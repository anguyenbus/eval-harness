"""
Tests for answer generation and citations.

These tests verify the ClaudeGenerator and citation extraction
components of the answer generation pipeline.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from eval_harness.stubs.rag.citations import extract_citations
from eval_harness.stubs.rag.generator import ClaudeGenerator


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client for testing."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is the answer [doc1_chunk_00000].")]

    mock_client = MagicMock()
    mock_client.messages.create = MagicMock(return_value=mock_response)

    with patch("anthropic.Anthropic", return_value=mock_client) as mock:
        yield mock


@pytest.fixture
def sample_chunks():
    """Sample retrieved chunks for testing."""
    return [
        {
            "chunk_id": "doc1_chunk_00000",
            "rank": 0,
            "score": 0.95,
            "text": "This is test content.",
            "doc_id": "doc1",
            "char_span": [0, 20],
        },
        {
            "chunk_id": "doc1_chunk_00001",
            "rank": 1,
            "score": 0.85,
            "text": "More test content.",
            "doc_id": "doc1",
            "char_span": [20, 40],
        },
    ]


class TestCitationExtraction:
    """Test suite for citation extraction."""

    def test_simple_sentence_to_chunk_citation_extraction(self):
        """Test simple sentence-to-chunk citation extraction."""
        answer = "The answer is [doc1_chunk_00000]."
        chunks = [{"chunk_id": "doc1_chunk_00000", "text": "test"}]

        citations = extract_citations(answer, chunks)

        assert len(citations) == 1
        assert citations[0]["chunk_ids"] == ["doc1_chunk_00000"]

    def test_citations_reference_valid_chunk_ids(self):
        """Test citations reference valid chunk_ids only."""
        answer = "The answer is [doc1_chunk_00000] and [invalid_chunk]."
        chunks = [{"chunk_id": "doc1_chunk_00000", "text": "test"}]

        citations = extract_citations(answer, chunks)

        assert len(citations) == 1
        assert citations[0]["chunk_ids"] == ["doc1_chunk_00000"]

    def test_citations_with_multiple_references(self):
        """Test citations with multiple chunk_id references."""
        answer = "First claim [doc1_chunk_00000]. Second claim [doc1_chunk_00001]."
        chunks = [
            {"chunk_id": "doc1_chunk_00000", "text": "test1"},
            {"chunk_id": "doc1_chunk_00001", "text": "test2"},
        ]

        citations = extract_citations(answer, chunks)

        assert len(citations) == 2
        assert citations[0]["chunk_ids"] == ["doc1_chunk_00000"]
        assert citations[1]["chunk_ids"] == ["doc1_chunk_00001"]

    def test_claim_span_offsets(self):
        """Test claim_span character offsets are correct."""
        answer = "The answer is [doc1_chunk_00000]."
        chunks = [{"chunk_id": "doc1_chunk_00000", "text": "test"}]

        citations = extract_citations(answer, chunks)

        assert len(citations) == 1
        start, end = citations[0]["claim_span"]
        assert start >= 0
        assert end <= len(answer)
        assert answer[start:end] == "The answer is [doc1_chunk_00000]."

    def test_empty_answer_returns_empty_citations(self):
        """Test empty answer returns empty citations."""
        citations = extract_citations("", [])
        assert citations == []

    def test_no_chunk_references_returns_empty_citations(self):
        """Test answer without chunk references returns empty citations."""
        answer = "This answer has no citations."
        citations = extract_citations(answer, [])
        assert citations == []


class TestClaudeGenerator:
    """Test suite for ClaudeGenerator."""

    def test_context_augmented_prompt_construction(self, mock_anthropic, sample_chunks):
        """Test context-augmented prompt construction."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            generator = ClaudeGenerator()

            result = generator.generate("What is this?", sample_chunks)

            assert "text" in result
            assert "answer_supported" in result
            assert "timings_ms" in result

    def test_answer_supported_parsing(self, sample_chunks):
        """Test answer_supported is parsed correctly."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I don't have enough information.")]

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)

        with patch("anthropic.Anthropic", return_value=mock_client):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                generator = ClaudeGenerator()
                result = generator.generate("What is this?", sample_chunks)

                assert result["answer_supported"] is False

    def test_timing_instrumentation(self, mock_anthropic, sample_chunks):
        """Test timing instrumentation is captured."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            generator = ClaudeGenerator()
            result = generator.generate("What is this?", sample_chunks)

            assert "generation" in result["timings_ms"]
            assert result["timings_ms"]["generation"] > 0

    def test_missing_api_key_raises_error(self):
        """Test missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                ClaudeGenerator()

    def test_api_failure_handling(self, sample_chunks):
        """Test API failure is handled properly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client:
                mock_client.side_effect = Exception("API Error")

                generator = ClaudeGenerator()

                with pytest.raises(ValueError, match="API call failed"):
                    generator.generate("What is this?", sample_chunks)
