"""
Tests for configurable chunking strategy.

Focused tests for Task Group 1: Configurable Chunking Strategy.
"""


import pytest

from eval_harness.stubs.rag.chunking import ConfigurableChunker


class TestConfigurableChunker:
    """Test ConfigurableChunker with parameterized chunking."""

    def test_chunk_size_parameter_produces_correct_boundaries(self) -> None:
        """Test that chunk_size parameter produces correct chunk boundaries."""
        chunker = ConfigurableChunker(chunk_size=100, chunk_overlap=0)
        text = "a" * 250
        chunks = chunker.chunk(doc_id="test", text=text)

        assert len(chunks) == 3
        assert chunks[0]["char_span"] == [0, 100]
        assert chunks[1]["char_span"] == [100, 200]
        assert chunks[2]["char_span"] == [200, 250]

    def test_chunk_overlap_parameter_produces_overlapping_chunks(self) -> None:
        """Test that chunk_overlap parameter produces overlapping chunks."""
        chunker = ConfigurableChunker(chunk_size=100, chunk_overlap=25)
        text = "a" * 250
        chunks = chunker.chunk(doc_id="test", text=text)

        assert len(chunks) == 4
        assert chunks[0]["char_span"] == [0, 100]
        assert chunks[1]["char_span"] == [75, 175]  # Overlaps by 25
        assert chunks[2]["char_span"] == [150, 250]

    def test_overlap_equals_chunk_size_no_infinite_loop(self) -> None:
        """Test edge case: overlap equals chunk_size (no infinite loop)."""
        chunker = ConfigurableChunker(chunk_size=100, chunk_overlap=100)
        text = "a" * 250
        chunks = chunker.chunk(doc_id="test", text=text)

        # Should advance when overlap equals chunk size
        assert len(chunks) == 3

    def test_invalid_chunk_size_raises_error(self) -> None:
        """Test that invalid chunk_size raises validation error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ConfigurableChunker(chunk_size=0, chunk_overlap=0)

    def test_invalid_overlap_raises_error(self) -> None:
        """Test that invalid overlap raises validation error."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            ConfigurableChunker(chunk_size=100, chunk_overlap=100)

    def test_chunking_is_deterministic(self) -> None:
        """Test that chunking produces deterministic output."""
        chunker = ConfigurableChunker(chunk_size=150, chunk_overlap=50)
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks1 = chunker.chunk(doc_id="test", text=text)
        chunks2 = chunker.chunk(doc_id="test", text=text)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=True):
            assert c1["char_span"] == c2["char_span"]
            assert c1["text"] == c2["text"]
