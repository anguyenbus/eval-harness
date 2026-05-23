"""Tests for configurable chunking strategies."""

import pytest

from eval_harness.stubs.rag.chunking import ChunkingStrategy, ConfigurableChunker


class TestChunkingStrategy:
    """Test suite for configurable chunking strategy."""

    def test_chunk_size_parameter_produces_correct_boundaries(self):
        """Test that chunk_size parameter creates correct chunk boundaries."""
        chunker = ConfigurableChunker(chunk_size=100, chunk_overlap=0)
        text = "a" * 250

        chunks = chunker.chunk(doc_id="test", text=text)

        assert len(chunks) == 3
        assert chunks[0]["char_span"] == [0, 100]
        assert chunks[1]["char_span"] == [100, 200]
        assert chunks[2]["char_span"] == [200, 250]

    def test_chunk_overlap_parameter_produces_overlapping_chunks(self):
        """Test that chunk_overlap creates overlapping chunks correctly."""
        chunker = ConfigurableChunker(chunk_size=100, chunk_overlap=50)
        text = "a" * 200

        chunks = chunker.chunk(doc_id="test", text=text)

        assert len(chunks) == 3
        # First chunk: [0, 100]
        assert chunks[0]["char_span"] == [0, 100]
        # Second chunk starts at 50 (100 - 50 overlap)
        assert chunks[1]["char_span"] == [50, 150]
        # Third chunk starts at 100
        assert chunks[2]["char_span"] == [100, 200]

    def test_overlap_equals_chunk_size_no_infinite_loop(self):
        """Test edge case where overlap equals chunk size doesn't infinite loop."""
        chunker = ConfigurableChunker(chunk_size=100, chunk_overlap=100)
        text = "a" * 250

        chunks = chunker.chunk(doc_id="test", text=text)

        # Should advance by at least 1 character to avoid infinite loop
        assert len(chunks) > 1
        assert len(chunks) <= 3

    def test_validates_parameters(self):
        """Test that invalid parameters are rejected."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ConfigurableChunker(chunk_size=0, chunk_overlap=0)

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            ConfigurableChunker(chunk_size=100, chunk_overlap=150)
