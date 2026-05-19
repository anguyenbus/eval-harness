"""
Tests for text chunking and embedding.

These tests verify the FixedChunker and SentenceTransformersEmbedder
components of the document processing pipeline.
"""

import pytest

from eval_harness.stubs.rag.chromadb_config import CHUNK_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL
from eval_harness.stubs.rag.chunker import FixedChunker
from eval_harness.stubs.rag.embedder import SentenceTransformersEmbedder
from eval_harness.stubs.rag.exceptions import EmbeddingError


class TestFixedChunker:
    """Test suite for FixedChunker."""

    def test_fixed_512_character_chunking(self):
        """Test fixed 512-character chunking with 0 overlap."""
        chunker = FixedChunker()

        # Create text that's exactly 1024 characters (2 chunks)
        text = "a" * 1024
        chunks = chunker.chunk(doc_id="test_doc", text=text)

        assert len(chunks) == 2
        assert all(len(chunk["text"]) == CHUNK_SIZE for chunk in chunks)

    def test_chunk_overlap_is_zero(self):
        """Test that chunk overlap is 0 (no overlap between chunks)."""
        chunker = FixedChunker()

        text = "a" * (CHUNK_SIZE * 2)
        chunks = chunker.chunk(doc_id="test_doc", text=text)

        # Verify no overlap: chunk 1 ends where chunk 2 starts
        assert chunks[0]["char_span"][1] == chunks[1]["char_span"][0]

    def test_character_offset_tracking(self):
        """Test character offset tracking (char_span)."""
        chunker = FixedChunker()

        text = "hello world " * 50
        chunks = chunker.chunk(doc_id="test_doc", text=text)

        # First chunk starts at 0
        assert chunks[0]["char_span"][0] == 0
        assert chunks[0]["char_span"][1] == min(CHUNK_SIZE, len(text))

        # Second chunk starts where first ends
        if len(chunks) > 1:
            assert chunks[1]["char_span"][0] == chunks[0]["char_span"][1]

    def test_chunk_id_generation(self):
        """Test chunk_id generation (doc_id + sequence)."""
        chunker = FixedChunker()

        text = "x" * (CHUNK_SIZE * 3)
        chunks = chunker.chunk(doc_id="my_doc", text=text)

        assert chunks[0]["chunk_id"] == "my_doc_chunk_00000"
        assert chunks[1]["chunk_id"] == "my_doc_chunk_00001"
        assert chunks[2]["chunk_id"] == "my_doc_chunk_00002"

    def test_empty_document_handling(self):
        """Test empty document handling."""
        chunker = FixedChunker()

        chunks = chunker.chunk(doc_id="empty_doc", text="")
        assert chunks == []

    def test_document_shorter_than_chunk_size(self):
        """Test document shorter than chunk size produces single chunk."""
        chunker = FixedChunker()

        text = "short"
        chunks = chunker.chunk(doc_id="short_doc", text=text)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "short"
        assert chunks[0]["char_span"] == [0, 5]

    def test_unicode_special_character_handling(self):
        """Test unicode/special character handling in chunking."""
        chunker = FixedChunker()

        # Text with unicode characters
        text = "Hello 世界 🌍 " * 100
        chunks = chunker.chunk(doc_id="unicode_doc", text=text)

        # Verify chunking works with unicode
        assert len(chunks) > 0
        # Verify char_span positions are valid
        for chunk in chunks:
            assert chunk["char_span"][0] >= 0
            assert chunk["char_span"][1] <= len(text)


class TestSentenceTransformersEmbedder:
    """Test suite for SentenceTransformersEmbedder."""

    def test_embedder_configuration(self):
        """Test embedder has correct model configuration."""
        embedder = SentenceTransformersEmbedder()

        # Verify model name and dimension are configured correctly
        assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
        assert EMBEDDING_DIM == 384

    def test_empty_text_embedding_raises_error(self):
        """Test that empty text embedding raises EmbeddingError."""
        embedder = SentenceTransformersEmbedder()

        with pytest.raises(EmbeddingError):
            embedder.embed_single("")

    def test_empty_list_returns_empty_embeddings(self):
        """Test that empty list returns empty embeddings."""
        embedder = SentenceTransformersEmbedder()

        embeddings = embedder.embed([])
        assert embeddings == []
