"""
Tests for semantic retrieval and ingestion.

These tests verify the SemanticRetriever and DocumentIngester
components of the query and retrieval pipeline.
"""

import pytest

from eval_harness.stubs.rag.chromadb_client import ChromaDBManager
from eval_harness.stubs.rag.chunker import FixedChunker
from eval_harness.stubs.rag.embedder import SentenceTransformersEmbedder
from eval_harness.stubs.rag.ingestion import DocumentIngester
from eval_harness.stubs.rag.retriever import SemanticRetriever


@pytest.fixture
def sample_collection():
    """Create a sample ChromaDB collection for testing."""
    manager = ChromaDBManager(persist=False)
    collection = manager.get_or_create_collection("test_collection")

    # Add some sample documents
    collection.add(
        documents=["The quick brown fox jumps over the lazy dog."],
        ids=["doc1_chunk_00000"],
        metadatas=[
            {
                "doc_id": "doc1",
                "char_span": [0, 45],
                "source_file": "doc1.txt",
            }
        ],
    )

    collection.add(
        documents=["Python is a high-level programming language."],
        ids=["doc2_chunk_00000"],
        metadatas=[
            {
                "doc_id": "doc2",
                "char_span": [0, 44],
                "source_file": "doc2.txt",
            }
        ],
    )

    yield collection

    # Cleanup
    manager.close()


@pytest.fixture
def retriever(sample_collection):
    """Create a SemanticRetriever for testing."""
    embedder = SentenceTransformersEmbedder()
    return SemanticRetriever(sample_collection, embedder)


class TestSemanticRetriever:
    """Test suite for SemanticRetriever."""

    def test_top_k_retrieval_returns_correct_number(self, retriever):
        """Test top-k retrieval returns correct number of chunks."""
        results = retriever.retrieve("fox", top_k=2)
        assert len(results) <= 2

    def test_results_ranked_by_score_descending(self, retriever):
        """Test results are ranked by score (descending)."""
        results = retriever.retrieve("what", top_k=5)

        # Check that scores are in descending order
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    def test_rank_field_starts_at_zero(self, retriever):
        """Test that rank field starts at 0 for highest score."""
        results = retriever.retrieve("test query", top_k=5)

        if results:
            assert results[0]["rank"] == 0
            for i, result in enumerate(results):
                assert result["rank"] == i

    def test_retrieval_stage_is_initial(self, retriever):
        """Test retrieval_stage='initial' in results."""
        results = retriever.retrieve("test", top_k=1)

        for result in results:
            assert result["retrieval_stage"] == "initial"

    def test_all_required_schema_fields_present(self, retriever):
        """Test all required schema fields are present in results."""
        results = retriever.retrieve("test", top_k=1)

        required_fields = [
            "chunk_id",
            "rank",
            "score",
            "retrieval_stage",
            "doc_id",
            "text",
            "char_span",
        ]

        for result in results:
            for field in required_fields:
                assert field in result

    def test_empty_query_returns_empty_results(self, retriever):
        """Test empty query returns empty results."""
        results = retriever.retrieve("", top_k=5)
        assert results == []

    def test_top_k_less_than_one_raises_error(self, retriever):
        """Test top_k < 1 raises ValueError."""
        with pytest.raises(ValueError):
            retriever.retrieve("test", top_k=0)


class TestDocumentIngester:
    """Test suite for DocumentIngester."""

    def test_ingest_creates_correct_chunks(self, tmp_path):
        """Test ingestion creates correct number of chunks."""
        # Create test documents
        (tmp_path / "doc1.txt").write_text("a" * 600)  # 2 chunks
        (tmp_path / "doc2.txt").write_text("b" * 600)  # 2 chunks

        # Setup ingestion
        manager = ChromaDBManager(persist=False)
        collection = manager.get_or_create_collection("test_ingest")
        chunker = FixedChunker()
        embedder = SentenceTransformersEmbedder()
        ingester = DocumentIngester(chunker, embedder)

        # Ingest
        stats = ingester.ingest_corpus(tmp_path, collection)

        # Verify
        assert stats["documents_processed"] == 2
        assert stats["chunks_created"] == 4
        assert stats["errors"] == 0

        manager.close()

    def test_ingest_empty_directory(self, tmp_path):
        """Test ingestion of empty directory returns zeros."""
        manager = ChromaDBManager(persist=False)
        collection = manager.get_or_create_collection("test_empty")
        chunker = FixedChunker()
        embedder = SentenceTransformersEmbedder()
        ingester = DocumentIngester(chunker, embedder)

        stats = ingester.ingest_corpus(tmp_path, collection)

        assert stats["documents_processed"] == 0
        assert stats["chunks_created"] == 0

        manager.close()

    def test_ingest_with_unicode_documents(self, tmp_path):
        """Test ingestion handles unicode documents."""
        (tmp_path / "unicode.txt").write_text("Hello 世界 🌍 " * 100)

        manager = ChromaDBManager(persist=False)
        collection = manager.get_or_create_collection("test_unicode")
        chunker = FixedChunker()
        embedder = SentenceTransformersEmbedder()
        ingester = DocumentIngester(chunker, embedder)

        stats = ingester.ingest_corpus(tmp_path, collection)

        assert stats["documents_processed"] == 1
        assert stats["chunks_created"] > 0
        assert stats["errors"] == 0

        manager.close()
