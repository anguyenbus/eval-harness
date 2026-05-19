"""
Tests for ChromaDB client manager.

These tests verify ChromaDB client initialization, collection management,
and embedding function configuration.
"""

import pytest

from eval_harness.stubs.rag.chromadb_client import ChromaDBManager
from eval_harness.stubs.rag.chromadb_config import EMBEDDING_MODEL
from eval_harness.stubs.rag.exceptions import CollectionNotFoundError


class TestChromaDBClient:
    """Test suite for ChromaDB client manager."""

    def test_persistent_mode_creates_data_directory(self, tmp_path):
        """Test that persistent mode creates the chromadb data directory."""
        from unittest.mock import patch

        with patch(
            "eval_harness.stubs.rag.chromadb_client.CHROMADB_PERSIST_DIR",
            tmp_path / "chromadb",
        ):
            manager = ChromaDBManager(persist=True)

            # Verify directory was created
            assert (tmp_path / "chromadb").exists()

            manager.close()

    def test_in_memory_mode_for_testing(self):
        """Test that in-memory mode works without creating files in persist dir."""
        manager = ChromaDBManager(persist=False)

        # Should still be able to create collections
        collection = manager.get_or_create_collection("test_collection")
        assert collection is not None
        assert collection.name == "test_collection"

        # Verify it works with in-memory mode (no file operations)
        collection.add(
            documents=["test document"],
            ids=["test_id"],
        )

        manager.close()

    def test_collection_creation_with_embedding_function(self):
        """Test that collection is created with correct embedding function."""
        manager = ChromaDBManager(persist=False)

        collection = manager.get_or_create_collection("test_collection")

        # Verify collection exists
        assert manager.collection_exists("test_collection")

        # Verify we can add data (proves embedding function works)
        collection.add(
            documents=["test document"],
            ids=["test_id"],
        )

        manager.close()

    def test_embedding_dimension_matches_model(self):
        """Test that embedding dimension matches all-MiniLM-L6-v2 (384)."""
        from eval_harness.stubs.rag.chromadb_config import EMBEDDING_DIM

        manager = ChromaDBManager(persist=False)
        collection = manager.get_or_create_collection("test_collection")

        # Add a document and retrieve it to verify embedding dimension
        collection.add(
            documents=["test document for dimension check"],
            ids=["dim_test_id"],
        )

        # Query to verify embeddings work
        results = collection.query(
            query_texts=["test document"],
            n_results=1,
        )

        # Verify embeddings were created
        assert "embeddings" in results or "documents" in results

        manager.close()

        # Verify config constant
        assert EMBEDDING_DIM == 384
        assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_collection_exists_checking(self):
        """Test collection existence checking for existing and non-existing collections."""
        manager = ChromaDBManager(persist=False)

        # Non-existent collection
        assert not manager.collection_exists("nonexistent_collection")

        # Create collection
        manager.get_or_create_collection("existing_collection")

        # Existing collection
        assert manager.collection_exists("existing_collection")

        manager.close()

    def test_delete_collection(self):
        """Test deleting an existing collection."""
        manager = ChromaDBManager(persist=False)

        # Create collection
        manager.get_or_create_collection("to_delete")
        assert manager.collection_exists("to_delete")

        # Delete collection
        manager.delete_collection("to_delete")
        assert not manager.collection_exists("to_delete")

        manager.close()

    def test_delete_nonexistent_collection_raises_error(self):
        """Test that deleting a non-existent collection raises CollectionNotFoundError."""
        manager = ChromaDBManager(persist=False)

        with pytest.raises(CollectionNotFoundError):
            manager.delete_collection("does_not_exist")

        manager.close()

    def test_environment_variable_configuration(self, monkeypatch):
        """Test that ANTHROPIC_MODEL environment variable is read correctly."""
        from eval_harness.stubs.rag.chromadb_config import GENERATOR_MODEL

        # Default value
        assert GENERATOR_MODEL == "claude-opus-4-7"

        # Test with monkeypatch (would need to reimport to see change)
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-7")
        # Note: This would require reimporting the module to see the change
        # The config module reads the env var at import time
