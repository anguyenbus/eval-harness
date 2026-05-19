"""
ChromaDB-backed RAG implementation.

NOTE: This is a reference stub implementation provided for demonstration purposes.
It is not intended for production use.

This package provides a complete RAG pipeline using ChromaDB for vector storage,
SentenceTransformers for embeddings, and OpenAI for answer generation.

WARNING: This stub is for demonstration and testing purposes only.
Do not use this implementation in production environments.
"""

from eval_harness.stubs.rag.chromadb_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CORPUS_LOADER_VERSION,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    GENERATOR_MODEL,
    PIPELINE_VERSION,
)
from eval_harness.stubs.rag.exceptions import (
    ChromaDBInitError,
    CollectionNotFoundError,
    EmbeddingError,
)

__all__ = [
    "CHROMADB_PERSIST_DIR",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "GENERATOR_MODEL",
    "PIPELINE_VERSION",
    "CORPUS_LOADER_VERSION",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "ChromaDBInitError",
    "CollectionNotFoundError",
    "EmbeddingError",
]
