"""
ChromaDB RAG configuration constants.

NOTE: Reference stub implementation for demonstration purposes.
Not intended for production use. Defines all configuration constants
for the ChromaDB-backed RAG pipeline, including model names and dimensions.
"""

import os
from pathlib import Path
from typing import Final

# ChromaDB storage configuration
CHROMADB_PERSIST_DIR: Final[Path] = Path("data/chromadb/")

# Embedding model configuration
EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: Final[int] = 384

# Generator model configuration (set via RAG_GENERATOR_MODEL env var)
# Supports: openai models (gpt-4o, gpt-4o-mini), bedrock models, etc.
GENERATOR_MODEL: Final[str] = os.getenv("RAG_GENERATOR_MODEL", "gpt-4o")

# Pipeline version tracking
PIPELINE_VERSION: Final[str] = "0.1.0-chromadb"
CORPUS_LOADER_VERSION: Final[str] = "0.1.0"

# Chunking configuration
CHUNK_SIZE: Final[int] = 512
CHUNK_OVERLAP: Final[int] = 0

# Default retrieval configuration
DEFAULT_TOP_K: Final[int] = 5

# Batch processing configuration
BATCH_SIZE: Final[int] = 100
