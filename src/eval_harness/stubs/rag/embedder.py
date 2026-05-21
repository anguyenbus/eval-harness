"""
SentenceTransformers embedding generation for RAG.

NOTE: Reference stub implementation for demonstration purposes.
Not intended for production use. Provides SentenceTransformersEmbedder
class which generates embeddings using the SentenceTransformers library.
"""

from __future__ import annotations

from beartype import beartype

from eval_harness.stubs.rag.chromadb_config import EMBEDDING_DIM, EMBEDDING_MODEL
from eval_harness.stubs.rag.exceptions import EmbeddingError


@beartype
class SentenceTransformersEmbedder:
    """
    SentenceTransformers embedding generator.

    NOTE: This is a reference stub implementation for demonstration purposes.
    It is not intended for production use.

    The embedder uses the all-MiniLM-L6-v2 model to generate 384-dimensional
    embeddings for text chunks. The model is loaded lazily on first use and
    cached for subsequent calls. Uses CPU device for compatibility.

    Attributes:
        _model_name: Name of the SentenceTransformers model.
        _dimension: Output embedding dimension (384).
        _model: Cached model instance (loaded on first use).
        _device: Device to use for inference (cpu).

    Example:
        >>> embedder = SentenceTransformersEmbedder()
        >>> embeddings = embedder.embed(["text 1", "text 2"])
        >>> len(embeddings)
        2
        >>> len(embeddings[0])
        384

    """

    __slots__ = ("_model_name", "_dimension", "_model", "_device")

    def __init__(self) -> None:
        """Initialize the embedder with configured model (lazy loading)."""
        self._model_name: str = EMBEDDING_MODEL
        self._dimension: int = EMBEDDING_DIM
        self._model: object | None = None
        self._device: str = "cpu"

    def _load_model(self) -> object:
        """
        Load the SentenceTransformers model (lazy loading).

        Returns:
            Loaded SentenceTransformer model instance.

        Raises:
            EmbeddingError: If model loading fails.

        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )

            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self._model_name}': {e}"
                ) from e

        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, where each vector is a list of floats
            with dimension equal to EMBEDDING_DIM (384).

        Raises:
            EmbeddingError: If embedding generation fails.

        """
        if not texts:
            return []

        try:
            model = self._load_model()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # Convert to list of lists
            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector as a list of floats with dimension equal to
            EMBEDDING_DIM (384).

        Raises:
            EmbeddingError: If embedding generation fails.

        """
        if not text:
            raise EmbeddingError("Cannot embed empty text")

        embeddings = self.embed([text])
        return embeddings[0]
