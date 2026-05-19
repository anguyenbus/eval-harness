"""
OpenAI-powered answer generation for RAG.

NOTE: This is a reference stub implementation provided for demonstration purposes.
It is not intended for production use. This module provides the OpenAIGenerator class
which generates answers using OpenAI models with retrieved context.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from beartype import beartype

from eval_harness.stubs.rag.chromadb_config import GENERATOR_MODEL

# Load .env file if it exists
_env_path = Path.cwd() / ".env"
if _env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_path)


@beartype
class OpenAIGenerator:
    """
    OpenAI-powered answer generator for RAG.

    NOTE: This is a reference stub implementation for demonstration purposes.
    It is not intended for production use.

    The OpenAIGenerator uses OpenAI models to generate answers based on
    retrieved context chunks. It constructs context-augmented prompts and
    instructs the model to cite sources using chunk_ids.

    Attributes:
        _model: OpenAI model identifier (e.g., gpt-4o, gpt-4o-mini).
        _api_key: OpenAI API key (from OPENAI_API_KEY env var).

    Example:
        >>> generator = OpenAIGenerator()
        >>> answer = generator.generate(
        ...     question="What is this?",
        ...     retrieved_chunks=[...]
        ... )
        >>> print(answer["text"])

    """

    __slots__ = ("_model", "_api_key")

    def __init__(self, model: str | None = None) -> None:
        """
        Initialize OpenAI generator.

        Args:
            model: OpenAI model identifier. If None, uses GPT-4O-mini (fast, cheap).

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set.

        """
        self._model: str = model if model is not None else "gpt-4o"
        self._api_key: str | None = os.getenv("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI generation"
            )

    def generate(
        self, question: str, retrieved_chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Generate answer using OpenAI with retrieved context.

        Args:
            question: User question to answer.
            retrieved_chunks: List of retrieved chunks with chunk_id, text, etc.

        Returns:
            Dictionary containing:
                - text: Generated answer text
                - answer_supported: Whether corpus supports the answer
                - citations: List of citation dictionaries
                - timings_ms: Timing information

        Raises:
            ValueError: If API call fails or returns invalid response.

        """
        start_time = time.perf_counter()

        # Construct context-augmented prompt
        context_parts = []
        for chunk in retrieved_chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            text = chunk.get("text", "")
            context_parts.append(f"[{chunk_id}]: {text}")

        context = "\n\n".join(context_parts)

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
When answering, you MUST cite your sources using the chunk_ids in square brackets like [chunk_id].
For example: "The answer is [doc1_chunk_00000]."

If the context doesn't contain enough information to answer the question confidently, say "I don't have enough information to answer this question."
Set answer_supported to false in this case."""

        user_message = f"""Context:
{context}

Question: {question}

Answer:"""

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)

            response = client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )

            answer_text = response.choices[0].message.content or ""

            # Determine if answer is supported
            answer_supported = "I don't have enough information" not in answer_text

            generation_time = (time.perf_counter() - start_time) * 1000

            return {
                "text": answer_text,
                "answer_supported": answer_supported,
                "citations": [],  # Will be extracted by extract_citations
                "timings_ms": {
                    "generation": generation_time,
                },
            }

        except Exception as e:
            raise ValueError(f"OpenAI API call failed: {e}") from e


# Alias for backward compatibility with imports
ClaudeGenerator = OpenAIGenerator
