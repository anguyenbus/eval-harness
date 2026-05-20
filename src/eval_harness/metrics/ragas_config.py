"""
RAGAS configuration for LLM-judge metrics.

This module provides configuration for RAGAS evaluation metrics including
LLM backend setup for OpenAI (primary) and AWS Bedrock (future support).

Metrics supported:
- Faithfulness: Detects hallucinations in generated answers
- ContextPrecision: Measures signal-to-noise in retrieved contexts
- ContextRecall: Evaluates coverage of relevant information
- AnswerRelevancy: Assesses directness of response to question (requires embeddings)
"""

from __future__ import annotations

import os
from typing import Any, Final

from beartype import beartype
from beartype.typing import Dict
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Constants
OPENAI_API_KEY_ENV: Final[str] = "OPENAI_API_KEY"
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o"
DEFAULT_TEMPERATURE: Final[float] = 0.0


@beartype
def _get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment.

    Returns:
        OpenAI API key string.

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set.

    """
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f"{OPENAI_API_KEY_ENV} environment variable must be set to use RAGAS metrics. "
            "Set it with: export OPENAI_API_KEY=your-key-here"
        )
    return api_key


@beartype
def get_openai_client(model: str = DEFAULT_OPENAI_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> ChatOpenAI:
    """
    Get OpenAI client for RAGAS evaluation.

    Args:
        model: OpenAI model name. Default: gpt-4o.
        temperature: Sampling temperature. Default: 0.0.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set.

    """
    api_key = _get_openai_api_key()

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
    )


@beartype
def get_llm_backend(
    provider: str = "openai",
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Any:
    """
    Get LLM backend for RAGAS evaluation.

    Args:
        provider: LLM provider name ("openai" or "bedrock"). Default: "openai".
        model: Model name. Default: gpt-4o.
        temperature: Sampling temperature. Default: 0.0.

    Returns:
        Wrapped LLM instance compatible with RAGAS.

    Raises:
        ValueError: If provider is not supported or API key is missing.

    """
    if provider == "openai":
        from ragas.llms import llm_factory
        from openai import AsyncOpenAI

        api_key = _get_openai_api_key()
        # Use AsyncOpenAI for async evaluation (required by RAGAS 0.4+)
        client = AsyncOpenAI(api_key=api_key)
        return llm_factory(model=model, client=client)
    elif provider == "bedrock":
        raise NotImplementedError(
            "Bedrock support is planned but not yet implemented. "
            "Use 'openai' provider for now."
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'bedrock'.")


@beartype
def get_embeddings_backend(
    provider: str = "huggingface",
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Any:
    """
    Get embeddings backend for RAGAS evaluation.

    Args:
        provider: Embeddings provider ("huggingface" or "openai"). Default: "huggingface".
        model: Embedding model name. Default: sentence-transformers/all-MiniLM-L6-v2.

    Returns:
        Wrapped embeddings instance compatible with RAGAS.

    Raises:
        ValueError: If provider is not supported.

    """
    if provider == "huggingface":
        from ragas.embeddings import HuggingFaceEmbeddings

        # Force CPU to avoid CUDA errors on unsupported GPUs
        return HuggingFaceEmbeddings(model=model, device="cpu")
    elif provider == "openai":
        from ragas.embeddings import embedding_factory

        api_key = _get_openai_api_key()
        return embedding_factory(
            model=model,
            provider="openai",
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'huggingface' or 'openai'.")


@beartype
def create_ragas_metrics(
    llm_provider: str = "openai",
    judge_model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    embeddings_provider: str = "huggingface",
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Create RAGAS metrics with configured LLM backend.

    Args:
        llm_provider: LLM provider ("openai" or "bedrock"). Default: "openai".
        judge_model: Judge model name. Default: gpt-4o.
        temperature: Sampling temperature. Default: 0.0.
        embeddings_provider: Embeddings provider ("huggingface" or "openai"). Default: "huggingface".
        embeddings_model: Embeddings model name. Default: sentence-transformers/all-MiniLM-L6-v2.

    Returns:
        Dictionary mapping metric names to instantiated RAGAS metric objects.

    Raises:
        ValueError: If provider is not supported or API key is missing.

    """
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    # Get LLM backend
    llm = get_llm_backend(
        provider=llm_provider,
        model=judge_model,
        temperature=temperature,
    )

    # Get embeddings backend (required for AnswerRelevancy)
    embeddings = get_embeddings_backend(
        provider=embeddings_provider,
        model=embeddings_model,
    )

    # Create metrics with LLM backend
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    }
