"""Tests for RAGAS configuration module."""

from unittest.mock import patch

import pytest

from eval_harness.metrics.ragas_config import (
    create_ragas_metrics,
    get_llm_backend,
    get_openai_client,
)


class TestRagasConfig:
    """Test suite for RAGAS configuration."""

    def test_ragas_imports_available(self):
        """Test that RAGAS imports are available."""
        from ragas import SingleTurnSample
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        assert SingleTurnSample is not None
        assert Faithfulness is not None
        assert ContextPrecision is not None
        assert ContextRecall is not None
        assert AnswerRelevancy is not None

    @patch.dict("os.environ", {"SSL_CERT_FILE": "", "OPENAI_API_KEY": "test-api-key"})
    def test_openai_client_initialization(self):
        """Test OpenAI client initialization with API key from environment."""
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(api_key="test-api-key", model="gpt-4o-mini")
        assert client is not None

    @patch.dict("os.environ", {"SSL_CERT_FILE": "", "OPENAI_API_KEY": "test-api-key"})
    def test_openai_client_with_custom_model(self):
        """Test OpenAI client initialization with custom model."""
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(api_key="test-api-key", model="gpt-4o-mini")
        assert client is not None

    def test_get_openai_client_with_api_key(self, monkeypatch):
        """Test that get_openai_client creates client with API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        client = get_openai_client()
        assert client is not None

    def test_get_openai_client_raises_without_api_key(self, monkeypatch):
        """Test that get_openai_client raises error without API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable"):
            get_openai_client()

    def test_get_llm_backend_returns_openai(self, monkeypatch):
        """Test that get_llm_backend returns OpenAI LLM by default."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        llm = get_llm_backend(provider="openai", model="gpt-4o-mini")
        assert llm is not None

    def test_get_llm_backend_with_temperature(self, monkeypatch):
        """Test that get_llm_backend accepts temperature parameter."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        llm = get_llm_backend(provider="openai", model="gpt-4o-mini", temperature=0.3)
        assert llm is not None

    def test_single_turn_sample_creation(self):
        """Test SingleTurnSample can be created with correct fields."""
        from ragas import SingleTurnSample

        sample = SingleTurnSample(
            user_input="What is the termination clause?",
            retrieved_contexts=["The contract can be terminated..."],
            response="The contract can be terminated with 30 days notice.",
            reference="The contract allows termination with 30 days notice.",
        )

        assert sample.user_input == "What is the termination clause?"
        assert len(sample.retrieved_contexts) == 1

    def test_metric_instantiation(self, monkeypatch):
        """Test that RAGAS metrics can be instantiated."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SSL_CERT_FILE", "")

        metrics = create_ragas_metrics()

        assert "faithfulness" in metrics
        assert "context_precision" in metrics
        assert "context_recall" in metrics
        assert "answer_relevancy" in metrics
