"""
Tests for Phoenix replay modules.
"""


import pytest

from eval_harness.replay.corpus import CorpusBuilder
from eval_harness.replay.phoenix_client import PhoenixClient


class TestPhoenixClient:
    """Tests for PhoenixClient."""

    def test_phoenix_client_initialization_with_mock(self) -> None:
        """Test PhoenixClient initialization with mock."""
        # Test with no actual Phoenix connection
        client = PhoenixClient(
            endpoint="http://localhost:6006",
            project_name="test-project",
        )

        assert client._endpoint == "http://localhost:6006"
        assert client._project_name == "test-project"
        # Without actual Phoenix, _client will be None

    def test_phoenix_client_without_connection(self) -> None:
        """Test PhoenixClient handles no connection gracefully."""
        client = PhoenixClient()

        assert client._client is None
        assert not client.is_connected()

    def test_query_root_spans_when_disconnected(self) -> None:
        """Test query_root_spans raises error when disconnected."""
        client = PhoenixClient()
        # Simulate disconnected state
        client._client = None

        with pytest.raises(ConnectionError):
            client.query_root_spans()


class TestCorpusBuilder:
    """Tests for CorpusBuilder."""

    def test_corpus_builder_initialization(self) -> None:
        """Test CorpusBuilder initialization."""
        builder = CorpusBuilder()
        assert builder is not None

    def test_build_dataset_empty_list(self) -> None:
        """Test build_dataset with empty span list."""
        builder = CorpusBuilder()
        dataset = builder.build_dataset([])

        assert dataset == []

    def test_build_dataset_with_retriever_span(self) -> None:
        """Test build_dataset extracts RETRIEVER span data."""
        builder = CorpusBuilder()

        span = {
            "span_kind": "RETRIEVER",
            "input.value": "test query",
            "retrieval.documents.0.document.id": "doc1",
            "retrieval.documents.0.document.content": "content 1",
            "retrieval.documents.0.document.score": 0.95,
            "retrieval.documents.1.document.id": "doc2",
            "retrieval.documents.1.document.content": "content 2",
            "retrieval.documents.1.document.score": 0.85,
        }

        dataset = builder.build_dataset([span])

        assert len(dataset) == 1
        assert "retrieval_documents" in dataset[0]
        assert "input" in dataset[0]

    def test_build_dataset_with_llm_span(self) -> None:
        """Test build_dataset extracts LLM span data."""
        builder = CorpusBuilder()

        span = {
            "span_kind": "LLM",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "test query",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "test response",
        }

        dataset = builder.build_dataset([span])

        assert len(dataset) == 1
        assert "llm_input_messages" in dataset[0]
        assert "llm_output_messages" in dataset[0]

    def test_build_dataset_filters_unknown_spans(self) -> None:
        """Test build_dataset ignores unknown span kinds."""
        builder = CorpusBuilder()

        span = {
            "span_kind": "UNKNOWN",
            "some.attribute": "value",
        }

        dataset = builder.build_dataset([span])

        assert len(dataset) == 0

    def test_is_retriever_span_case_insensitive(self) -> None:
        """Test _is_retriever_span is case-insensitive."""
        builder = CorpusBuilder()

        assert builder._is_retriever_span({"span_kind": "RETRIEVER"})
        assert builder._is_retriever_span({"span_kind": "retriever"})
        assert builder._is_retriever_span({"span_kind": "Retriever"})
        assert not builder._is_retriever_span({"span_kind": "LLM"})

    def test_is_llm_span_case_insensitive(self) -> None:
        """Test _is_llm_span is case-insensitive."""
        builder = CorpusBuilder()

        assert builder._is_llm_span({"span_kind": "LLM"})
        assert builder._is_llm_span({"span_kind": "llm"})
        assert builder._is_llm_span({"span_kind": "Llm"})
        assert not builder._is_llm_span({"span_kind": "RETRIEVER"})
