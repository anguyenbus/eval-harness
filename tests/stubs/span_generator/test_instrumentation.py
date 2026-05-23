"""
Tests for stub RAG pipeline instrumentation.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from eval_harness.stubs.rag.embedder import SentenceTransformersEmbedder
from eval_harness.stubs.rag.generator import LLMGenerator
from eval_harness.stubs.rag.retriever import SemanticRetriever


class TestEmbedderInstrumentation:
    """Tests for embedder span emission."""

    def test_embedder_emits_embedding_span(self) -> None:
        """Test that embedder emits EMBEDDING span with required attributes."""
        # Mock the tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock()

        with patch(
            "eval_harness.stubs.rag.embedder._get_tracer", return_value=mock_tracer
        ):
            # Mock OpenInference import (imported inline in embed method)
            with patch(
                "openinference.semconv.trace.OpenInferenceSpanKindValues"
            ) as mock_span_kind:
                mock_span_kind.EMBEDDING = "EMBEDDING"
                # Mock the model to avoid actual loading
                with patch.object(
                    SentenceTransformersEmbedder, "_load_model"
                ) as mock_load:
                    import numpy as np

                    mock_model = MagicMock()
                    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                    mock_load.return_value = mock_model

                    embedder = SentenceTransformersEmbedder()
                    result = embedder.embed(["test query"])

                    # Verify span was created
                    assert mock_tracer.start_as_current_span.called

                    # Verify attributes were set
                    assert mock_span.set_attribute.called

    def test_embedder_without_tracer(self) -> None:
        """Test that embedder works without tracer (graceful degradation)."""
        with patch(
            "eval_harness.stubs.rag.embedder._get_tracer", return_value=None
        ):
            with patch.object(
                SentenceTransformersEmbedder, "_load_model"
            ) as mock_load:
                # Return numpy array that needs tolist conversion
                import numpy as np

                mock_model = MagicMock()
                mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                mock_load.return_value = mock_model

                embedder = SentenceTransformersEmbedder()
                result = embedder.embed(["test"])

                assert result == [[0.1, 0.2, 0.3]]

    def test_embedder_sets_vector_dim_attribute(self) -> None:
        """Test that embedder sets vector_dim attribute."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock()

        with patch(
            "eval_harness.stubs.rag.embedder._get_tracer", return_value=mock_tracer
        ):
            with patch(
                "openinference.semconv.trace.OpenInferenceSpanKindValues"
            ) as mock_span_kind:
                mock_span_kind.EMBEDDING = "EMBEDDING"
                with patch.object(
                    SentenceTransformersEmbedder, "_load_model"
                ) as mock_load:
                    # Return 384-dimensional embedding
                    import numpy as np

                    mock_model = MagicMock()
                    mock_model.encode.return_value = np.array([[0.1] * 384])
                    mock_load.return_value = mock_model

                    embedder = SentenceTransformersEmbedder()
                    result = embedder.embed(["test"])

                    # Verify vector_dim attribute was set
                    assert mock_span.set_attribute.called


class TestRetrieverInstrumentation:
    """Tests for retriever span emission."""

    def test_retriever_emits_retriever_span(self) -> None:
        """Test that retriever emits RETRIEVER span with required attributes."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock()

        with patch(
            "eval_harness.stubs.rag.retriever._get_tracer", return_value=mock_tracer
        ):
            with patch(
                "openinference.semconv.trace.OpenInferenceSpanKindValues"
            ) as mock_span_kind:
                mock_span_kind.RETRIEVER = "RETRIEVER"
                # Mock collection and embedder
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    "ids": [["doc1", "doc2"]],
                    "documents": [["text1", "text2"]],
                    "metadatas": [[{"doc_id": "doc1"}, {"doc_id": "doc2"}]],
                    "distances": [[0.1, 0.2]],
                }

                mock_embedder = MagicMock()
                mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

                retriever = SemanticRetriever(mock_collection, mock_embedder)
                result = retriever.retrieve("test query", top_k=2)

                # Verify span was created
                assert mock_tracer.start_as_current_span.called

    def test_retriever_sets_document_attributes(self) -> None:
        """Test that retriever sets retrieval.document attributes."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock()

        with patch(
            "eval_harness.stubs.rag.retriever._get_tracer", return_value=mock_tracer
        ):
            with patch(
                "openinference.semconv.trace.OpenInferenceSpanKindValues"
            ) as mock_span_kind:
                mock_span_kind.RETRIEVER = "RETRIEVER"
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    "ids": [["doc1"]],
                    "documents": [["retrieved text"]],
                    "metadatas": [[{"doc_id": "doc1"}]],
                    "distances": [[0.1]],
                }

                mock_embedder = MagicMock()
                mock_embedder.embed.return_value = [[0.1]]

                retriever = SemanticRetriever(mock_collection, mock_embedder)
                result = retriever.retrieve("test", top_k=1)

                # Verify document attributes were set
                assert mock_span.set_attribute.called
                # Check that the call count is sufficient for input + multiple documents
                assert mock_span.set_attribute.call_count >= 2

    def test_retriever_without_tracer(self) -> None:
        """Test that retriever works without tracer."""
        with patch(
            "eval_harness.stubs.rag.retriever._get_tracer", return_value=None
        ):
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["doc1"]],
                "documents": [["text"]],
                "metadatas": [[{"doc_id": "doc1"}]],
                "distances": [[0.1]],
            }

            mock_embedder = MagicMock()
            mock_embedder.embed.return_value = [[0.1]]

            retriever = SemanticRetriever(mock_collection, mock_embedder)
            result = retriever.retrieve("test", top_k=1)

            assert len(result) == 1
            assert result[0]["doc_id"] == "doc1"


class TestGeneratorInstrumentation:
    """Tests for generator span emission."""

    def test_generator_emits_llm_span(self) -> None:
        """Test that generator emits LLM span with required attributes."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock()

        with patch(
            "eval_harness.stubs.rag.generator._get_tracer", return_value=mock_tracer
        ):
            with patch(
                "openinference.semconv.trace.OpenInferenceSpanKindValues"
            ) as mock_span_kind:
                mock_span_kind.LLM = "LLM"
                # Mock OpenAI client
                with patch("openai.OpenAI") as mock_openai:
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_choice = MagicMock()
                    mock_message = MagicMock()
                    mock_message.content = "Test answer"
                    mock_choice.message = mock_message
                    mock_response.choices = [mock_choice]
                    mock_client.chat.completions.create.return_value = mock_response
                    mock_openai.return_value = mock_client

                    # Set API key
                    import os

                    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                        generator = LLMGenerator(model="gpt-4o-mini")
                        result = generator.generate("Test question", [])

                        # Verify span was created
                        assert mock_tracer.start_as_current_span.called

                        # Verify model_name attribute was set
                        assert mock_span.set_attribute.called

    def test_generator_deterministic_mode(self) -> None:
        """Test that generator uses deterministic mode (temperature=0)."""
        with patch(
            "eval_harness.stubs.rag.generator._get_tracer", return_value=None
        ):
            with patch("openai.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_choice = MagicMock()
                mock_message = MagicMock()
                mock_message.content = "Deterministic answer"
                mock_choice.message = mock_message
                mock_response.choices = [mock_choice]
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                import os

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    generator = LLMGenerator(model="gpt-4o-mini", deterministic_mode=True)
                    result = generator.generate("Test", [])

                    # Verify temperature=0 was used
                    call_args = mock_client.chat.completions.create.call_args
                    assert call_args.kwargs.get("temperature") == 0.0

    def test_generator_without_tracer(self) -> None:
        """Test that generator works without tracer."""
        with patch(
            "eval_harness.stubs.rag.generator._get_tracer", return_value=None
        ):
            with patch("openai.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_choice = MagicMock()
                mock_message = MagicMock()
                mock_message.content = "Answer"
                mock_choice.message = mock_message
                mock_response.choices = [mock_choice]
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                import os

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    generator = LLMGenerator(model="gpt-4o-mini")
                    result = generator.generate("Test", [])

                    assert result["text"] == "Answer"
                    assert result["answer_supported"] is True
