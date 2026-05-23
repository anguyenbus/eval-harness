"""
End-to-end tests for span generator runner.

Uses small_qa_sample.json fixture for deterministic fast tests.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from eval_harness.stubs.span_generator.config import GeneratorConfig
from eval_harness.stubs.span_generator.runner import GeneratorResult, run_generator


class TestRunnerEndToEnd:
    """End-to-end tests for span generation."""

    @pytest.fixture
    def small_fixture_path(self) -> Path:
        """Path to small QA sample fixture."""
        return (
            Path(__file__).parent
            / "fixtures"
            / "small_qa_sample.json"
        )

    @pytest.fixture
    def mock_config(self) -> GeneratorConfig:
        """Mock generator config for testing."""
        return GeneratorConfig(
            phoenix_endpoint="http://localhost:6006",
            project_name="test-project",
            default_limit=100,
            batch_export=True,
            seed=42,
            stub_model_id="model",
            stub_embedding_model="embedder",
        )

    def test_run_generator_with_mock_tracer(
        self, mock_config, small_fixture_path
    ) -> None:
        """Test run_generator with mocked tracer."""
        with patch(
            "eval_harness.stubs.span_generator.runner.setup_tracer"
        ) as mock_setup:
            # Mock tracer
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock()
            mock_provider = MagicMock()
            mock_setup.return_value = (mock_provider, mock_tracer)

            # Mock iter_questions to return fixture data
            with patch(
                "eval_harness.stubs.span_generator.runner.iter_questions"
            ) as mock_iter:
                from eval_harness.stubs.span_generator.loader import GeneratorQuestion

                mock_iter.return_value = iter(
                    [
                        GeneratorQuestion(
                            id="q1",
                            question="Test question?",
                            expected_answer="Test answer",
                            relevant_passage_id="passage_001",
                            case_id="synth-case-0000",
                            tenant_id_hashed="synth-tenant-0",
                        )
                    ]
                )

                # Mock stub pipeline query (patch at import location)
                with patch(
                    "eval_harness.stubs.rag.chromadb_query.query"
                ) as mock_query:
                    mock_query.return_value = {
                        "answer": {"text": "Generated answer"},
                        "retrieved_chunks": [],
                    }

                    result = run_generator(
                        config=mock_config,
                        corpus_dir=Path("/tmp/corpus"),
                        limit=1,
                    )

                    # Verify result
                    assert isinstance(result, GeneratorResult)
                    assert result.successes == 1
                    assert result.failures == 0
                    assert len(result.run_id) > 0

    def test_run_generator_with_failure(self, mock_config) -> None:
        """Test run_generator handles failures gracefully."""
        with patch(
            "eval_harness.stubs.span_generator.runner.setup_tracer"
        ) as mock_setup:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.side_effect = (
                Exception("Test error")
            )
            mock_provider = MagicMock()
            mock_setup.return_value = (mock_provider, mock_tracer)

            with patch(
                "eval_harness.stubs.span_generator.runner.iter_questions"
            ) as mock_iter:
                from eval_harness.stubs.span_generator.loader import GeneratorQuestion

                mock_iter.return_value = iter(
                    [
                        GeneratorQuestion(
                            id="q1",
                            question="Test",
                            expected_answer="Answer",
                            relevant_passage_id="p1",
                            case_id="synth-case-0000",
                            tenant_id_hashed="synth-tenant-0",
                        )
                    ]
                )

                result = run_generator(
                    config=mock_config,
                    corpus_dir=Path("/tmp/corpus"),
                    limit=1,
                )

                # Should have 1 failure
                assert result.successes == 0
                assert result.failures == 1

    def test_run_generator_without_tracer(self, mock_config) -> None:
        """Test run_generator when tracer setup fails."""
        with patch(
            "eval_harness.stubs.span_generator.runner.setup_tracer"
        ) as mock_setup:
            # Tracer setup fails
            mock_setup.return_value = (None, None)

            result = run_generator(
                config=mock_config,
                corpus_dir=Path("/tmp/corpus"),
                limit=1,
            )

            # Should return empty result
            assert result.successes == 0
            assert result.failures == 0


class TestFixtureValidation:
    """Tests for fixture file validation."""

    def test_small_qa_sample_fixture_exists(self) -> None:
        """Test that small_qa_sample.json fixture exists."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "small_qa_sample.json"
        )
        assert fixture_path.exists()

    def test_small_qa_sample_fixture_valid_json(self) -> None:
        """Test that small_qa_sample.json is valid JSON."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "small_qa_sample.json"
        )
        with open(fixture_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 5

    def test_small_qa_sample_fixture_structure(self) -> None:
        """Test that fixture items have required fields."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "small_qa_sample.json"
        )
        with open(fixture_path) as f:
            data = json.load(f)

        for item in data:
            assert "id" in item
            assert "question" in item
            assert "answer" in item
            assert "relevant_passage_id" in item
