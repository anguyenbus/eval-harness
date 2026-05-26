"""Tests for Legal RAG Bench dataset loader."""

from unittest.mock import MagicMock, patch

import pytest

from eval_harness.datasets.legal_rag_bench import (
    _get_hf_token,
    _get_slice_limit,
    load_legal_rag_bench,
)


class TestGetSliceLimit:
    """Test suite for _get_slice_limit function."""

    def test_pico_slice_returns_2(self):
        """Test that pico slice returns limit of 2."""
        limit = _get_slice_limit("pico")
        assert limit == 2

    def test_nano_slice_returns_10(self):
        """Test that nano slice returns limit of 10."""
        limit = _get_slice_limit("nano")
        assert limit == 10

    def test_full_slice_returns_none(self):
        """Test that full slice returns None (no limit)."""
        limit = _get_slice_limit("full")
        assert limit is None

    def test_invalid_slice_raises_value_error(self):
        """Test that invalid slice raises ValueError."""
        with pytest.raises(ValueError, match="slice must be"):
            _get_slice_limit("invalid")


class TestGetHfToken:
    """Test suite for _get_hf_token function."""

    def test_hf_token_from_environment(self, monkeypatch):
        """Test that HF token is read from environment variable."""
        monkeypatch.setenv("HF_TOKEN", "test_token_123")

        token = _get_hf_token()
        assert token == "test_token_123"

    def test_hf_token_from_default_path(self, tmp_path, monkeypatch):
        """Test that HF token is read from default path when env var not set."""
        # Remove env var if set
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Create temp .huggingface directory with token
        hf_dir = tmp_path / ".huggingface"
        hf_dir.mkdir(parents=True)
        (hf_dir / "token").write_text("file_token_456")

        with patch(
            "eval_harness.datasets.legal_rag_bench.DEFAULT_HF_TOKEN_PATH",
            hf_dir / "token",
        ):
            token = _get_hf_token()
            assert token == "file_token_456"

    def test_hf_token_none_when_not_found(self, tmp_path, monkeypatch):
        """Test that None is returned when token not found."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Create temp directory without token file
        hf_dir = tmp_path / ".huggingface"
        hf_dir.mkdir(parents=True)

        with patch(
            "eval_harness.datasets.legal_rag_bench.DEFAULT_HF_TOKEN_PATH",
            hf_dir / "token",
        ):
            token = _get_hf_token()
            assert token is None


class TestLegalRagBenchLoader:
    """Test suite for Legal RAG Bench dataset loader."""

    def test_nano_slice_yields_10_questions(self, tmp_path):
        """Test that nano slice yields 10 questions."""
        # Mock the HuggingFace dataset loading
        mock_dataset = MagicMock()

        # Create mock data for 100 questions
        # Note: The actual dataset uses: id, question, answer, relevant_passage_id
        mock_data = [
            {
                "id": f"q{i:03d}",
                "question": f"Question {i}?",
                "relevant_passage_id": f"passage_{i}",
                "answer": f"Answer {i}.",
            }
            for i in range(100)
        ]

        mock_dataset.__iter__.return_value = iter(mock_data)

        with patch("datasets.load_dataset", return_value=mock_dataset):
            results = list(
                load_legal_rag_bench(tmp_path, slice="nano", force_refresh=False)
            )

            # Nano slice should have 10 questions
            assert len(results) == 10

    def test_returns_correct_tuple_structure(self, tmp_path):
        """Test loader returns (query_id, query_text, relevant_passage_id, answer)."""
        # Mock dataset with single item
        # Note: The actual isaacus/legal-rag-bench dataset uses:
        # id, question, answer, relevant_passage_id
        mock_dataset = MagicMock()

        mock_data = [
            {
                "id": "q001",  # Dataset uses "id" not "query_id"
                "question": "What is the termination clause?",
                "relevant_passage_id": "passage_123",
                "answer": "The contract can be terminated with 30 days notice.",
            }
        ]

        mock_dataset.__iter__.return_value = iter(mock_data)

        with patch("datasets.load_dataset", return_value=mock_dataset):
            results = list(
                load_legal_rag_bench(tmp_path, slice="pico", force_refresh=False)
            )

            assert len(results) == 1
            query_id, query_text, relevant_passage_id, reference_answer = results[0]

            assert query_id == "q001"
            assert query_text == "What is the termination clause?"
            assert relevant_passage_id == "passage_123"
            assert (
                reference_answer
                == "The contract can be terminated with 30 days notice."
            )

    def test_iterator_pattern_works(self, tmp_path):
        """Test that loader uses iterator pattern."""
        mock_dataset = MagicMock()

        mock_data = [
            {
                "id": f"q{i:03d}",
                "question": f"Question {i}?",
                "relevant_passage_id": f"passage_{i}",
                "answer": f"Answer {i}.",
            }
            for i in range(5)
        ]

        mock_dataset.__iter__.return_value = iter(mock_data)

        with patch("datasets.load_dataset", return_value=mock_dataset):
            result = load_legal_rag_bench(tmp_path, slice="nano", force_refresh=False)
            assert hasattr(result, "__iter__")

            count = sum(1 for _ in result)
            assert count == 5

    def test_slice_limits_enforced(self, tmp_path):
        """Test that slice limits are enforced correctly."""
        mock_dataset = MagicMock()

        # Create 200 items
        mock_data = [
            {
                "id": f"q{i:03d}",
                "question": f"Question {i}?",
                "relevant_passage_id": f"passage_{i}",
                "answer": f"Answer {i}.",
            }
            for i in range(200)
        ]

        mock_dataset.__iter__.return_value = iter(mock_data)

        with patch("datasets.load_dataset", return_value=mock_dataset):
            # Pico should give 2
            pico_results = list(
                load_legal_rag_bench(tmp_path, slice="pico", force_refresh=False)
            )
            assert len(pico_results) == 2

            # Nano should give 10
            nano_results = list(
                load_legal_rag_bench(tmp_path, slice="nano", force_refresh=False)
            )
            assert len(nano_results) == 10

    def test_cache_directory_created(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "cache" / "legal_rag_bench"

        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter([])

        with patch("datasets.load_dataset", return_value=mock_dataset):
            _ = list(load_legal_rag_bench(cache_dir, slice="nano", force_refresh=False))

            # Cache directory should be created
            assert cache_dir.exists()

    def test_force_refresh_loads_dataset(self, tmp_path):
        """Test that force_refresh flag triggers dataset loading."""
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter([])

        with patch("datasets.load_dataset", return_value=mock_dataset) as mock_load:
            # Load with force_refresh
            _ = list(load_legal_rag_bench(tmp_path, slice="nano", force_refresh=True))

            # load_dataset should be called
            assert mock_load.called
