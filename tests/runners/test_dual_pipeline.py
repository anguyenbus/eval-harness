"""
Tests for dual pipeline runner (Task Group 3).

Focused tests for dual pipeline execution with baseline and candidate
configurations.
"""

from pathlib import Path
from unittest.mock import MagicMock


class TestDualPipelineRunner:
    """Test DualPipelineRunner for side-by-side RAG comparison."""

    def test_dual_runner_executes_both_baseline_and_candidate(self) -> None:
        """Test that dual runner executes both baseline and candidate."""
        from eval_harness.runners.dual_pipeline import DualPipelineRunner

        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "answer": {"text": "Test answer"},
            "retrieved_chunks": [],
        }

        def mock_factory(*args: object, **kwargs: object) -> MagicMock:
            return mock_adapter

        runner = DualPipelineRunner(
            baseline_chunk_size=512,
            baseline_overlap=0,
            candidate_chunk_size=512,
            candidate_overlap=150,
            adapter_factory=mock_factory,
        )

        results = runner.run_comparison(
            questions=["What is test?"],
            corpus_dir=Path("/fake/corpus"),
        )

        # Verify both baseline and candidate were executed
        assert "baseline_scores" in results
        assert "candidate_scores" in results
        assert len(results["baseline_scores"]) == 1
        assert len(results["candidate_scores"]) == 1

    def test_results_structure_contains_per_question_metrics(self) -> None:
        """Test that results structure contains per-question metrics."""
        from eval_harness.runners.dual_pipeline import DualPipelineRunner

        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "answer": {"text": "Test answer"},
            "retrieved_chunks": [],
        }

        runner = DualPipelineRunner(
            baseline_chunk_size=512,
            baseline_overlap=0,
            candidate_chunk_size=512,
            candidate_overlap=150,
            adapter_factory=lambda *a, **k: mock_adapter,
        )

        results = runner.run_comparison(
            questions=["Q1", "Q2"],
            corpus_dir=Path("/fake/corpus"),
        )

        # Check per-question metrics
        assert "per_question" in results
        assert len(results["per_question"]) == 2

        # Check structure of per-question entry
        pq = results["per_question"][0]
        assert "question_id" in pq
        assert "baseline_score" in pq
        assert "candidate_score" in pq
        assert "delta" in pq

    def test_naive_mode_uses_different_question_sets(self) -> None:
        """Test that naive mode uses different question sets."""
        from eval_harness.runners.dual_pipeline import DualPipelineRunner

        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "answer": {"text": "Test answer"},
            "retrieved_chunks": [],
        }

        runner = DualPipelineRunner(
            baseline_chunk_size=512,
            baseline_overlap=0,
            candidate_chunk_size=512,
            candidate_overlap=150,
            adapter_factory=lambda *a, **k: mock_adapter,
        )

        results_naive = runner.run_comparison(
            questions=["Q1", "Q2", "Q3", "Q4"],
            corpus_dir=Path("/fake/corpus"),
            naive_mode=True,
        )

        # Verify naive mode executed
        assert "baseline_scores" in results_naive
        assert "candidate_scores" in results_naive
        # Naive mode splits questions: baseline gets first half, candidate gets second
        assert len(results_naive["baseline_scores"]) == 2
        assert len(results_naive["candidate_scores"]) == 2
        # In naive mode, per_question should be empty (unpaired)
        assert len(results_naive["per_question"]) == 0

    def test_baseline_and_candidate_configs_stored_in_results(self) -> None:
        """Test that baseline and candidate configs are stored in results."""
        from eval_harness.runners.dual_pipeline import DualPipelineRunner

        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "answer": {"text": "Test answer"},
            "retrieved_chunks": [],
        }

        runner = DualPipelineRunner(
            baseline_chunk_size=512,
            baseline_overlap=0,
            candidate_chunk_size=512,
            candidate_overlap=150,
            adapter_factory=lambda *a, **k: mock_adapter,
        )

        results = runner.run_comparison(
            questions=["Q1"],
            corpus_dir=Path("/fake/corpus"),
        )

        # Verify configs stored
        assert results["baseline_config"] == {"chunk_size": 512, "overlap": 0}
        assert results["candidate_config"] == {"chunk_size": 512, "overlap": 150}
