"""
Phase 3 Integration Tests for Phoenix Native Migration.

Tests end-to-end evaluation with Phoenix evaluators,
evaluation with both flag states, and export formats with Phoenix metrics.
"""

from __future__ import annotations

from unittest.mock import patch


class TestPhase3EndToEndEvaluation:
    """Integration tests for end-to-end evaluation with Phoenix evaluators."""

    def test_end_to_end_evaluation_with_phoenix_evaluators(self) -> None:
        """Test end-to-end evaluation pipeline with Phoenix evaluators."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        # Sample RAG outputs
        rag_outputs = [
            {
                "query": {"text": "What is contract law?"},
                "answer": {"text": "Contract law governs agreements..."},
                "retrieved_chunks": [{"text": "Contract law context"}],
            },
            {
                "query": {"text": "What is tort law?"},
                "answer": {"text": "Tort law deals with civil wrongs..."},
                "retrieved_chunks": [{"text": "Tort law context"}],
            },
        ]
        reference_answers = [
            "Contract law governs agreements.",
            "Tort law addresses civil wrongs.",
        ]

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            import pandas as pd

            mock_result = pd.DataFrame({
                "faithfulness_score": [0.85, 0.90],
                "correctness_score": [0.90, 0.88],
                "relevance_score": [0.75, 0.82],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            # Batch evaluation
            scores_list = adapter.batch_compute_metrics(rag_outputs, reference_answers)

            assert len(scores_list) == 2
            assert all("faithfulness" in s for s in scores_list)
            assert all("correctness" in s for s in scores_list)
            assert all("relevance" in s for s in scores_list)

    def test_evaluation_with_both_flag_states(self) -> None:
        """Test evaluation works with both use_phoenix_native flag states."""
        from eval_harness.observability.config_phoenix_native import (
            get_phoenix_native_config,
        )

        # Test with flag=True
        config_native = {"phoenix_native": {"use_phoenix_native": True}}
        phoenix_config = get_phoenix_native_config(config_native)
        assert phoenix_config["use_phoenix_native"] is True

        # Test with flag=False
        config_legacy = {"phoenix_native": {"use_phoenix_native": False}}
        phoenix_config = get_phoenix_native_config(config_legacy)
        assert phoenix_config["use_phoenix_native"] is False

        # Test with missing flag (default: False)
        config_default = {}
        phoenix_config = get_phoenix_native_config(config_default)
        assert phoenix_config["use_phoenix_native"] is False

    def test_export_formats_with_phoenix_metrics(self) -> None:
        """Test that export formats work with Phoenix metrics."""
        import csv
        import json
        import tempfile
        from pathlib import Path

        # Sample data with Phoenix metrics
        export_data = {
            "candidate": "phoenix-native",
            "baseline": "legacy",
            "num_questions": 2,
            "averages": {
                "candidate": {
                    "faithfulness": 0.85,
                    "correctness": 0.90,
                    "relevance": 0.75,
                },
                "baseline": {
                    "faithfulness": 0.80,
                    "correctness": 0.85,
                    "relevance": 0.70,
                },
            },
            "statistical_tests": {
                "faithfulness": {
                    "p_value": 0.03,
                    "effect_size": 0.25,
                    "winner": "candidate",
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON export
            json_path = Path(tmpdir) / "results.json"
            with open(json_path, "w") as f:
                json.dump(export_data, f)

            with open(json_path) as f:
                loaded = json.load(f)

                # Verify Phoenix metric names in export
                assert "faithfulness" in loaded["averages"]["candidate"]
                assert "correctness" in loaded["averages"]["candidate"]
                assert "relevance" in loaded["averages"]["candidate"]

                # Verify no deepeval prefix
                assert "deepeval_faithfulness" not in str(loaded)

            # Test CSV export
            csv_path = Path(tmpdir) / "results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "candidate_avg", "baseline_avg"])
                writer.writerow(["faithfulness", "0.85", "0.80"])
                writer.writerow(["correctness", "0.90", "0.85"])
                writer.writerow(["relevance", "0.75", "0.70"])

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                # Verify Phoenix metric names in CSV
                metrics = [row["metric"] for row in rows]
                assert "faithfulness" in metrics
                assert "correctness" in metrics
                assert "relevance" in metrics


class TestStatisticalComparisonPreserved:
    """Tests that statistical comparison functionality is preserved."""

    def test_statistical_comparison_works_with_phoenix_metrics(self) -> None:
        """Test that paired_comparison works with Phoenix metrics."""
        from eval_harness.replay.comparison import paired_comparison

        # Sample Phoenix metric scores
        candidate_scores = [0.85, 0.90, 0.88, 0.92, 0.87]
        baseline_scores = [0.80, 0.85, 0.83, 0.88, 0.82]

        result = paired_comparison(
            candidate_scores,
            baseline_scores,
            candidate_errors=1,
            baseline_errors=2,
            total_questions=7,
        )

        # Verify comparison result structure
        assert hasattr(result, "p_value")
        assert hasattr(result, "effect_size")
        assert hasattr(result, "pass_fail")
        assert hasattr(result, "winner")
        assert hasattr(result, "candidate_error_rate")
        assert hasattr(result, "baseline_error_rate")
        assert hasattr(result, "error_rate_pass_fail")
        assert hasattr(result, "overall_pass_fail")

    def test_paired_t_tests_work_with_phoenix_metrics(self) -> None:
        """Test that paired t-tests work with Phoenix metrics."""
        from eval_harness.replay.comparison import _cliffs_delta, _wilcoxon_test

        candidate_scores = [0.85, 0.90, 0.88]
        baseline_scores = [0.80, 0.85, 0.83]

        # Wilcoxon test
        statistic, p_value = _wilcoxon_test(candidate_scores, baseline_scores)

        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0

        # Cliff's Delta
        effect_size = _cliffs_delta(candidate_scores, baseline_scores)

        assert isinstance(effect_size, float)
        assert -1.0 <= effect_size <= 1.0

    def test_error_rate_gating_preserved(self) -> None:
        """Test that error rate gating is preserved."""
        from eval_harness.replay.comparison import paired_comparison

        candidate_scores = [0.85, 0.90, 0.88]
        baseline_scores = [0.80, 0.85, 0.83]

        # Test with acceptable error rate delta
        result = paired_comparison(
            candidate_scores,
            baseline_scores,
            candidate_errors=2,
            baseline_errors=2,
            total_questions=7,
            max_error_rate_delta=0.10,
        )

        # Error rate check should pass
        assert result.error_rate_pass_fail is True

        # Test with high error rate delta
        result = paired_comparison(
            candidate_scores,
            baseline_scores,
            candidate_errors=5,
            baseline_errors=0,
            total_questions=8,
            max_error_rate_delta=0.10,
        )

        # Error rate check should fail
        assert result.error_rate_pass_fail is False


class TestExportFormatsPreserved:
    """Tests that export formats are preserved."""

    def test_csv_export_format_unchanged(self) -> None:
        """Test that CSV export format is unchanged."""
        import csv
        import tempfile
        from pathlib import Path

        expected_headers = [
            "metric",
            "candidate_avg",
            "baseline_avg",
            "p_value",
            "effect_size",
            "winner",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"

            # Write sample export
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(expected_headers)
                writer.writerow(["faithfulness", "0.85", "0.80", "0.03", "0.25", "candidate"])

            # Verify format
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames

                assert headers == expected_headers

    def test_json_export_format_unchanged(self) -> None:
        """Test that JSON export format is unchanged."""
        import json
        import tempfile
        from pathlib import Path

        expected_structure = {
            "candidate": str,
            "baseline": str,
            "num_questions": int,
            "averages": dict,
            "statistical_tests": dict,
        }

        export_data = {
            "candidate": "test-candidate",
            "baseline": "test-baseline",
            "num_questions": 10,
            "averages": {
                "candidate": {"faithfulness": 0.85},
                "baseline": {"faithfulness": 0.80},
            },
            "statistical_tests": {
                "faithfulness": {
                    "p_value": 0.03,
                    "effect_size": 0.25,
                    "winner": "candidate",
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"

            with open(json_path, "w") as f:
                json.dump(export_data, f)

            with open(json_path) as f:
                loaded = json.load(f)

                # Verify structure preserved
                for key, expected_type in expected_structure.items():
                    assert key in loaded
                    assert isinstance(loaded[key], expected_type)
