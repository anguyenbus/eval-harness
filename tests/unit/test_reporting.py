"""Tests for reporting layer (CSV, HTML, regression check)."""

import pandas as pd
import pytest

from eval_harness.reporting.csv_writer import write_results
from eval_harness.reporting.html_summary import generate_summary
from eval_harness.reporting.regression_check import check_regression


class TestCSVWriter:
    """Test suite for CSV writer."""

    def test_write_results_creates_csv(self, tmp_path):
        """Test that write_results creates CSV file with correct columns."""
        results = [
            {
                "query_id": "q001",
                "question_id": "text_fidelity",
                "score": 0.95,
                "label": "pass",
                "error": "",
            },
            {
                "query_id": "q002",
                "question_id": "structure_recall",
                "score": 0.80,
                "label": "fail",
                "error": "",
            },
        ]

        output_path = tmp_path / "results.csv"
        write_results(results, output_path)

        assert output_path.exists()

        # Verify content
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert list(df.columns) == [
            "query_id",
            "question_id",
            "score",
            "label",
            "error",
        ]
        assert df.iloc[0]["score"] == 0.95
        assert df.iloc[1]["label"] == "fail"

    def test_write_results_handles_empty_list(self, tmp_path):
        """Test that write_results handles empty results list."""
        output_path = tmp_path / "empty.csv"
        write_results([], output_path)

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 0
        assert list(df.columns) == [
            "query_id",
            "question_id",
            "score",
            "label",
            "error",
        ]


class TestHTMLSummary:
    """Test suite for HTML summary report."""

    def test_generate_summary_creates_html(self, tmp_path):
        """Test that generate_summary creates HTML file."""
        # Create CSV results file
        results_path = tmp_path / "results.csv"
        results_data = pd.DataFrame(
            {
                "query_id": ["q001", "q002", "q003"],
                "question_id": ["text_fidelity", "structure_recall", "text_fidelity"],
                "score": [0.95, 0.80, 0.90],
                "label": ["pass", "fail", "pass"],
                "error": ["", "", ""],
            }
        )
        results_data.to_csv(results_path, index=False)

        output_path = tmp_path / "summary.html"
        generate_summary(results_path, output_path)

        assert output_path.exists()

        # Verify HTML content
        html = output_path.read_text()
        assert "<html>" in html or "<HTML" in html
        assert "pass" in html
        assert "fail" in html

    def test_generate_summary_aggregates_pass_rates(self, tmp_path):
        """Test that summary aggregates pass rates by pillar."""
        results_path = tmp_path / "results.csv"
        results_data = pd.DataFrame(
            {
                "query_id": ["q001", "q002", "q003", "q004"],
                "question_id": ["metric1", "metric1", "metric2", "metric2"],
                "score": [1.0, 0.5, 0.9, 0.3],
                "label": ["pass", "pass", "fail", "fail"],
                "error": ["", "", "", ""],
            }
        )
        results_data.to_csv(results_path, index=False)

        output_path = tmp_path / "summary.html"
        generate_summary(results_path, output_path)

        html = output_path.read_text()
        # Should show pass rate information
        assert "50" in html  # 50% pass rate


class TestRegressionCheck:
    """Test suite for regression checking."""

    def test_check_regression_passes_on_no_regression(self, tmp_path):
        """Test that regression check passes when current matches baseline."""
        # Create baseline file
        baseline_path = tmp_path / "baseline.json"
        import json

        baseline_data = {
            "metrics": {
                "text_fidelity": {"score": 0.95, "severity": "major"},
                "structure_recall": {"score": 0.90, "severity": "blocker"},
            },
        }
        baseline_path.write_text(json.dumps(baseline_data))

        # Create current results (same as baseline)
        current_path = tmp_path / "current.json"
        current_data = {
            "metrics": {
                "text_fidelity": {"score": 0.95, "severity": "major"},
                "structure_recall": {"score": 0.90, "severity": "blocker"},
            },
        }
        current_path.write_text(json.dumps(current_data))

        # Should not raise
        check_regression(current_path, baseline_path)

    def test_check_regression_fails_on_blocker_regression(self, tmp_path):
        """Test that regression check fails on blocker severity regression."""
        # Create baseline file
        baseline_path = tmp_path / "baseline.json"
        import json

        baseline_data = {
            "metrics": {
                "structure_recall": {"score": 0.90, "severity": "blocker"},
            },
        }
        baseline_path.write_text(json.dumps(baseline_data))

        # Create current results with regression
        current_path = tmp_path / "current.json"
        current_data = {
            "metrics": {
                "structure_recall": {
                    "score": 0.70,
                    "severity": "blocker",
                },  # Regression!
            },
        }
        current_path.write_text(json.dumps(current_data))

        # Should raise
        with pytest.raises(RuntimeError, match="Regression detected"):
            check_regression(current_path, baseline_path)

    def test_check_regression_ignores_minor_regressions(self, tmp_path):
        """Test that regression check ignores minor severity regressions."""
        # Create baseline file
        baseline_path = tmp_path / "baseline.json"
        import json

        baseline_data = {
            "metrics": {
                "some_metric": {"score": 0.80, "severity": "minor"},
            },
        }
        baseline_path.write_text(json.dumps(baseline_data))

        # Create current results with regression
        current_path = tmp_path / "current.json"
        current_data = {
            "metrics": {
                "some_metric": {
                    "score": 0.50,
                    "severity": "minor",
                },  # Regression but minor
            },
        }
        current_path.write_text(json.dumps(current_data))

        # Should not raise for minor
        check_regression(current_path, baseline_path)
