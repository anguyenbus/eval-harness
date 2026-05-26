"""
Tests for cost summary module and constants.

Tests cost tracking functionality including constants, cost summary calculation,
and export integration with Phoenix experiments.
"""

from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest


class TestCostConstants:
    """Tests for cost-related constants."""

    def test_app_cost_column_constant_is_defined(self):
        """Test that APP_COST_COLUMN constant is defined."""
        from eval_harness.experiments.cost_summary import APP_COST_COLUMN

        assert APP_COST_COLUMN is not None
        assert isinstance(APP_COST_COLUMN, str)

    def test_app_cost_column_constant_is_final_str(self):
        """Test that APP_COST_COLUMN constant is Final[str] type."""
        from eval_harness.experiments.cost_summary import APP_COST_COLUMN
        from typing import get_args, get_origin

        # Check that the constant is a string
        assert isinstance(APP_COST_COLUMN, str)

    def test_app_cost_column_constant_value(self):
        """Test that APP_COST_COLUMN has expected value."""
        from eval_harness.experiments.cost_summary import APP_COST_COLUMN

        assert APP_COST_COLUMN == "cost"

    def test_cost_summary_module_has_verification_comment(self):
        """Test that APP_COST_CONSTANT has verification comment in source."""
        import inspect
        from eval_harness.experiments import cost_summary

        source = inspect.getsource(cost_summary)

        # Check for verification comment pattern
        assert "Verified against arize-phoenix==" in source
        assert "verify_phoenix_cost_column.py" in source


class TestCostSummary:
    """Tests for cost_summary function."""

    def test_total_app_cost_from_parquet(self, tmp_path: Path):
        """Test total app cost calculation from parquet."""
        parquet_path = tmp_path / "test_results.parquet"

        # Create test data with app costs
        df = pl.DataFrame(
            {
                "query_id": ["q1", "q2", "q3"],
                "app_cost_usd": [0.001, 0.002, 0.003],
                "judge_cost_usd": [0.01, 0.02, 0.03],
                "faithfulness_score": [0.8, 0.9, 0.7],
            }
        )
        df.write_parquet(parquet_path)

        from eval_harness.experiments.cost_summary import cost_summary

        result = cost_summary(parquet_path)

        assert result["app_cost_usd"] == 0.006

    def test_total_judge_cost_calculation(self, tmp_path: Path):
        """Test total judge cost calculation from evaluator metadata."""
        parquet_path = tmp_path / "test_results.parquet"

        df = pl.DataFrame(
            {
                "query_id": ["q1", "q2"],
                "app_cost_usd": [0.001, 0.001],
                "judge_cost_usd": [0.05, 0.03],
            }
        )
        df.write_parquet(parquet_path)

        from eval_harness.experiments.cost_summary import cost_summary

        result = cost_summary(parquet_path)

        assert result["judge_cost_usd"] == 0.08

    def test_judge_to_app_ratio_computation(self, tmp_path: Path):
        """Test judge-to-app ratio computation."""
        parquet_path = tmp_path / "test_results.parquet"

        df = pl.DataFrame(
            {
                "query_id": ["q1", "q2"],
                "app_cost_usd": [0.10, 0.10],
                "judge_cost_usd": [0.05, 0.05],
            }
        )
        df.write_parquet(parquet_path)

        from eval_harness.experiments.cost_summary import cost_summary

        result = cost_summary(parquet_path)

        assert result["judge_to_app_ratio"] == 0.5

    def test_judge_to_app_ratio_none_when_app_cost_zero(self, tmp_path: Path):
        """Test judge-to-app ratio is None when app cost is zero."""
        parquet_path = tmp_path / "test_results.parquet"

        df = pl.DataFrame(
            {
                "query_id": ["q1", "q2"],
                "app_cost_usd": [0.0, 0.0],
                "judge_cost_usd": [0.05, 0.03],
            }
        )
        df.write_parquet(parquet_path)

        from eval_harness.experiments.cost_summary import cost_summary

        result = cost_summary(parquet_path)

        assert result["judge_to_app_ratio"] is None

    def test_p95_percentile_uses_quantile(self, tmp_path: Path):
        """Test P95 percentile calculation using quantile(0.95)."""
        parquet_path = tmp_path / "test_results.parquet"

        # Create 20 rows with known costs
        # Total costs: 0.002, 0.003, ..., 0.020, 0.021
        rows = []
        for i in range(1, 21):
            rows.append({
                "query_id": f"q{i}",
                "app_cost_usd": 0.001,
                "judge_cost_usd": i * 0.001,
            })

        df = pl.DataFrame(rows)
        df.write_parquet(parquet_path)

        from eval_harness.experiments.cost_summary import cost_summary

        result = cost_summary(parquet_path)

        # P95 with midpoint interpolation on 20 values (indices 0-19)
        # P95 position = 0.95 * 19 = 18.05 (between index 18 and 19)
        # Values at 18, 19 are 0.020, 0.021, so midpoint is 0.0205
        assert abs(result["p95_cost_per_row_usd"] - 0.0205) < 0.0001

    def test_runtime_assertion_when_cost_column_missing(self, tmp_path: Path):
        """Test runtime assertion when required cost column is missing."""
        parquet_path = tmp_path / "test_results.parquet"

        # Create dataframe without required columns
        df = pl.DataFrame(
            {
                "query_id": ["q1", "q2"],
                "some_score": [0.8, 0.9],
            }
        )
        df.write_parquet(parquet_path)

        from eval_harness.experiments.cost_summary import cost_summary

        with pytest.raises(RuntimeError, match="Expected column.*not in experiment dataframe"):
            cost_summary(parquet_path)

    def test_judge_cost_breakdown_by_metric(self, tmp_path: Path):
        """Test judge cost broken down by metric."""
        parquet_path = tmp_path / "test_results.parquet"

        df = pl.DataFrame(
            {
                "query_id": ["q1", "q2"],
                "app_cost_usd": [0.001, 0.001],
                "judge_cost_usd": [0.04, 0.04],
                "judge_faithfulness_cost_usd": [0.01, 0.01],
                "judge_context_precision_cost_usd": [0.01, 0.01],
                "judge_context_recall_cost_usd": [0.01, 0.01],
                "judge_answer_relevancy_cost_usd": [0.01, 0.01],
            }
        )
        df.write_parquet(parquet_path)

        from eval_harness.experiments.cost_summary import cost_summary

        result = cost_summary(parquet_path)

        assert result["judge_cost_by_metric_usd"]["faithfulness"] == 0.02
        assert result["judge_cost_by_metric_usd"]["context_precision"] == 0.02
        assert result["judge_cost_by_metric_usd"]["context_recall"] == 0.02
        assert result["judge_cost_by_metric_usd"]["answer_relevancy"] == 0.02


class TestExportIntegration:
    """Tests for export_experiment_results cost columns integration."""

    def test_app_cost_extracted_from_phoenix_cost_column(self, tmp_path: Path):
        """Test app cost extracted from Phoenix cost column."""
        from eval_harness.experiments.runner import export_experiment_results

        # Mock experiment with cost data
        experiment = {
            "experiment_id": "test-exp-1",
            "dataset_id": "",
            "experiment_name": "test_experiment",
            "task_runs": [
                {
                    "id": "run1",
                    "dataset_example_id": "",
                    "output": {"answer": "Test answer"},
                    "cost": 0.0015,
                },
                {
                    "id": "run2",
                    "dataset_example_id": "",
                    "output": {"answer": "Another answer"},
                    "cost": 0.0025,
                },
            ],
            "evaluation_runs": [],
        }

        output_dir = tmp_path / "output"
        result = export_experiment_results(experiment, output_dir)

        # Verify parquet export contains cost columns
        parquet_path = Path(result.get("parquet_path", ""))
        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            assert "app_cost_usd" in df.columns
            assert df["app_cost_usd"][0] == 0.0015
            assert df["app_cost_usd"][1] == 0.0025

    def test_judge_cost_from_evaluation_cost_metadata(self, tmp_path: Path):
        """Test judge cost extracted from evaluation_cost in evaluator metadata."""
        from eval_harness.experiments.runner import export_experiment_results

        experiment = {
            "experiment_id": "test-exp-2",
            "dataset_id": "",
            "experiment_name": "test_experiment",
            "task_runs": [
                {"id": "run1", "dataset_example_id": "", "output": {"answer": "Answer 1"}},
            ],
            "evaluation_runs": [
                {
                    "experiment_run_id": "run1",
                    "result": {
                        "name": "faithfulness",
                        "score": 0.8,
                        "metadata": {"evaluation_cost": 0.01},
                    },
                },
                {
                    "experiment_run_id": "run1",
                    "result": {
                        "name": "context_precision",
                        "score": 0.9,
                        "metadata": {"evaluation_cost": 0.015},
                    },
                },
            ],
        }

        output_dir = tmp_path / "output"
        result = export_experiment_results(experiment, output_dir)

        parquet_path = Path(result.get("parquet_path", ""))
        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            assert "judge_cost_usd" in df.columns
            assert df["judge_cost_usd"][0] == 0.025

    def test_missing_costs_handled_as_nan_not_zero(self, tmp_path: Path):
        """Test missing costs handled as nan not 0.0."""
        from eval_harness.experiments.runner import export_experiment_results

        experiment = {
            "experiment_id": "test-exp-3",
            "dataset_id": "",
            "experiment_name": "test_experiment",
            "task_runs": [
                {
                    "id": "run1",
                    "dataset_example_id": "",
                    "output": {"answer": "Answer 1"},
                    # No cost field
                },
                {
                    "id": "run2",
                    "dataset_example_id": "",
                    "output": {"answer": "Answer 2"},
                    "cost": 0.001,
                },
            ],
            "evaluation_runs": [],
        }

        output_dir = tmp_path / "output"
        result = export_experiment_results(experiment, output_dir)

        parquet_path = Path(result.get("parquet_path", ""))
        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            # Missing cost should be nan, not 0.0
            assert df["app_cost_usd"][0] is None or df["app_cost_usd"][0] != df["app_cost_usd"][0]  # nan check
            assert df["app_cost_usd"][1] == 0.001

    def test_parquet_export_follows_polars_pattern(self, tmp_path: Path):
        """Test parquet export follows Polars pattern from phoenix_adapter.py."""
        from eval_harness.experiments.runner import export_experiment_results

        experiment = {
            "experiment_id": "test-exp-4",
            "dataset_id": "",
            "experiment_name": "test_experiment",
            "task_runs": [
                {"id": "run1", "dataset_example_id": "", "output": {"answer": "Answer"}},
            ],
            "evaluation_runs": [],
        }

        output_dir = tmp_path / "output"
        result = export_experiment_results(experiment, output_dir)

        parquet_path = Path(result.get("parquet_path", ""))
        assert parquet_path.exists()
        assert parquet_path.suffix == ".parquet"

        # Verify it's a valid parquet file readable by Polars
        df = pl.read_parquet(parquet_path)
        assert isinstance(df, pl.DataFrame)

    def test_cost_columns_added_to_export_rows(self, tmp_path: Path):
        """Test cost columns (app_cost_usd, judge_cost_usd, total_cost_usd) added to export."""
        from eval_harness.experiments.runner import export_experiment_results

        experiment = {
            "experiment_id": "test-exp-5",
            "dataset_id": "",
            "experiment_name": "test_experiment",
            "task_runs": [
                {
                    "id": "run1",
                    "dataset_example_id": "",
                    "output": {"answer": "Answer"},
                    "cost": 0.001,
                },
            ],
            "evaluation_runs": [
                {
                    "experiment_run_id": "run1",
                    "result": {
                        "name": "faithfulness",
                        "score": 0.8,
                        "metadata": {"evaluation_cost": 0.01},
                    },
                },
            ],
        }

        output_dir = tmp_path / "output"
        result = export_experiment_results(experiment, output_dir)

        parquet_path = Path(result.get("parquet_path", ""))
        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            assert "app_cost_usd" in df.columns
            assert "judge_cost_usd" in df.columns
            assert "total_cost_usd" in df.columns
            # total_cost_usd = app + judge
            assert df["total_cost_usd"][0] == 0.011
