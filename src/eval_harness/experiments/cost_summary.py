"""
Cost summary calculation for RAG evaluation experiments.

This module provides functionality to calculate cost summaries from exported
experiment parquet files. Costs are separated into:
- Application cost: LLM API calls for RAG generation (from Phoenix cost column)
- Judge cost: LLM-as-judge evaluation costs (from DeepEval evaluation_cost)

The cost column name was verified against Phoenix on the verification date.
Re-run scripts/verify_phoenix_cost_column.py if upgrading Phoenix.

Example:
    >>> from pathlib import Path
    >>> from eval_harness.experiments.cost_summary import cost_summary
    >>> summary = cost_summary(Path("experiment_results.parquet"))
    >>> print(f"Total cost: ${summary['total_cost_usd']:.4f}")
    >>> print(f"Judge-to-app ratio: {summary['judge_to_app_ratio']}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from beartype import beartype
import polars as pl


# Verified against arize-phoenix==4.0.0 on 2026-05-26.
# Re-run scripts/verify_phoenix_cost_column.py if upgrading Phoenix.
APP_COST_COLUMN: Final[str] = "cost"

# Judge cost column prefixes in exported parquet
JUDGE_COST_COLUMNS: Final[tuple[str, ...]] = (
    "judge_faithfulness_cost_usd",
    "judge_context_precision_cost_usd",
    "judge_context_recall_cost_usd",
    "judge_answer_relevancy_cost_usd",
)

# Metric names for judge cost breakdown
METRIC_NAMES: Final[dict[str, str]] = {
    "judge_faithfulness_cost_usd": "faithfulness",
    "judge_context_precision_cost_usd": "context_precision",
    "judge_context_recall_cost_usd": "context_recall",
    "judge_answer_relevancy_cost_usd": "answer_relevancy",
}


@beartype
def cost_summary(parquet_path: Path) -> dict[str, Any]:
    """
    Compute cost summary from exported experiment parquet.

    Reads cost data from parquet file exported by export_experiment_results()
    and calculates aggregated statistics. No IO to Phoenix - all data comes
    from the parquet export.

    Args:
        parquet_path: Path to parquet file exported by export_experiment_results().

    Returns:
        Dictionary with cost summary statistics:
            - experiment_id: Experiment identifier from parquet metadata
            - n_rows: Number of rows in the dataset
            - app_cost_usd: Total application cost in USD (rounded to 4 decimals)
            - judge_cost_usd: Total judge cost in USD (rounded to 4 decimals)
            - total_cost_usd: Total cost (app + judge) in USD (rounded to 4 decimals)
            - mean_cost_per_row_usd: Mean total cost per row (rounded to 4 decimals)
            - p95_cost_per_row_usd: 95th percentile cost per row using quantile(0.95)
            - judge_cost_by_metric_usd: Dict of judge costs broken down by metric
            - judge_to_app_ratio: Ratio of judge to app cost (None if app_cost is 0)

    Raises:
        RuntimeError: If required cost columns are missing from the parquet.
        FileNotFoundError: If parquet_path does not exist.

    Example:
        >>> from pathlib import Path
        >>> from eval_harness.experiments.cost_summary import cost_summary
        >>> summary = cost_summary(Path("experiment_results.parquet"))
        >>> print(f"Total cost: ${summary['total_cost_usd']:.4f}")
        >>> print(f"Judge-to-app ratio: {summary['judge_to_app_ratio']}")

    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pl.read_parquet(parquet_path)

    # Verify required columns exist
    if "app_cost_usd" not in df.columns:
        raise RuntimeError(
            f"Expected column 'app_cost_usd' not in experiment dataframe. "
            f"Re-run scripts/verify_phoenix_cost_column.py."
        )
    if "judge_cost_usd" not in df.columns:
        raise RuntimeError(
            f"Expected column 'judge_cost_usd' not in experiment dataframe. "
            f"Ensure export_experiment_results() includes judge cost extraction."
        )

    # Count rows (excluding null costs)
    n_rows = len(df)

    # Calculate total costs (Polars handles NaN by skipping in sum)
    app_total = df["app_cost_usd"].fill_nan(0).sum()
    judge_total = df["judge_cost_usd"].fill_nan(0).sum()
    total_cost = app_total + judge_total

    # Calculate per-row total costs
    total_per_row = df["app_cost_usd"].fill_nan(0) + df["judge_cost_usd"].fill_nan(0)

    # Calculate statistics
    mean_cost_per_row = total_per_row.mean()
    p95_cost_per_row = total_per_row.quantile(0.95, interpolation="midpoint")

    # Judge cost breakdown by metric
    judge_by_metric: dict[str, float] = {}
    for col_name, metric_name in METRIC_NAMES.items():
        if col_name in df.columns:
            metric_total = df[col_name].fill_nan(0).sum()
            if metric_total > 0:
                judge_by_metric[metric_name] = round(float(metric_total), 4)

    # Calculate judge-to-app ratio
    judge_to_app_ratio = (
        round(float(judge_total / app_total), 2) if app_total > 0 else None
    )

    # Extract experiment_id if present
    experiment_id = ""
    if "experiment_id" in df.columns:
        # Get first non-null value
        id_val = df["experiment_id"].drop_nulls().first()
        if id_val is not None:
            experiment_id = str(id_val)

    return {
        "experiment_id": experiment_id,
        "n_rows": n_rows,
        "app_cost_usd": round(float(app_total), 4),
        "judge_cost_usd": round(float(judge_total), 4),
        "total_cost_usd": round(float(total_cost), 4),
        "mean_cost_per_row_usd": round(float(mean_cost_per_row), 4),
        "p95_cost_per_row_usd": round(float(p95_cost_per_row), 4),
        "judge_cost_by_metric_usd": judge_by_metric,
        "judge_to_app_ratio": judge_to_app_ratio,
    }
