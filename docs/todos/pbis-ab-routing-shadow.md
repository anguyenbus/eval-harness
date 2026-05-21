# A/B Routing & Shadow Evaluation: PBIs

**Capabilities**: A/B Routing (comparing system variants) + Shadow Evaluation (running variants offline).

**Current State**: Manual A/B via separate runs. No built-in comparison tools. Shadow only works offline.

---

## PBI-15: Add Built-in Comparison Tool

**Priority**: P1 (Usability)
**Estimate**: 4 hours
**Category**: A/B Routing → Comparison

### Problem

Comparing two JSON summaries requires manual work. Open both files, eye-ball differences, calculate deltas yourself. Error-prone and tedious.

### Acceptance Criteria

1. [ ] `eval-compare` CLI command
2. [ ] Takes two JSON file paths as input
3. [ ] Highlights metric differences
4. [ ] Shows relative and absolute changes
5. [ ] Flags regressions (> threshold)
6. [ ] Outputs markdown report

### Implementation Notes

**Create `src/eval_harness/runners/run_compare.py`**:

```python
"""
CLI runner for comparing evaluation results.

Usage:
    uv run eval-compare baseline.json current.json
"""
import json
import sys
from pathlib import Path
from typing import Any

def load_summary(path: Path) -> dict:
    """Load and validate JSON summary."""
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if "metrics_avg" not in data:
        raise ValueError(f"Invalid summary format: {path}")

    return data

def compare_summaries(
    baseline: dict,
    current: dict,
    threshold: float = 0.05
) -> dict[str, Any]:
    """
    Compare two evaluation summaries.

    Returns:
        Comparison dict with deltas and regressions
    """
    baseline_metrics = baseline.get("metrics_avg", {})
    current_metrics = current.get("metrics_avg", {})

    all_keys = set(baseline_metrics.keys()) | set(current_metrics.keys())

    comparison = {
        "baseline_file": str(baseline.get("csv_file", "unknown")),
        "current_file": str(current.get("csv_file", "unknown")),
        "threshold": threshold,
        "metrics": {}
    }

    regressions = []

    for key in sorted(all_keys):
        baseline_val = baseline_metrics.get(key, 0)
        current_val = current_metrics.get(key, 0)

        # Calculate delta
        if baseline_val != 0:
            relative_delta = (current_val - baseline_val) / baseline_val
        else:
            relative_delta = 0.0

        absolute_delta = current_val - baseline_val

        metric_info = {
            "baseline": baseline_val,
            "current": current_val,
            "absolute_delta": absolute_delta,
            "relative_delta": relative_delta,
            "status": "same"
        }

        # Determine status
        if abs(relative_delta) < threshold:
            metric_info["status"] = "same"
        elif current_val < baseline_val:
            metric_info["status"] = "regression"
            regressions.append(key)
        else:
            metric_info["status"] = "improvement"

        comparison["metrics"][key] = metric_info

    comparison["regressions"] = regressions
    comparison["summary"] = {
        "total_metrics": len(all_keys),
        "regressions": len(regressions),
        "improvements": sum(1 for m in comparison["metrics"].values()
                          if m["status"] == "improvement"),
        "unchanged": sum(1 for m in comparison["metrics"].values()
                        if m["status"] == "same")
    }

    return comparison

def format_markdown(comparison: dict) -> str:
    """Format comparison as markdown report."""
    lines = [
        "# Evaluation Comparison Report",
        "",
        f"**Baseline**: {comparison['baseline_file']}",
        f"**Current**: {comparison['current_file']}",
        f"**Threshold**: {comparison['threshold']:.1%}",
        "",
        "## Summary",
        "",
        f"- Total Metrics: {comparison['summary']['total_metrics']}",
        f"- Regressions: {comparison['summary']['regressions']}",
        f"- Improvements: {comparison['summary']['improvements']}",
        f"- Unchanged: {comparison['summary']['unchanged']}",
        "",
        "## Metric Details",
        ""
    ]

    # Table header
    lines.append("| Metric | Baseline | Current | Delta | Change |")
    lines.append("|--------|----------|---------|-------|--------|")

    for metric, info in comparison["metrics"].items():
        baseline = f"{info['baseline']:.4f}"
        current = f"{info['current']:.4f}"
        delta_rel = f"{info['relative_delta']:+.1%}"
        delta_abs = f"{info['absolute_delta']:+.4f}"

        status_emoji = {
            "regression": "🔴",
            "improvement": "🟢",
            "same": "⚪"
        }.get(info["status"], "")

        lines.append(
            f"| {metric} | {baseline} | {current} | {delta_rel} | {status_emoji} |"
        )

    # Regressions section
    if comparison["regressions"]:
        lines.append("")
        lines.append("## Regressions")
        lines.append("")
        for metric in comparison["regressions"]:
            info = comparison["metrics"][metric]
            lines.append(
                f"- **{metric}**: {info['baseline']:.4f} → {info['current']:.4f} "
                f"({info['relative_delta']:+.1%})"
            )

    return "\n".join(lines)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("baseline", type=Path, help="Baseline JSON summary")
    parser.add_argument("current", type=Path, help="Current JSON summary")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Regression threshold (default: 5%%)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for comparison report (stdout if not specified)"
    )

    args = parser.parse_args()

    # Load summaries
    baseline = load_summary(args.baseline)
    current = load_summary(args.current)

    # Compare
    comparison = compare_summaries(baseline, current, args.threshold)

    # Format output
    report = format_markdown(comparison)

    # Write or print
    if args.output:
        args.output.write_text(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)

    # Exit with error if regressions detected
    if comparison["regressions"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Register in `pyproject.toml`**:

```toml
[project.scripts]
eval-compare = "eval_harness.runners.run_compare:main"
```

**Usage**:

```bash
# Compare two runs
uv run eval-compare \
    results/baseline_summary.json \
    results/current_summary.json

# With custom threshold and output
uv run eval-compare \
    results/baseline_summary.json \
    results/current_summary.json \
    --threshold 0.03 \
    --output comparison_report.md
```

**Sample output**:

```markdown
# Evaluation Comparison Report

**Baseline**: legal_rag_bench_nano_results_20260501_120000.json
**Current**: legal_rag_bench_nano_results_20260515_140000.json
**Threshold**: 5.0%

## Summary

- Total Metrics: 6
- Regressions: 2
- Improvements: 1
- Unchanged: 3

## Metric Details

| Metric | Baseline | Current | Delta | Change |
|--------|----------|---------|-------|--------|
| answer_relevancy_score | 0.8300 | 0.8500 | +2.4% | 🟢 |
| context_precision_score | 0.2500 | 0.1450 | -42.0% | 🔴 |
| context_recall_score | 0.5000 | 0.4375 | -12.5% | 🔴 |
| faithfulness_score | 0.9000 | 0.9167 | +1.9% | 🟢 |
| relevant_passage_retrieved | 0.7500 | 0.7500 | +0.0% | ⚪ |
| total_ms | 5000.00 | 5200.00 | +4.0% | ⚪ |

## Regressions

- **context_precision_score**: 0.2500 → 0.1450 (-42.0%)
- **context_recall_score**: 0.5000 → 0.4375 (-12.5%)
```

### Definition of Done

- [ ] `eval-compare` CLI working
- [ ] Markdown report generation
- [ ] Regressions flagged with threshold
- [ ] Exit code 1 on regression
- [ ] Tests for comparison logic

---

## PBI-16: Add Statistical Testing

**Priority**: P1 (Rigor)
**Estimate**: 6 hours
**Category**: A/B Routing → Statistics

### Problem

Point estimates don't tell the whole story. Is a 0.75 vs 0.70 difference significant or noise? Without statistical tests, can't know.

### Acceptance Criteria

1. [ ] Fisher exact test for binary metrics (relevant_passage_retrieved)
2. [ ] Bootstrap CI for continuous metrics (faithfulness, etc.)
3. [ ] Mann-Whitney U for non-parametric comparison
4. [ ] Report p-values with each comparison
5. [ ] Flag statistically significant differences (p < 0.05)

### Implementation Notes

**Create `src/eval_harness/reporting/statistics.py`**:

```python
"""
Statistical tests for evaluation result comparison.
"""
import numpy as np
from scipy import stats
from typing import Tuple, List

def fisher_exact_test(
    baseline_success: int,
    baseline_total: int,
    current_success: int,
    current_total: int
) -> Tuple[float, float]:
    """
    Fisher's exact test for binary outcomes.

    Returns:
        (odds_ratio, p_value)
    """
    # Contingency table
    #           Success  Fail
    # Baseline    a        b
    # Current     c        d
    a = baseline_success
    b = baseline_total - baseline_success
    c = current_success
    d = current_total - current_success

    contingency = [[a, b], [c, d]]
    _, p_value = stats.fisher_exact(contingency)

    baseline_rate = baseline_success / baseline_total
    current_rate = current_success / current_total

    if b == 0 or d == 0:
        odds_ratio = float('inf') if current_rate > baseline_rate else 0.0
    else:
        odds_ratio = (a * d) / (b * c) if (b * c) != 0 else float('inf')

    return odds_ratio, p_value

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    arr = np.array(values)
    n = len(arr)

    if n == 0:
        return (0.0, 0.0, 0.0)

    # Resample with replacement
    bootstrapped = np.random.choice(arr, size=(n_bootstrap, n), replace=True)
    means = np.mean(bootstrapped, axis=1)

    # Calculate CI
    alpha = 1 - ci
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    mean = np.mean(arr)

    return (float(mean), float(lower), float(upper))

def mann_whitney_u(
    baseline: List[float],
    current: List[float]
) -> Tuple[float, float]:
    """
    Mann-Whitney U test (non-parametric).

    Returns:
        (statistic, p_value)
    """
    statistic, p_value = stats.mannwhitneyu(baseline, current, alternative='two-sided')
    return (float(statistic), float(p_value))

def compare_metrics_statistical(
    baseline_csv: Path,
    current_csv: Path
) -> dict:
    """
    Perform statistical comparison between two result CSVs.

    Returns:
        Dict with statistical test results per metric
    """
    import pandas as pd

    baseline_df = pd.read_csv(baseline_csv)
    current_df = pd.read_csv(current_csv)

    results = {}

    # Binary metrics
    binary_metrics = ["relevant_passage_retrieved"]
    for metric in binary_metrics:
        if metric not in baseline_df.columns:
            continue

        baseline_success = baseline_df[metric].sum()
        baseline_total = len(baseline_df)
        current_success = current_df[metric].sum()
        current_total = len(current_df)

        odds_ratio, p_value = fisher_exact_test(
            baseline_success, baseline_total,
            current_success, current_total
        )

        results[metric] = {
            "test": "fisher_exact",
            "baseline_rate": baseline_success / baseline_total,
            "current_rate": current_success / current_total,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "significant": p_value < 0.05
        }

    # Continuous metrics
    continuous_metrics = [
        "faithfulness_score",
        "context_precision_score",
        "context_recall_score",
        "answer_relevancy_score"
    ]
    for metric in continuous_metrics:
        if metric not in baseline_df.columns:
            continue

        baseline_vals = baseline_df[metric].dropna().tolist()
        current_vals = current_df[metric].dropna().tolist()

        if len(baseline_vals) == 0 or len(current_vals) == 0:
            continue

        # Bootstrap CI for each
        bl_mean, bl_lower, bl_upper = bootstrap_ci(baseline_vals)
        cur_mean, cur_lower, cur_upper = bootstrap_ci(current_vals)

        # Mann-Whitney U test
        _, p_value = mann_whitney_u(baseline_vals, current_vals)

        results[metric] = {
            "test": "mann_whitney",
            "baseline_mean": bl_mean,
            "baseline_ci": (bl_lower, bl_upper),
            "current_mean": cur_mean,
            "current_ci": (cur_lower, cur_upper),
            "p_value": p_value,
            "significant": p_value < 0.05
        }

    return results
```

**Integrate into `eval-compare`**:

```python
# Add --statistical flag
parser.add_argument(
    "--statistical",
    action="store_true",
    help="Perform statistical significance tests"
)

# In comparison function
if args.statistical:
    stats_results = compare_metrics_statistical(
        baseline.replace('.json', '.csv'),
        current.replace('.json', '.csv')
    )
    comparison["statistical"] = stats_results
```

**Report format**:

```markdown
## Statistical Significance

| Metric | Test | Baseline | Current | P-value | Significant |
|--------|------|----------|---------|---------|-------------|
| relevant_passage_retrieved | Fisher exact | 75.0% (6/8) | 62.5% (5/8) | 1.000 | No |
| faithfulness_score | Mann-Whitney U | 0.90 [0.85-0.95] | 0.92 [0.88-0.96] | 0.632 | No |
| context_precision_score | Mann-Whitney U | 0.25 [0.15-0.35] | 0.15 [0.10-0.20] | 0.034 | **Yes** |
```

### Definition of Done

- [ ] Fisher exact test for binary metrics
- [ ] Bootstrap CI for continuous metrics
- [ ] Mann-Whitney U test
- [ ] P-values reported
- [ ] Significant differences flagged
- [ ] Tests documented

---

## PBI-17: Add Side-by-Side HTML Report

**Priority**: P2 (Visualization)
**Estimate**: 4 hours
**Category**: A/B Routing → Visualization

### Problem

Text-based comparison is functional but not visual. Can't see patterns at a glance.

### Acceptance Criteria

1. [ ] Generate HTML report with side-by-side metric comparison
2. [ ] Visual indicators (color coding) for regression/improvement
3. [ ] Sparkline charts for metric distributions
4. [ ] Per-question diff view
5. [ ] Export to standalone HTML file

### Implementation Notes

**Create `src/eval_harness/reporting/ab_report.py`**:

```python
def generate_ab_html_report(
    baseline_path: Path,
    current_path: Path,
    output_path: Path
) -> None:
    """Generate side-by-side A/B comparison HTML report."""

    # Load both CSVs
    baseline_df = pd.read_csv(baseline_path.with_suffix('.csv'))
    current_df = pd.read_csv(current_path.with_suffix('.csv'))

    # Merge on query_id
    merged = pd.merge(
        baseline_df,
        current_df,
        on='query_id',
        suffixes=('_baseline', '_current')
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>A/B Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .regression {{ background-color: #fee; border-left: 4px solid #f00; }}
        .improvement {{ background-color: #efe; border-left: 4px solid #0f0; }}
        .neutral {{ background-color: #fff; border-left: 4px solid #ccc; }}
        table {{ width: 100%%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        .sparkline {{ width: 100px; height: 30px; }}
    </style>
</head>
<body>
    <h1>A/B Comparison Report</h1>
    <p><strong>Baseline:</strong> {baseline_path.name}</p>
    <p><strong>Current:</strong> {current_path.name}</p>

    <h2>Metric Summary</h2>
    {generate_metric_cards(baseline_df, current_df)}

    <h2>Per-Question Comparison</h2>
    {generate_question_table(merged)}
</body>
</html>"""

    output_path.write_text(html)
```

### Definition of Done

- [ ] HTML report generated
- [ ] Color coding for regression/improvement
- [ ] Per-question comparison table
- [ ] Standalone (no external dependencies)
- [ ] Tested with real results

---

## PBI-18: Build Live Shadow Mode Server

**Priority**: P2 (Architecture)
**Estimate**: 16 hours
**Category**: Shadow Evaluation → Server

### Problem

Shadow evaluation only works offline. No way to shadow production traffic in real-time.

### Acceptance Criteria

1. [ ] FastAPI server that accepts RAG queries
2. [ ] Mirrors queries to shadow variant
3. [ ] Stores both production and shadow results
4. [ ] Comparison API for paired results
5. [ ] Graceful degradation if shadow fails

### Implementation Notes

**Create `src/eval_harness/shadow/server.py`**:

```python
"""
Shadow mode server for live RAG evaluation.

Mirrors production traffic to shadow variant for comparison.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

app = FastAPI(title="Eval Shadow Server")

class RAGQuery(BaseModel):
    question: str
    query_id: Optional[str] = None
    production_result: Optional[dict] = None

class ShadowResponse(BaseModel):
    query_id: str
    production: Optional[dict]
    shadow: dict
    comparison: dict

# In-memory storage (use Redis for production)
shadow_results = {}

@app.post("/shadow/query")
async def shadow_query(query: RAGQuery) -> ShadowResponse:
    """
    Mirror query to shadow variant.

    If production_result provided, compare with shadow.
    """
    query_id = query.query_id or str(uuid.uuid4())

    # Call shadow RAG system
    shadow_result = call_shadow_rag(query.question)

    # Compare if production provided
    comparison = {}
    if query.production_result:
        comparison = compare_results(query.production_result, shadow_result)

    result = ShadowResponse(
        query_id=query_id,
        production=query.production_result,
        shadow=shadow_result,
        comparison=comparison
    )

    shadow_results[query_id] = result
    return result

@app.get("/shadow/results/{query_id}")
async def get_shadow_result(query_id: str) -> ShadowResponse:
    """Get shadow result by query ID."""
    if query_id not in shadow_results:
        raise HTTPException(status_code=404, detail="Query not found")
    return shadow_results[query_id]

@app.get("/shadow/stats")
async def get_shadow_stats() -> dict:
    """Get shadow mode statistics."""
    if not shadow_results:
        return {"total_queries": 0}

    # Calculate agreement rate, etc.
    agreements = sum(
        1 for r in shadow_results.values()
        if r.comparison.get("answers_match", False)
    )

    return {
        "total_queries": len(shadow_results),
        "agreement_rate": agreements / len(shadow_results),
        "shadow_errors": sum(
            1 for r in shadow_results.values()
            if r.shadow.get("error")
        )
    }
```

### Definition of Done

- [ ] FastAPI server functional
- [ ] Shadow query endpoint working
- [ ] Comparison logic implemented
- [ ] Stats endpoint working
- [ ] Dockerfile for deployment
- [ ] API documentation

---

## PBI-19: Add Automatic Result Sync

**Priority**: P2 (Automation)
**Estimate**: 4 hours
**Category**: Shadow Evaluation → Sync

### Problem

Shadow and production results stored separately. Manual matching required.

### Acceptance Criteria

1. [ ] Match results by query_id
2. [ ] Handle unmatched queries (log warnings)
3. [ ] Generate paired CSV for analysis
4. [ ] Calculate agreement metrics

### Definition of Done

- [ ] Automatic matching working
- [ ] Unmatched queries logged
- [ ] Paired CSV generated
- [ ] Agreement metrics calculated

---

## Dependencies

```
PBI-15 (Comparison tool) ────┐
                             ├──→ Enable PBI-16 (Statistical tests)
PBI-16 (Statistical) ────────┘
                                    ↓
PBI-17 (HTML report) ──────────────┘ (Visualization)

PBI-18 (Shadow server) ───→ Enables PBI-19 (Result sync)
```

## Summary Table

| PBI | Priority | Estimate | Dependencies |
|-----|----------|----------|--------------|
| Built-in comparison tool | P1 | 4h | None |
| Statistical testing | P1 | 6h | PBI-15 |
| Side-by-side HTML report | P2 | 4h | PBI-15 |
| Live shadow mode server | P2 | 16h | None |
| Automatic result sync | P2 | 4h | PBI-18 |

**Total P1**: 10 hours
**Total all**: 34 hours
