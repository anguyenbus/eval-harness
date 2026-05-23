# Metrics Retrieval: PBIs

**Capability**: Output and present evaluation metrics in useful formats (JSON, CSV, HTML, time-series, visualization).

**Current State**:
- JSON summary with point estimates only
- CSV with per-query results
- HTML generator exists but **broken** (column mismatch)
- No confidence intervals
- No time-series tracking
- No visualization

---

## adversarial Review Questions (Answer Before Review)

### Q1: "html_summary.py is broken. Why is this P0 and not P2?"

**Answer**: Broken tool > no tool.

**Evidence**:

[`html_summary.py:24`](src/eval_harness/reporting/html_summary.py:24) expects:
```python
pass_count = len(df[df["label"] == "pass"])  # ← Expects 'label' column
score_mean = df["score"].mean()  # ← Expects 'score' column
```

**Actual CSV columns** (from `results/*.csv`):
```csv
query_id,question,gold_answer,generated_answer,faithfulness_score,context_precision_score,context_recall_score,answer_relevancy_score,judge_verdict,total_ms,error,...
```

**Result**: `KeyError: 'label'` or `KeyError: 'score'` when running `eval-html`.

**adversarial reviewer says**: "You have an HTML report feature. I try to use it. It crashes. Now I think the project is unmaintained. Either fix it or delete it."

**PBI-36 addresses this**.

---

### Q2: "Why confidence intervals? Point estimates work fine."

**Answer**: Point estimates without CIs mislead.

**Evidence**:

```json
{
  "metrics_avg": {
    "faithfulness_score": 0.75
  },
  "total_processed": 5
}
```

**Question**: Is 0.75 good? Is it 0.75 ± 0.05 (precise) or 0.75 ± 0.35 (meaningless)?

**adversarial reviewer says**: "You claim your system scores 0.75. I run it again. I get 0.65. I claim you're wrong. You claim variance. Where's your CI?"

**Without CI**:
- Can't distinguish noise from signal
- Can't detect regressions (is 0.65 a regression or variance?)
- Can't compare systems (is 0.75 vs 0.73 significant?)

**PBI-37 addresses this**.

---

### Q3: "Why document details.json? Just read the file."

**Answer**: No, read the docs first.

**Evidence**:

[`details.json`](results/legal_rag_bench_nano_results_20260521_211101_details.json) structure:
```json
{
  "queries": [
    {
      "query_id": 1,
      "reasoning": {
        "faithfulness": {
          "reason": "...",
          "claims": [...],
          "truths": [...]
        },
        "context_precision": {
          "reason": "...",
          "verdicts": [...]
        }
      }
    }
  ]
}
```

**adversarial reviewer says**: "What's in `reasoning`? What's a `claim` vs `truth`? How do I interpret this? Don't make me reverse-engineer your data structure."

**PBI-38 addresses this**.

---

### Q4: "Time-series tracking? Why not just query the database?"

**Answer**: We don't have a database yet. And query ≠ time-series.

**Evidence**:

Current storage: flat files with timestamps in filenames:
```
results/legal_rag_bench_nano_results_20260521_211101.json
results/legal_rag_bench_nano_results_20260521_210254.json
results/legal_rag_bench_nano_results_20260521_203524.json
```

**Question**: How do I plot faithfulness over time?

**Answer**:
1. Parse filenames to extract timestamps
2. Load each JSON
3. Extract metrics
4. Sort by timestamp
5. Plot

**That's 100 lines of Python. Every time.**

**adversarial reviewer says**: "I want to see if my system is improving. I have 50 result files. How do I plot faithfulness vs time? Don't tell me to write a script."

**PBI-39 addresses this**.

---

### Q5: "Visualization? Just export to CSV and use Excel."

**Answer**: Not acceptable for automated reporting.

**Evidence**:

Current workflow:
1. Run eval
2. Open CSV in Excel
3. Select columns
4. Insert chart
5. Format chart
6. Save as PNG
7. Copy to report

**Every time. For every eval.**

**adversarial reviewer says**: "I want a PRD-ready report. Not 'open Excel and make charts'. Automation matters."

**PBI-40 addresses this**.

---

## PBI-36: Fix html_summary.py (Column Mismatch)

**Priority**: P0 (Correctness)
**Estimate**: 2 hours
**Category**: Metrics Retrieval → HTML Reporting

### Problem Statement

[`html_summary.py`](src/eval_harness/reporting/html_summary.py:24) expects hardcoded columns that don't match actual CSV format.

**Evidence of mismatch**:

**Code expects**:
```python
# html_summary.py:24
df[df["label"] == "pass"]  # column: label
df["score"].mean()  # column: score
df.groupby("question_id")  # column: question_id
```

**Actual CSV has** (from `results/legal_rag_bench_nano_results_*.csv`):
```csv
query_id,question,gold_answer,generated_answer,faithfulness_score,judge_verdict,...
```

| Expected by code | Actual in CSV | Result |
|------------------|---------------|--------|
| `label` | `judge_verdict` | KeyError |
| `score` | `faithfulness_score`, `context_precision_score`, etc. | KeyError |
| `question_id` | `query_id` | Wrong grouping |

**Impact**: HTML generator crashes on any real result file.

### Root Cause

Code was written for a different CSV format (likely parsing eval, not RAG eval).

### Acceptance Criteria

1. [ ] Auto-detect CSV format (RAG vs parsing)
2. [ ] Use correct columns for RAG format (`judge_verdict`, `faithfulness_score`, etc.)
3. [ ] Handle parsing format too (backward compatibility)
4. [ ] Generate working HTML report
5. [ ] CLI command `eval-html` works end-to-end

### Implementation

**Rewrite `src/eval_harness/reporting/html_summary.py`**:

```python
"""HTML summary report generator.

FIXED: Now works with actual RAG and parsing CSV formats.
"""
from pathlib import Path
from typing import Any
import pandas as pd


def generate_summary(
    results_path: Path,
    output_path: Path | None = None,
    metric_column: str | None = None
) -> str:
    """
    Generate HTML summary report from evaluation results CSV.

    Auto-detects CSV format (RAG vs parsing) and generates appropriate report.

    Args:
        results_path: Path to CSV file
        output_path: Path for HTML output (optional)
        metric_column: Primary metric to use (auto-detected if None)

    Returns:
        HTML string
    """
    df = pd.read_csv(results_path)

    # Detect format
    is_rag = "judge_verdict" in df.columns
    is_parsing = "nid" in df.columns or "page_number" in df.columns

    if is_rag:
        html = _generate_rag_html(df, results_path, metric_column)
    elif is_parsing:
        html = _generate_parsing_html(df, results_path, metric_column)
    else:
        # Generic format
        html = _generate_generic_html(df, results_path, metric_column)

    # Write to file if output path specified
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html


def _generate_rag_html(
    df: pd.DataFrame,
    results_path: Path,
    metric_column: str | None
) -> str:
    """Generate HTML for RAG evaluation results."""

    # Default metric: faithfulness_score
    if metric_column is None:
        metric_column = "faithfulness_score"

    # Calculate pass/fail from judge_verdict
    total = len(df)
    passed = len(df[df["judge_verdict"] == "PASS"])
    failed = len(df[df["judge_verdict"] == "NEEDS_REVIEW"])
    errored = len(df[df["error"].notna() & (df["error"] != "")])

    pass_rate = (passed / total * 100) if total > 0 else 0

    # Metric stats
    metrics = [
        "faithfulness_score",
        "context_precision_score",
        "context_recall_score",
        "answer_relevancy_score"
    ]

    metric_stats = {}
    for m in metrics:
        if m in df.columns:
            valid = df[df[m].notna()]
            if len(valid) > 0:
                metric_stats[m] = {
                    "mean": valid[m].mean(),
                    "min": valid[m].min(),
                    "max": valid[m].max(),
                    "count": len(valid)
                }

    # Determine stat class
    stat_class = "pass" if pass_rate >= 80 else "fail"

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: #f5f5f5; padding: 20px; border-radius: 8px; flex: 1; text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
        .metric-name {{ font-family: monospace; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>RAG Evaluation Summary</h1>
    <p><strong>File:</strong> {results_path.name}</p>

    <div class="summary">
        <div class="stat-box">
            <div class="stat-value {stat_class}">{pass_rate:.1f}%</div>
            <div class="stat-label">Pass Rate</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Queries</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{passed}</div>
            <div class="stat-label">Passed</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{failed}</div>
            <div class="stat-label">Needs Review</div>
        </div>
    </div>

    <h2>Metric Statistics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Mean</th>
            <th>Min</th>
            <th>Max</th>
            <th>Count</th>
        </tr>
"""

    for m, stats in metric_stats.items():
        html += f"""
        <tr>
            <td class="metric-name">{m}</td>
            <td>{stats['mean']:.4f}</td>
            <td>{stats['min']:.4f}</td>
            <td>{stats['max']:.4f}</td>
            <td>{stats['count']}</td>
        </tr>
"""

    html += """
    </table>

    <h2>Per-Query Results</h2>
    <table>
        <tr>
            <th>Query ID</th>
            <th>Question</th>
            <th>Verdict</th>
"""

    for m in metrics:
        if m in df.columns:
            html += f"<th>{m}</th>\n"

    html += """
        </tr>
"""

    for _, row in df.iterrows():
        verdict_class = "pass" if row.get("judge_verdict") == "PASS" else "fail"
        question_preview = str(row.get("question", ""))[:50] + "..." if len(str(row.get("question", ""))) > 50 else row.get("question", "")

        html += f"""
        <tr>
            <td>{row['query_id']}</td>
            <td>{question_preview}</td>
            <td class="{verdict_class}">{row.get('judge_verdict', '')}</td>
"""

        for m in metrics:
            if m in df.columns:
                val = row.get(m)
                if pd.notna(val):
                    html += f"<td>{val:.4f}</td>\n"
                else:
                    html += "<td>-</td>\n"

        html += "        </tr>\n"

    html += """
    </table>
</body>
</html>"""

    return html


def _generate_parsing_html(
    df: pd.DataFrame,
    results_path: Path,
    metric_column: str | None
) -> str:
    """Generate HTML for parsing evaluation results."""

    # Parsing metrics: different structure
    # Has columns like: nid, page_number, bbox_accuracy, etc.

    total = len(df)

    # Calculate metrics based on parsing columns
    # This is a placeholder - adjust based on actual parsing CSV format
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Parsing Evaluation Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Parsing Evaluation Summary</h1>
    <p><strong>File:</strong> {results_path.name}</p>
    <p><strong>Total Documents:</strong> {total}</p>
    <!-- Add parsing-specific metrics here -->
</body>
</html>"""

    return html


def _generate_generic_html(
    df: pd.DataFrame,
    results_path: Path,
    metric_column: str | None
) -> str:
    """Generate HTML for generic CSV format."""

    total = len(df)

    # Try to find numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>Evaluation Summary</h1>
    <p><strong>File:</strong> {results_path.name}</p>
    <p><strong>Total Rows:</strong> {total}</p>

    <h2>Data Preview</h2>
    <table>
        <tr>
"""

    for col in df.columns[:10]:  # First 10 columns
        html += f"<th>{col}</th>\n"

    html += "        </tr>\n"

    # First 5 rows
    for _, row in df.head(5).iterrows():
        html += "        <tr>\n"
        for col in df.columns[:10]:
            val = row[col]
            if pd.isna(val):
                html += "            <td>-</td>\n"
            elif isinstance(val, str) and len(val) > 50:
                html += f"            <td>{val[:50]}...</td>\n"
            else:
                html += f"            <td>{val}</td>\n"
        html += "        </tr>\n"

    html += """
    </table>
</body>
</html>"""

    return html


def main() -> None:
    """CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate HTML summary from evaluation results"
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to CSV file with evaluation results"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output HTML file path (default: same as input with .html extension)"
    )
    parser.add_argument(
        "-m", "--metric",
        type=str,
        help="Primary metric column (auto-detected if not specified)"
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = args.results_path.with_suffix(".html")

    try:
        generate_summary(args.results_path, args.output, args.metric)
        print(f"[INFO] Generated HTML report: {args.output}")
    except Exception as e:
        print(f"[ERROR] Failed to generate HTML: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Add to `pyproject.toml`**:

```toml
[project.scripts]
eval-html = "eval_harness.reporting.html_summary:main"
```

**Usage**:

```bash
# Generate HTML from RAG eval results
uv run eval-html results/legal_rag_bench_nano_results_20260521_211101.csv

# Specify output path
uv run eval-html results/*.csv -o report.html

# Specify metric column
uv run eval-html results/*.csv -m context_precision_score
```

### Definition of Done

- [ ] RAG format working (`judge_verdict`, `faithfulness_score`, etc.)
- [ ] Parsing format working (backward compatibility)
- [ ] Generic format fallback
- [ ] CLI command `eval-html` added
- [ ] Tested with actual result files
- [ ] Pass/fail from `judge_verdict` (not `label`)
- [ ] Multiple metrics shown in table

### What a adversarial Reviewer Will Ask

**Q**: "What if my CSV has different columns?"

**A**: Generic format fallback shows all columns. You can then add support for your format.

**Q**: "Why not use a template engine like Jinja2?"

**A**: Overkill. f-strings work fine. Jinja2 adds dependency for simple HTML generation.

**Q**: "Can I customize the HTML?"

**A**: Not in PBI-36. Future: `--template` flag to use custom Jinja2 template.

---

## PBI-37: Add Confidence Intervals

**Priority**: P1 (Statistics)
**Estimate**: 4 hours
**Category**: Metrics Retrieval → Statistics

### Problem Statement

Current JSON reports point estimates without uncertainty. This misleads when sample size is small.

**Evidence**:

```json
{
  "metrics_avg": {
    "faithfulness_score": 0.75
  },
  "total_processed": 5
}
```

**Question**: Is 0.75 a precise estimate or a guess?

**Answer**: With n=5, 95% CI is approximately ±0.35. That's huge. But the JSON doesn't show this.

**Statistical background**:

| Metric | Type | CI Method |
|--------|------|-----------|
| faithfulness_score | Continuous (0-1) | Bootstrap |
| context_precision_score | Continuous (0-1) | Bootstrap |
| context_recall_score | Continuous (0-1) | Bootstrap |
| judge_verdict | Binary (PASS/NEEDS_REVIEW) | Clopper-Pearson |

### Acceptance Criteria

1. [ ] Compute 95% CI for each metric
2. [ ] Use bootstrap for continuous metrics
3. [ ] Use Clopper-Pearson for binary metrics
4. [ ] Include CI in JSON summary (`ci_lower`, `ci_upper`)
5. [ ] Show CI in HTML report
6. [ ] Document confidence level

### Implementation

**Create `src/eval_harness/metrics/confidence_intervals.py`**:

```python
"""Confidence interval computation for evaluation metrics."""
from typing import Any
import numpy as np


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = None
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        values: List of metric values (can contain NaN)
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        (mean, ci_lower, ci_upper)
    """
    # Remove NaN values
    clean_values = [v for v in values if not np.isnan(v)]

    if len(clean_values) == 0:
        return np.nan, np.nan, np.nan

    if len(clean_values) < 2:
        # Not enough data for bootstrap
        return clean_values[0], clean_values[0], clean_values[0]

    rng = np.random.default_rng(seed)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(clean_values, size=len(clean_values), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute CI percentiles
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(np.mean(clean_values)), float(ci_lower), float(ci_upper)


def clopper_pearson_ci(
    successes: int,
    n: int,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Compute Clopper-Pearson confidence interval for binary proportion.

    Args:
        successes: Number of successes
        n: Total number of trials
        confidence: Confidence level (default 0.95)

    Returns:
        (proportion, ci_lower, ci_upper)
    """
    from scipy import stats

    if n == 0:
        return np.nan, np.nan, np.nan

    proportion = successes / n
    alpha = 1 - confidence

    # Clopper-Pearson interval
    ci_lower = stats.beta.ppf(alpha / 2, successes, n - successes + 1) if successes > 0 else 0.0
    ci_upper = stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes) if successes < n else 1.0

    return float(proportion), float(ci_lower), float(ci_upper)


def compute_all_cis(
    df: "pd.DataFrame",
    confidence: float = 0.95
) -> dict[str, dict[str, float]]:
    """
    Compute confidence intervals for all metrics in a DataFrame.

    Args:
        df: DataFrame with metric columns
        confidence: Confidence level

    Returns:
        Dict mapping metric name to {mean, ci_lower, ci_upper}
    """
    import pandas as pd

    results = {}

    # Continuous metrics (use bootstrap)
    continuous_metrics = [
        "faithfulness_score",
        "context_precision_score",
        "context_recall_score",
        "answer_relevancy_score"
    ]

    for metric in continuous_metrics:
        if metric in df.columns:
            values = df[metric].dropna().tolist()
            if values:
                mean, ci_lower, ci_upper = bootstrap_ci(values, confidence=confidence)
                results[metric] = {
                    "mean": mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n": len(values)
                }

    # Binary metrics (use Clopper-Pearson)
    if "judge_verdict" in df.columns:
        total = len(df[df["judge_verdict"].notna()])
        passed = len(df[df["judge_verdict"] == "PASS"])

        proportion, ci_lower, ci_upper = clopper_pearson_ci(passed, total, confidence)
        results["pass_rate"] = {
            "mean": proportion,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": total
        }

    return results
```

**Integrate into JSON output**:

```python
# In run_rag_eval.py, after computing metrics
from eval_harness.metrics.confidence_intervals import compute_all_cis

# Load results CSV
df = pd.read_csv(csv_path)

# Compute CIs
ci_results = compute_all_cis(df, confidence=0.95)

# Add to JSON
json_output["metrics_avg"] = {}
json_output["metrics_ci"] = {}

for metric_name, stats in ci_results.items():
    json_output["metrics_avg"][metric_name] = stats["mean"]
    json_output["metrics_ci"][metric_name] = {
        "ci_lower": stats["ci_lower"],
        "ci_upper": stats["ci_upper"],
        "n": stats["n"]
    }
```

**Output format**:

```json
{
  "metrics_avg": {
    "faithfulness_score": 0.75
  },
  "metrics_ci": {
    "faithfulness_score": {
      "ci_lower": 0.60,
      "ci_upper": 0.90,
      "n": 10
    }
  },
  "confidence_level": 0.95
}
```

**Add to HTML report**:

```python
# In _generate_rag_html()
for m, stats in metric_stats.items():
    ci = ci_results.get(m, {})
    ci_lower = ci.get("ci_lower", stats["mean"])
    ci_upper = ci.get("ci_upper", stats["mean"])

    html += f"""
    <tr>
        <td class="metric-name">{m}</td>
        <td>{stats['mean']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]</td>
        ...
    </tr>
"""
```

### Dependencies

Add to [`pyproject.toml`](pyproject.toml:11):

```toml
dependencies = [
    # ... existing
    "numpy",
    "scipy",  # For Clopper-Pearson
]
```

### Definition of Done

- [ ] Bootstrap CI implemented for continuous metrics
- [ ] Clopper-Pearson CI implemented for binary metrics
- [ ] CIs added to JSON output
- [ ] CIs shown in HTML report
- [ ] Confidence level documented
- [ ] Tests for CI computation
- [ ] Handles NaN values correctly
- [ ] Handles small samples (n < 2) gracefully

### What a adversarial Reviewer Will Ask

**Q**: "Why bootstrap 10,000 times? That's slow."

**A**: 10K is standard for 95% CI. Takes ~100ms for 100 values. Negligible compared to LLM calls.

**Q**: "Why Clopper-Pearson instead of normal approximation?"

**A**: Normal approximation (Wald) fails for small n or extreme p. Clopper-Pearson is exact.

**Q**: "What if n < 2?"

**A**: CI = [value, value]. Degenerate interval. Document this limitation.

**Q**: "Why not use Wilson score interval for binary?"

**A**: Clopper-Pearson is more conservative. Wilson is fine too, but Clopper-Pearson is standard for small n.

---

## PBI-38: Add details.json Sample to Docs

**Priority**: P1 (Documentation)
**Estimate**: 1 hour
**Category**: Metrics Retrieval → Documentation

### Problem Statement

`details.json` contains LLM judge reasoning but has no documentation. Reviewers can't interpret the structure.

**Evidence**:

[`details.json`](results/legal_rag_bench_nano_results_20260521_211101_details.json) structure exists but isn't documented:

```json
{
  "queries": [
    {
      "query_id": 1,
      "reasoning": {
        "faithfulness": {
          "reason": "...",
          "claims": [...],
          "truths": [...]
        },
        "context_precision": {
          "reason": "...",
          "verdicts": [...]
        }
      }
    }
  ]
}
```

**adversarial reviewer says**: "What's a `claim` vs `truth`? What's a `verdict`? Don't make me read the code."

### Acceptance Criteria

1. [ ] Extract real example (Bob & Ted juror query)
2. [ ] Document structure with annotations
3. [ ] Explain each field
4. [ ] Add to docs
5. [ ] Explain how to use reasoning for debugging

### Implementation

**Create `docs/research/details-json-format.md`**:

```markdown
# details.json Format

## Overview

`details.json` contains the LLM judge's reasoning for each metric. Use it to understand why a query got a specific score.

**Location**: Generated alongside JSON summary, named `{dataset}_results_{timestamp}_details.json`

---

## Structure

```json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "timestamp": "20260521_211101",
  "total_queries": 10,
  "evaluation_framework": "deepeval",
  "framework_version": "4.0.3",
  "judge_model": "gpt-4o",
  "queries": [
    {
      "query_id": 1,
      "question": "...",
      "reasoning": {
        "faithfulness": { ... },
        "context_precision": { ... },
        "context_recall": { ... },
        "answer_relevancy": { ... }
      }
    }
  ]
}
```

---

## Per-Metric Reasoning

### Faithfulness

Measures whether the generated answer contradicts the retrieved context.

**Structure**:
```json
{
  "faithfulness": {
    "score": 1.0,
    "reason": "The score is 1.00 because there are no contradictions...",
    "claims": [
      {
        "text": "The provided context does not specifically address whether a juror..."
      }
    ],
    "truths": [
      {
        "text": "A person who has experienced a particular life event..."
      }
    ]
  }
}
```

**Fields**:
- `score`: 0.0 to 1.0 (1.0 = no contradictions)
- `reason`: LLM's explanation
- `claims`: Statements extracted from the generated answer
- `truths`: Facts extracted from the retrieved context

**Interpretation**:
- The LLM extracts claims from the answer
- The LLM extracts truths from the context
- If any claim contradicts any truth, score < 1.0

### Context Precision

Measures whether relevant nodes are ranked high in retrieval.

**Structure**:
```json
{
  "context_precision": {
    "score": 0.25,
    "reason": "The score is 0.25 because the relevant node...",
    "verdicts": [
      {
        "verdict": "no",
        "reason": "The context discusses excusing jurors based on personal experiences..."
      }
    ]
  }
}
```

**Fields**:
- `score`: 0.0 to 1.0 (1.0 = all relevant nodes ranked highest)
- `reason`: LLM's explanation
- `verdicts`: Per-node judgment (yes/no)

**Interpretation**:
- For each retrieved node, LLM asks: "Is this relevant to the question?"
- Score = (relevant nodes) / (total nodes)
- Penalizes relevant nodes ranked low

### Context Recall

Measures whether the retrieved context contains all information needed to answer the question.

**Structure**:
```json
{
  "context_recall": {
    "score": 0.5,
    "reason": "The score is 0.5 because some expected information is missing...",
    "expected_points": [
      {
        "text": "The judge may excuse a juror who knows the accused..."
      }
    ],
    "found_points": [
      {
        "text": "The court may excuse a juror if they cannot consider the case impartially..."
      }
    ]
  }
}
```

**Fields**:
- `score`: 0.0 to 1.0 (1.0 = all expected points found)
- `reason`: LLM's explanation
- `expected_points`: Information that should be in context
- `found_points`: Information actually found in context

### Answer Relevancy

Measures whether the answer directly addresses the question.

**Structure**:
```json
{
  "answer_relevancy": {
    "score": 1.0,
    "reason": "The score is 1.00 because the answer directly addresses...",
    "question": "Original question here...",
    "answer": "Generated answer here..."
  }
}
```

---

## Real Example: Bob & Ted Juror

**Query ID**: 1

**Question**:
> Bob and Ted are close friends. Ted is on trial for drug offences, and Bob has been selected as a juror in Ted's case. Is the judge required to excuse Bob from serving on the jury?

**Gold Answer**:
> No. While the bench book instructs judges to inform members of the jury panel that they can excuse themselves if they know the accused, this is not mandatory. Instead, the court may excuse a potential juror if they are satisfied that the person will not be able to consider the case impartially.

**Generated Answer**:
> I don't have enough information to answer this question. The provided context does not specifically address whether a juror who is a close friend of the accused must be excused from serving on the jury.

**Metrics**:
- Faithfulness: 1.0 (no contradictions in the answer)
- Context Precision: 0.25 (relevant node ranked 4th out of 5)
- Context Recall: 0.5 (some expected information missing)
- Answer Relevancy: 1.0 (answer addresses the question, even if it's "I don't know")

**Faithfulness Reasoning**:
- Claims extracted: "The provided context does not specifically address..."
- Truths extracted: "A person who has experienced a particular life event..."
- No contradictions → score 1.0

**Context Precision Reasoning**:
- Node 1: "excusing jurors based on personal experiences..." → Verdict: no (not about knowing accused)
- Node 2: "directing the jury regarding official duties..." → Verdict: no (irrelevant)
- Node 3: "directing the jury..." → Verdict: no (irrelevant)
- Node 4: "excuse if cannot consider impartially" → Verdict: yes (relevant!)
- Node 5: "manslaughter verdict reconsideration" → Verdict: no (irrelevant)
- 1 relevant out of 5 = 0.2, but LLM gave 0.25 (some partial credit)

---

## Using details.json for Debugging

### Problem: Low Faithfulness

Check `claims` vs `truths`:
- Are there contradictions?
- Is the answer hallucinating?

### Problem: Low Context Precision

Check `verdicts`:
- Which nodes are irrelevant?
- Is your retrieval system bringing in noise?

### Problem: Low Context Recall

Check `expected_points` vs `found_points`:
- What information is missing?
- Is your retrieval system missing key documents?

### Problem: Low Answer Relevancy

Check the `reason`:
- Is the answer evading the question?
- Is the answer too verbose?

---

## Tooling

**View details for a specific query**:
```bash
# Get query ID 1
jq '.queries[] | select(.query_id == 1)' results/*_details.json

# Get faithfulness reasoning only
jq '.queries[] | select(.query_id == 1) | .reasoning.faithfulness' results/*_details.json
```

**Compare reasoning across queries**:
```bash
# Get all faithfulness scores
jq '.queries[] | {query_id, score: .reasoning.faithfulness.score}' results/*_details.json
```
```

### Definition of Done

- [ ] Real example (Bob & Ted) documented
- [ ] All metric structures explained
- [ ] Field definitions provided
- [ ] Debugging guide included
- [ ] Tooling examples added
- [ ] Documented at `docs/research/details-json-format.md`

---

## PBI-39: Add Time-Series Tracking

**Priority**: P2 (Observability)
**Estimate**: 12 hours
**Category**: Metrics Retrieval → Time-Series

### Problem Statement

Current storage (flat files) doesn't support time-series queries. Can't answer "is my system improving?"

**Evidence**:

```bash
$ ls results/*.json
results/legal_rag_bench_nano_results_20260521_211101.json
results/legal_rag_bench_nano_results_20260521_210254.json
results/legal_rag_bench_nano_results_20260521_203524.json
...
# 42 files

# Question: How has faithfulness changed over time?
# Answer: Parse filenames, load JSONs, extract metrics, plot. 100 lines of Python.
```

**adversarial reviewer says**: "I want to see metric trends. Don't make me write a script every time."

### Solution Options

| Option | Pros | Cons | Effort |
|--------|------|------|--------|
| A. CLI tool | Simple, no deps | Manual step | 4h |
| B. SQLite DB | Queries, time-series | Requires migration | 8h |
| C. Prometheus | Industry standard | Complex setup | 16h |

**Recommended**: Start with A (CLI tool), evolve to B if needed.

### Acceptance Criteria

1. [ ] `eval-trends` CLI command
2. [ ] Plot metric over time (faithfulness vs date)
3. [ ] Compare multiple datasets
4. [ ] Optional CSV export for external plotting
5. [ ] Works with existing flat files

### Implementation

**Create `src/eval_harness/reporting/trends.py`**:

```python
"""Time-series tracking for evaluation metrics."""
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all JSON result files into a DataFrame.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        DataFrame with columns: timestamp, dataset, slice, metric1, metric2, ...
    """
    rows = []

    for json_file in results_dir.glob("*_results_*.json"):
        # Skip details.json
        if "details" in json_file.name:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            # Parse timestamp from filename or JSON
            timestamp_str = data.get("timestamp", json_file.stem.split("_")[-2])
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

            row = {
                "timestamp": timestamp,
                "dataset": data.get("dataset"),
                "slice": data.get("slice"),
                "judge_model": data.get("judge_model"),
                "file": json_file.name
            }

            # Add all metrics
            for metric_name, value in data.get("metrics_avg", {}).items():
                row[metric_name] = value

            rows.append(row)

        except (json.JSONDecodeError, ValueError, KeyError):
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp")

    return df


def plot_metric_trend(
    df: pd.DataFrame,
    metric: str,
    dataset: str | None = None,
    slice: str | None = None,
    output_path: Path | None = None
) -> None:
    """
    Plot metric trend over time.

    Args:
        df: DataFrame from load_all_results
        metric: Metric name to plot (e.g., "faithfulness_score")
        dataset: Filter by dataset (optional)
        slice: Filter by slice (optional)
        output_path: Save plot to this path (optional)
    """
    # Filter
    if dataset:
        df = df[df["dataset"] == dataset]
    if slice:
        df = df[df["slice"] == slice]

    if df.empty:
        print(f"[WARN] No data found for metric={metric}, dataset={dataset}, slice={slice}")
        return

    # Group by timestamp (handle multiple runs per timestamp)
    df_plot = df.groupby("timestamp").agg({
        metric: ["mean", "min", "max", "count"]
    }).reset_index()
    df_plot.columns = ["timestamp", "mean", "min", "max", "count"]

    # Plot
    plt.figure(figsize=(12, 6))

    # Mean line
    plt.plot(df_plot["timestamp"], df_plot["mean"], marker="o", label="Mean")

    # Fill between min and max
    plt.fill_between(
        df_plot["timestamp"],
        df_plot["min"],
        df_plot["max"],
        alpha=0.3,
        label="Min-Max Range"
    )

    # Formatting
    plt.xlabel("Time")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Count annotations
    for i, row in df_plot.iterrows():
        if row["count"] > 1:
            plt.annotate(f"n={row['count']}", (row["timestamp"], row["max"]), fontsize=8)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100)
        print(f"[INFO] Saved plot to {output_path}")
    else:
        plt.show()


def compare_datasets(
    df: pd.DataFrame,
    metric: str,
    datasets: list[str],
    output_path: Path | None = None
) -> None:
    """
    Compare metric across datasets over time.

    Args:
        df: DataFrame from load_all_results
        metric: Metric name to compare
        datasets: List of datasets to compare
        output_path: Save plot to this path
    """
    plt.figure(figsize=(12, 6))

    for dataset in datasets:
        df_dataset = df[df["dataset"] == dataset]
        if df_dataset.empty:
            continue

        df_grouped = df_dataset.groupby("timestamp").agg({metric: "mean"}).reset_index()
        plt.plot(df_grouped["timestamp"], df_grouped[metric], marker="o", label=dataset)

    plt.xlabel("Time")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100)
        print(f"[INFO] Saved plot to {output_path}")
    else:
        plt.show()


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot metric trends over time"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="faithfulness_score",
        help="Metric to plot"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Filter by dataset"
    )
    parser.add_argument(
        "--slice",
        type=str,
        help="Filter by slice"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple datasets"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Save plot to file"
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Export data to CSV for external plotting"
    )

    args = parser.parse_args()

    # Load all results
    df = load_all_results(args.results_dir)

    if df.empty:
        print("[WARN] No result files found")
        return

    # Export CSV if requested
    if args.export_csv:
        df.to_csv(args.export_csv, index=False)
        print(f"[INFO] Exported data to {args.export_csv}")

    # Plot
    if args.compare:
        compare_datasets(df, args.metric, args.compare, args.output)
    else:
        plot_metric_trend(df, args.metric, args.dataset, args.slice, args.output)


if __name__ == "__main__":
    main()
```

**Add to `pyproject.toml`**:

```toml
[project.scripts]
eval-trends = "eval_harness.reporting.trends:main"
```

**Add dependency**:

```toml
[project.optional-dependencies]
visualization = [
    "matplotlib>=3.5.0",
]
```

**Usage**:

```bash
# Plot faithfulness over time
uv run eval-trends --metric faithfulness_score

# Filter by dataset
uv run eval-trends --metric faithfulness_score --dataset legal_rag_bench

# Compare datasets
uv run eval-trends --metric faithfulness_score --compare legal_rag_bench omnidocbench

# Save plot
uv run eval-trends -o trends.png

# Export CSV for Excel
uv run eval-trends --export-csv trends.csv
```

### Definition of Done

- [ ] `eval-trends` CLI command working
- [ ] Plots single metric over time
- [ ] Compares multiple datasets
- [ ] CSV export working
- [ ] Handles missing timestamps gracefully
- [ ] Groups by timestamp for multiple runs per day

---

## PBI-40: Add Visualization

**Priority**: P2 (UX)
**Estimate**: 8 hours
**Category**: Metrics Retrieval → Visualization

### Problem Statement

No built-in visualization. Users must export to CSV and use external tools.

**Evidence**:

Current workflow:
```bash
# Run eval
uv run eval-rag --dataset legal_rag_bench --slice nano

# Get CSV
# Open Excel
# Load CSV
# Select columns
# Insert chart
# Format chart
# Export PNG
# That's 5 minutes every time
```

**adversarial reviewer says**: "I want a one-page PDF report with charts. Not 'here's a CSV, go make your own charts'."

### Acceptance Criteria

1. [ ] `eval-report` CLI command
2. [ ] Generate PDF with charts
3. [ ] Include metric distributions (histograms)
4. [ ] Include per-query breakdown
5. [ ] Include time-series (if historical data exists)
6. [ ] Branding/customization options

### Implementation

**Create `src/eval_harness/reporting/pdf_report.py`**:

```python
"""Generate PDF report with visualizations."""
from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_pdf_report(
    csv_path: Path,
    output_path: Path,
    title: str | None = None
) -> None:
    """
    Generate PDF report with charts and tables.

    Args:
        csv_path: Path to CSV results
        output_path: Path for PDF output
        title: Report title (auto-generated if None)
    """
    df = pd.read_csv(csv_path)

    # Auto-generate title
    if title is None:
        dataset = csv_path.stem.split("_")[0] if "_" in csv_path.stem else csv_path.stem
        title = f"Evaluation Report: {dataset}"

    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        _page_summary(df, title, pdf)

        # Page 2: Metric distributions
        _page_distributions(df, pdf)

        # Page 3: Per-query details
        _page_details(df, pdf)

        # Page 4: Correlations
        _page_correlations(df, pdf)

    print(f"[INFO] Generated PDF report: {output_path}")


def _page_summary(df: pd.DataFrame, title: str, pdf: PdfPages) -> None:
    """Generate summary page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Calculate stats
    total = len(df)
    if "judge_verdict" in df.columns:
        passed = len(df[df["judge_verdict"] == "PASS"])
        pass_rate = (passed / total * 100) if total > 0 else 0
    else:
        passed = 0
        pass_rate = 0

    # Metrics
    metrics = ["faithfulness_score", "context_precision_score",
               "context_recall_score", "answer_relevancy_score"]

    metric_summary = ""
    for m in metrics:
        if m in df.columns:
            valid = df[df[m].notna()]
            if len(valid) > 0:
                metric_summary += f"{m}: {valid[m].mean():.3f} ± {valid[m].std():.3f} (n={len(valid)})\n"

    # Text
    text = f"""
{title}

{'=' * 80}

SUMMARY
--------
Total Queries: {total}
Passed: {passed}
Pass Rate: {pass_rate:.1f}%

METRICS
-------
{metric_summary}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def _page_distributions(df: pd.DataFrame, pdf: PdfPages) -> None:
    """Generate metric distribution page."""
    metrics = ["faithfulness_score", "context_precision_score",
               "context_recall_score", "answer_relevancy_score"]

    available = [m for m in metrics if m in df.columns]

    if not available:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.flatten()

    for i, metric in enumerate(available[:4]):
        ax = axes[i]
        data = df[metric].dropna()

        if len(data) > 0:
            ax.hist(data, bins=20, alpha=0.7, edgecolor="black")
            ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():.3f}")
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle("Metric Distributions", fontsize=14, y=0.995)
    plt.tight_layout()

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def _page_details(df: pd.DataFrame, pdf: PdfPages) -> None:
    """Generate per-query details page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Get columns for table
    display_cols = ["query_id", "judge_verdict"]
    metrics = ["faithfulness_score", "context_precision_score",
               "context_recall_score", "answer_relevancy_score"]

    for m in metrics:
        if m in df.columns:
            display_cols.append(m)

    # Create table
    table_data = df[display_cols].head(20).fillna("-").values.tolist()

    # Add header
    table_data = [display_cols] + table_data

    table = ax.table(cellText=table_data, cellLoc="left", loc="center",
                     colWidths=[0.1] * len(display_cols))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header
    for i in range(len(display_cols)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax.set_title("Per-Query Results (First 20)", fontsize=12, pad=20)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def _page_correlations(df: pd.DataFrame, pdf: PdfPages) -> None:
    """Generate correlation page."""
    metrics = ["faithfulness_score", "context_precision_score",
               "context_recall_score", "answer_relevancy_score"]

    available = [m for m in metrics if m in df.columns]

    if len(available) < 2:
        return

    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Compute correlation matrix
    corr = df[available].corr()

    # Plot heatmap
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)

    # Labels
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels([m.replace("_", "\n") for m in available], rotation=0)
    ax.set_yticklabels([m.replace("_", "\n") for m in available])

    # Colorbar
    plt.colorbar(im, ax=ax)

    # Add correlation values
    for i in range(len(available)):
        for j in range(len(available)):
            text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black")

    ax.set_title("Metric Correlations", fontsize=12)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PDF report with visualizations"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to CSV results file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output PDF path (default: same as CSV with .pdf extension)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Report title"
    )

    args = parser.parse_args()

    # Default output
    if args.output is None:
        args.output = args.csv_path.with_suffix(".pdf")

    generate_pdf_report(args.csv_path, args.output, args.title)


if __name__ == "__main__":
    main()
```

**Add to `pyproject.toml`**:

```toml
[project.scripts]
eval-report = "eval_harness.reporting.pdf_report:main"
```

**Usage**:

```bash
# Generate PDF report
uv run eval-report results/legal_rag_bench_nano_results_20260521_211101.csv

# Custom title
uv run eval-report results/*.csv --title "Legal RAG Bench - May 2026"

# Specify output
uv run eval-report results/*.csv -o report.pdf
```

### Definition of Done

- [ ] PDF generation working
- [ ] Summary page with stats
- [ ] Distribution histograms
- [ ] Per-query table
- [ ] Correlation heatmap
- [ ] CLI command `eval-report`
- [ ] Handles missing metrics gracefully

---

## Summary Table

| PBI | Priority | Estimate | Category | Risk if Deferred |
|-----|----------|----------|----------|------------------|
| Fix html_summary.py | P0 | 2h | Retrieval → HTML | Broken tool, can't view results |
| Add confidence intervals | P1 | 4h | Retrieval → Statistics | Misleading point estimates |
| Add details.json sample | P1 | 1h | Retrieval → Docs | Can't interpret reasoning |
| Add time-series tracking | P2 | 12h | Retrieval → Trends | Can't track improvement |
| Add visualization | P2 | 8h | Retrieval → Viz | Manual chart creation |

**Total P0**: 2 hours
**Total P1**: 5 hours
**Total P2**: 20 hours
**Total all**: 27 hours

---

## Dependencies

```
PBI-36 (Fix html_summary) ─────────────────────┐
PBI-37 (CI) ─────────────────────────────────────┤
PBI-38 (details.json docs) ───────────────────────→ Can implement in parallel
PBI-39 (Time-series) ─────────────────────────────┤
PBI-40 (Visualization) ───────────────────────────┘
```

**PBI-39 and PBI-40 share**: Both need matplotlib and data loading. Implement together for efficiency.

---

## Implementation Sequence

**Week 1 (P0)**:
1. PBI-36: Fix html_summary.py (2h)

**Week 2 (P1)**:
2. PBI-37: Add confidence intervals (4h)
3. PBI-38: Add details.json sample (1h)

**Week 3-4 (P2)**:
4. PBI-39: Add time-series tracking (12h)
5. PBI-40: Add visualization (8h) - can overlap with PBI-39

---

**Document version**: 1.0
**Last updated**: 2026-05-22
**For**: Team planning meeting
