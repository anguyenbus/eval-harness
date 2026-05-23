# A/B Routing: PBIs

**Capability**: Compare two evaluation runs (different models, configs, or code) with statistical rigor.

**Current State**:
- No built-in comparison tool
- No statistical testing (is difference significant?)
- No visual A/B report
- Manual comparison only (diff JSONs)

---

## adversarial Review Questions (Answer Before Review)

### Q1: "Why do we need an A/B tool? Can't we just use diff?"

**Answer**: Diff doesn't answer "is B better than A?"

**Evidence**:

Current workflow:
```bash
# Run variant A
uv run eval-rag --dataset legal_rag_bench --slice nano --output results_a.json

# Run variant B
uv run eval-rag --dataset legal_rag_bench --slice nano --output results_b.json

# Compare with diff
diff results_a.json results_b.json
# Output:
# < "faithfulness_score": 0.75
# ---
# > "faithfulness_score": 0.78

# Question: Is 0.78 vs 0.75 a real improvement or noise?
# Answer: diff doesn't tell you.
```

**adversarial reviewer says**: "You claim your new model is better. faithfulness went from 0.75 to 0.78. Is that +0.03 real or variance? Give me a p-value."

**PBI-47 addresses this**.

---

### Q2: "Why statistical testing? Just look at the difference."

**Answer**: Difference ≠ significance.

**Evidence**:

| Scenario | A → B | Difference | Significant? |
|----------|-------|------------|--------------|
| 1 | 0.75 → 0.78 | +0.03 | No (n=5, stdev=0.10) |
| 2 | 0.75 → 0.78 | +0.03 | Yes (n=100, stdev=0.02) |

**Same difference, different conclusion.**

**Without statistical testing**:
- Scenario 1: You ship B thinking it's better. It's actually noise.
- Scenario 2: You don't ship B thinking it's noise. It's actually better.

**adversarial reviewer says**: "I'm not shipping your model change based on 'looks better'. Give me a p-value or I'm not approving."

**PBI-48 addresses this**.

---

### Q3: "Why HTML report? Can't we just output to console?"

**Answer**: Console is for logs. Reports are for stakeholders.

**Evidence**:

Console output:
```
Variant A: faithfulness=0.75
Variant B: faithfulness=0.78
Difference: +0.03
```

**adversarial reviewer says**: "I need to present this to the product team. They want charts, colors, something they can forward. Console output doesn't cut it."

**PBI-49 addresses this**.

---

## PBI-47: Add Built-in Comparison Tool

**Priority**: P1 (Rigor)
**Estimate**: 4 hours
**Category**: A/B Routing → Comparison

### Problem Statement

No CLI tool to compare two evaluation runs. Users must manually inspect JSON files.

**Evidence**:

Current state:
```bash
$ ls results/*.json
results/variant_a_20260521.json
results/variant_b_20260521.json

# How do I compare?
# Option 1: diff (hard to read)
# Option 2: Open both in editor (manual)
# Option 3: Write Python script (time-consuming)

# There's no: eval-compare variant_a.json variant_b.json
```

**adversarial reviewer says**: "I ran two variants. Give me a one-liner to compare them."

### Acceptance Criteria

1. [ ] `eval-compare` CLI command
2. [ ] Load two JSON result files
3. [ ] Compare all metrics
4. [ ] Show: A value, B value, difference, relative change
5. [ ] Color-coded output (green for improvement, red for regression)
6. [ ] Summary: which variant won?

### Implementation

**Create `src/eval_harness/comparison/compare.py`**:

```python
"""Compare two evaluation runs."""
import json
import sys
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_runs(
    path_a: Path,
    path_b: Path,
    threshold: float = 0.05
) -> dict[str, Any]:
    """
    Compare two evaluation runs.

    Args:
        path_a: Path to variant A results
        path_b: Path to variant B results
        threshold: Relative difference threshold for significance

    Returns:
        Comparison results
    """
    results_a = load_results(path_a)
    results_b = load_results(path_b)

    metrics_a = results_a.get("metrics_avg", {})
    metrics_b = results_b.get("metrics_avg", {})

    # Get all metric names
    all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())

    comparisons = []
    wins_a = 0
    wins_b = 0
    ties = 0

    for metric in sorted(all_metrics):
        val_a = metrics_a.get(metric)
        val_b = metrics_b.get(metric)

        # Handle missing values
        if val_a is None or val_b is None:
            comparisons.append({
                "metric": metric,
                "value_a": val_a,
                "value_b": val_b,
                "difference": None,
                "relative_change": None,
                "winner": None,
                "status": "missing"
            })
            continue

        # Calculate difference
        diff = val_b - val_a
        rel_change = (diff / val_a * 100) if val_a != 0 else 0

        # Determine winner (higher is better for all metrics)
        if abs(diff) < 0.001:
            winner = "tie"
            ties += 1
        elif val_b > val_a:
            winner = "B"
            wins_b += 1
        else:
            winner = "A"
            wins_a += 1

        # Status (improvement, regression, tie)
        if abs(rel_change) < threshold * 100:
            status = "tie"
        elif rel_change > 0:
            status = "improvement"
        else:
            status = "regression"

        comparisons.append({
            "metric": metric,
            "value_a": val_a,
            "value_b": val_b,
            "difference": diff,
            "relative_change": rel_change,
            "winner": winner,
            "status": status
        })

    # Overall winner
    if wins_b > wins_a:
        overall_winner = "B"
    elif wins_a > wins_b:
        overall_winner = "A"
    else:
        overall_winner = "tie"

    return {
        "variant_a": str(path_a),
        "variant_b": str(path_b),
        "metrics": comparisons,
        "summary": {
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "overall_winner": overall_winner
        }
    }


def print_comparison(result: dict[str, Any]) -> None:
    """Print comparison to stdout with colors."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Header
    console.print()
    console.print("="*70)
    console.print(f"[bold]A/B Comparison Report[/bold]")
    console.print("="*70)
    console.print()

    # Summary
    summary = result["summary"]
    winner = summary["overall_winner"]

    if winner == "B":
        console.print(f"[bold green]Variant B wins![/bold green] ({summary['wins_b']} vs {summary['wins_a']})")
    elif winner == "A":
        console.print(f"[bold red]Variant A wins[/bold red] ({summary['wins_a']} vs {summary['wins_b']})")
    else:
        console.print(f"[bold]Tie![/bold] ({summary['ties']} ties)")

    console.print()

    # Metrics table
    table = Table(title="Metric Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Variant A", justify="right")
    table.add_column("Variant B", justify="right")
    table.add_column("Difference", justify="right")
    table.add_column("Change %", justify="right")
    table.add_column("Winner", justify="center")

    for comp in result["metrics"]:
        metric = comp["metric"]

        if comp["status"] == "missing":
            table.add_row(
                metric,
                "-" if comp["value_a"] is None else f"{comp['value_a']:.4f}",
                "-" if comp["value_b"] is None else f"{comp['value_b']:.4f}",
                "N/A",
                "N/A",
                "[dim]missing[/dim]"
            )
            continue

        val_a = f"{comp['value_a']:.4f}"
        val_b = f"{comp['value_b']:.4f}"
        diff = f"{comp['difference']:+.4f}"
        change = f"{comp['relative_change']:+.1f}%"

        # Color coding
        if comp["winner"] == "B":
            winner_str = "[bold green]B[/bold green]"
        elif comp["winner"] == "A":
            winner_str = "[bold red]A[/bold red]"
        else:
            winner_str = "[dim]tie[/dim]"

        if comp["status"] == "improvement":
            diff_str = f"[green]{diff}[/green]"
            change_str = f"[green]{change}[/green]"
        elif comp["status"] == "regression":
            diff_str = f"[red]{diff}[/red]"
            change_str = f"[red]{change}[/red]"
        else:
            diff_str = diff
            change_str = change

        table.add_row(metric, val_a, val_b, diff_str, change_str, winner_str)

    console.print(table)
    console.print()


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two evaluation runs"
    )
    parser.add_argument(
        "variant_a",
        type=Path,
        help="Path to variant A results (JSON)"
    )
    parser.add_argument(
        "variant_b",
        type=Path,
        help="Path to variant B results (JSON)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Relative difference threshold (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Save comparison to JSON file"
    )

    args = parser.parse_args()

    # Compare
    result = compare_runs(args.variant_a, args.variant_b, args.threshold)

    # Print
    print_comparison(result)

    # Save if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved comparison to {args.output}")

    # Exit code based on winner (useful for CI/CD)
    winner = result["summary"]["overall_winner"]
    if winner == "B":
        sys.exit(0)
    elif winner == "A":
        sys.exit(1)  # B didn't win
    else:
        sys.exit(2)  # Tie


if __name__ == "__main__":
    main()
```

**Add to `pyproject.toml`**:

```toml
[project.scripts]
eval-compare = "eval_harness.comparison.compare:main"
```

**Usage**:

```bash
# Compare two variants
uv run eval-compare \
    results/variant_a_20260521.json \
    results/variant_b_20260521.json

# Output:
# ======================================================================
# A/B Comparison Report
# ======================================================================
#
# Variant B wins! (3 vs 1)
#
# ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━┓
# ┃ Metric               ┃ Variant A ┃ Variant B ┃ Difference ┃ Change% ┃Winner┃
# ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━┩
# │ faithfulness_score   │   0.7500  │   0.7800  │  +0.0300 │  +4.0% │  B   │
# │ context_precision... │   0.2500  │   0.2800  │  +0.0300 │ +12.0% │  B   │
# │ context_recall_score │   0.5000  │   0.4800  │  -0.0200 │  -4.0% │  A   │
# │ answer_relevancy...  │   0.7200  │   0.7400  │  +0.0200 │  +2.8% │  B   │
# └──────────────────────┴───────────┴───────────┴──────────┴───────┴──────┘
```

### Definition of Done

- [ ] `eval-compare` CLI working
- [ ] Compares all metrics
- [ ] Shows difference and relative change
- [ ] Color-coded output (via rich)
- [ ] Summary: overall winner
- [ ] Exit codes for CI/CD
- [ ] JSON output option

### What a adversarial Reviewer Will Ask

**Q**: "Why rich for colors? Standard ANSI works."

**A**: rich is cleaner API, handles Windows/macOS/Linux differences. Already in dependencies for Phoenix.

**Q**: "What if metrics have different names?"

**A**: Only compare matching names. Report mismatches separately.

**Q**: "Exit code 1 if B doesn't win? That's harsh."

**A**: Useful for CI/CD gate. Can override with `--no-fail`.

---

## PBI-48: Add Statistical Testing

**Priority**: P1 (Rigor)
**Estimate**: 6 hours
**Category**: A/B Routing → Statistics

### Problem Statement

Current comparison shows raw differences but doesn't answer "is this significant?"

**Evidence**:

| Metric | A | B | Diff | Significant? |
|--------|---|---|-----|--------------|
| faithfulness (n=5) | 0.75 | 0.78 | +0.03 | No (p=0.32) |
| faithfulness (n=100) | 0.75 | 0.78 | +0.03 | Yes (p=0.01) |

**Same difference, different conclusion based on sample size and variance.**

**Without statistical testing**:
- Can't distinguish signal from noise
- Can't make data-driven decisions
- Risk shipping noise or missing improvements

**adversarial reviewer says**: "Your A/B test shows +3% improvement. Is that real or did you get lucky? Give me a p-value."

### Solution

Add statistical tests:
- **Binary metrics** (pass/fail): Fisher exact test
- **Continuous metrics** (scores): Bootstrap or t-test

### Acceptance Criteria

1. [ ] Add `--statistical` flag to `eval-compare`
2. [ ] Load per-query results (CSV) for variance
3. [ ] Fisher exact test for binary metrics
4. [ ] Bootstrap CI for continuous metrics
5. [ ] Report p-values and confidence intervals
6. [ ] Interpret result: "significant improvement" vs "not significant"

### Implementation

**Extend `src/eval_harness/comparison/compare.py`**:

```python
"""Statistical comparison of two evaluation runs."""
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
from scipy import stats
import pandas as pd


def compare_statistical(
    csv_a: Path,
    csv_b: Path,
    confidence: float = 0.95
) -> dict[str, Any]:
    """
    Compare two evaluation runs with statistical testing.

    Args:
        csv_a: Path to variant A CSV results
        csv_b: Path to variant B CSV results
        confidence: Confidence level for CI (default: 0.95)

    Returns:
        Statistical comparison results
    """
    # Load CSVs
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    # Match by query_id (only compare common queries)
    common_ids = set(df_a["query_id"]) & set(df_b["query_id"])

    df_a_matched = df_a[df_a["query_id"].isin(common_ids)]
    df_b_matched = df_b[df_b["query_id"].isin(common_ids)]

    results = {
        "n_common": len(common_ids),
        "n_a_only": len(df_a) - len(common_ids),
        "n_b_only": len(df_b) - len(common_ids),
        "metrics": {}
    }

    # Binary metrics (Fisher exact test)
    if "judge_verdict" in df_a_matched.columns and "judge_verdict" in df_b_matched.columns:
        # Count passes in each
        passes_a = len(df_a_matched[df_a_matched["judge_verdict"] == "PASS"])
        fails_a = len(df_a_matched) - passes_a

        passes_b = len(df_b_matched[df_b_matched["judge_verdict"] == "PASS"])
        fails_b = len(df_b_matched) - passes_b

        # Fisher exact test
        _, p_value = stats.fisher_exact([[passes_a, fails_a], [passes_b, fails_b]])

        # Difference in proportions
        prop_a = passes_a / len(df_a_matched)
        prop_b = passes_b / len(df_b_matched)
        diff = prop_b - prop_a

        results["metrics"]["pass_rate"] = {
            "type": "binary",
            "test": "fisher_exact",
            "prop_a": prop_a,
            "prop_b": prop_b,
            "difference": diff,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "interpretation": _interpret_p_value(p_value, prop_b > prop_a)
        }

    # Continuous metrics (bootstrap CI)
    continuous_metrics = [
        "faithfulness_score",
        "context_precision_score",
        "context_recall_score",
        "answer_relevancy_score"
    ]

    for metric in continuous_metrics:
        if metric not in df_a_matched.columns or metric not in df_b_matched.columns:
            continue

        # Get values (drop NaN)
        values_a = df_a_matched[metric].dropna().tolist()
        values_b = df_b_matched[metric].dropna().tolist()

        if len(values_a) == 0 or len(values_b) == 0:
            continue

        # Paired bootstrap CI
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)

        # Bootstrap CI for difference
        n_bootstrap = 10000
        np.random.seed(42)  # For reproducibility
        boot_diffs = []

        for _ in range(n_bootstrap):
            # Resample with replacement (paired)
            indices = np.random.choice(len(values_a), size=len(values_a), replace=True)
            sample_a = [values_a[i] for i in indices]
            sample_b = [values_b[i] for i in indices]

            boot_diffs.append(np.mean(sample_b) - np.mean(sample_a))

        # CI
        alpha = 1 - confidence
        ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

        # P-value (proportion of bootstrap diffs <= 0)
        p_value_two_sided = 2 * min(
            np.mean([d <= 0 for d in boot_diffs]),
            np.mean([d >= 0 for d in boot_diffs])
        )

        # Also do paired t-test (faster, assumes normality)
        _, p_value_ttest = stats.ttest_rel(values_b, values_a)

        results["metrics"][metric] = {
            "type": "continuous",
            "mean_a": mean_a,
            "mean_b": mean_b,
            "difference": mean_b - mean_a,
            "bootstrap_ci": {
                "lower": ci_lower,
                "upper": ci_upper,
                "confidence": confidence
            },
            "p_value_bootstrap": p_value_two_sided,
            "p_value_ttest": p_value_ttest,
            "is_significant": p_value_two_sided < 0.05,
            "interpretation": _interpret_p_value(p_value_two_sided, mean_b > mean_a)
        }

    return results


def _interpret_p_value(p_value: float, b_better: bool) -> str:
    """Interpret p-value into human-readable string."""
    if p_value >= 0.05:
        return "No significant difference"
    elif p_value < 0.001:
        direction = "B better" if b_better else "A better"
        return f"*** {direction} (p < 0.001)"
    elif p_value < 0.01:
        direction = "B better" if b_better else "A better"
        return f"** {direction} (p < 0.01)"
    else:
        direction = "B better" if b_better else "A better"
        return f"* {direction} (p < 0.05)"


def print_statistical_results(result: dict[str, Any]) -> None:
    """Print statistical comparison."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print()
    console.print("="*70)
    console.print("[bold]Statistical Comparison Report[/bold]")
    console.print("="*70)
    console.print()

    # Summary
    console.print(f"Common queries: {result['n_common']}")
    console.print(f"A only: {result['n_a_only']}")
    console.print(f"B only: {result['n_b_only']}")
    console.print()

    # Metrics table
    table = Table(title="Statistical Tests")
    table.add_column("Metric", style="cyan")
    table.add_column("A Mean", justify="right")
    table.add_column("B Mean", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("P-value", justify="right")
    table.add_column("Interpretation")

    for metric, stats in result["metrics"].items():
        if stats["type"] == "binary":
            table.add_row(
                metric,
                f"{stats['prop_a']:.1%}",
                f"{stats['prop_b']:.1%}",
                f"{stats['difference']:+.1%}",
                "N/A",
                f"{stats['p_value']:.4f}",
                stats['interpretation']
            )
        else:
            ci = stats["bootstrap_ci"]
            ci_str = f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"

            # Color based on significance
            p_val = stats["p_value_bootstrap"]
            if p_val < 0.001:
                p_str = f"[green]{p_val:.4f}***[/green]"
            elif p_val < 0.01:
                p_str = f"[green]{p_val:.4f}**[/green]"
            elif p_val < 0.05:
                p_str = f"[green]{p_val:.4f}*[/green]"
            else:
                p_str = f"{p_val:.4f}"

            table.add_row(
                metric,
                f"{stats['mean_a']:.4f}",
                f"{stats['mean_b']:.4f}",
                f"{stats['difference']:+.4f}",
                ci_str,
                p_str,
                stats['interpretation']
            )

    console.print(table)

    # Legend
    console.print()
    console.print("[dim]* p < 0.05, ** p < 0.01, *** p < 0.001[/dim]")
    console.print()


def main_statistical() -> None:
    """CLI entry point for statistical comparison."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two evaluation runs with statistical testing"
    )
    parser.add_argument(
        "csv_a",
        type=Path,
        help="Path to variant A CSV results"
    )
    parser.add_argument(
        "csv_b",
        type=Path,
        help="Path to variant B CSV results"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for CI (default: 0.95)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Save comparison to JSON file"
    )

    args = parser.parse_args()

    # Compare
    result = compare_statistical(args.csv_a, args.csv_b, args.confidence)

    # Print
    print_statistical_results(result)

    # Save if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved comparison to {args.output}")

    # Exit code: 0 if B significantly better, 1 otherwise
    # Check if any metric shows significant B improvement
    b_significantly_better = any(
        m.get("is_significant") and m["mean_b"] > m["mean_a"]
        for m in result["metrics"].values()
        if m.get("type") == "continuous"
    )

    sys.exit(0 if b_significantly_better else 1)


if __name__ == "__main__":
    main_statistical()
```

**Add to `pyproject.toml`**:

```toml
[project.scripts]
eval-compare-stats = "eval_harness.comparison.compare:main_statistical"
```

**Add dependency**:

```toml
[project.dependencies]
# ... existing
"scipy>=1.10.0",  # For statistical tests
```

**Usage**:

```bash
# Statistical comparison
uv run eval-compare-stats \
    results/variant_a.csv \
    results/variant_b.csv

# Output:
# ======================================================================
# Statistical Comparison Report
# ======================================================================
#
# Common queries: 100
#
# ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
# ┃ Metric               ┃ A Mean ┃ B Mean ┃  Diff  ┃  95% CI   ┃P-value ┃ Interpretation    ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
# │ faithfulness_score   │ 0.7500 │ 0.7800 │+0.0300│[0.01,0.05]│0.0023**│** B better (p<0.01)│
# │ context_precision... │ 0.2500 │ 0.2800 │+0.0300│[-0.01,0.07]│0.1234 │No significant diff.│
# │ pass_rate            │  75.0% │  78.0% │ +3.0% │    N/A   │0.3210 │No significant diff.│
# └──────────────────────┴────────┴────────┴───────┴──────────┴────────┴───────────────────┘
#
# * p < 0.05, ** p < 0.01, *** p < 0.001
```

### Definition of Done

- [ ] Fisher exact test for binary metrics
- [ ] Bootstrap CI for continuous metrics
- [ ] P-values reported
- [ ] Human-readable interpretation
- [ ] CLI command `eval-compare-stats` working
- [ ] Handles NaN values
- [ ] Paired comparison (same query_id)

### What a adversarial Reviewer Will Ask

**Q**: "Why bootstrap instead of t-test?"

**A**: Bootstrap doesn't assume normality. Metric scores are bounded (0-1), often not normal. Bootstrap is safer.

**Q**: "Why 10,000 bootstrap samples?"

**A**: Standard for stable CI. Takes ~1 second. Negligible compared to eval time.

**Q**: "Why paired comparison?"

**A**: Same queries in both variants. Paired is more powerful than unpaired.

**Q**: "What if n < 30? Bootstrap still works?"

**A**: Yes, bootstrap works for small n. But CI will be wide (appropriately).

---

## PBI-49: Add Side-by-Side HTML Report

**Priority**: P2 (UX)
**Estimate**: 4 hours
**Category**: A/B Routing → Reporting

### Problem Statement

Console output doesn't work for stakeholders. Need shareable report.

**Evidence**:

**For engineers**:
```
faithfulness: A=0.75, B=0.78, diff=+0.03, p=0.002**
```
This is fine.

**For product managers**:
"Can you send me a report I can forward to the product team?"

**adversarial reviewer says**: "I need a report that looks professional. Something with charts, colors, executive summary. Console output doesn't work for non-engineers."

### Acceptance Criteria

1. [ ] `eval-compare-html` CLI command
2. [ ] Generate single HTML file
3. [ ] Executive summary (which variant won?)
4. [ ] Metric comparison table (color-coded)
5. [ ] Charts (bar chart for metrics)
6. [ ] Statistical significance indicators
7. [ ] Embeddable (email, share)

### Implementation

**Create `src/eval_harness/comparison/html_report.py`**:

```python
"""Generate HTML report for A/B comparison."""
import json
from pathlib import Path
from typing import Any
from datetime import datetime


def generate_html_report(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    output_path: Path
) -> None:
    """
    Generate HTML A/B comparison report.

    Args:
        result_a: Variant A results (from JSON)
        result_b: Variant B results (from JSON)
        output_path: Where to write HTML
    """
    metrics_a = result_a.get("metrics_avg", {})
    metrics_b = result_b.get("metrics_avg", {})

    # Calculate comparisons
    all_metrics = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

    rows = []
    wins_a = 0
    wins_b = 0

    for metric in all_metrics:
        val_a = metrics_a.get(metric)
        val_b = metrics_b.get(metric)

        if val_a is None or val_b is None:
            rows.append(_render_row_missing(metric, val_a, val_b))
            continue

        diff = val_b - val_a
        rel_change = (diff / val_a * 100) if val_a != 0 else 0

        if abs(diff) < 0.001:
            winner = "tie"
        elif val_b > val_a:
            winner = "b"
            wins_b += 1
        else:
            winner = "a"
            wins_a += 1

        rows.append(_render_row_metric(metric, val_a, val_b, diff, rel_change, winner))

    # Overall winner
    if wins_b > wins_a:
        overall_winner = "Variant B"
        overall_class = "winner-b"
    elif wins_a > wins_b:
        overall_winner = "Variant A"
        overall_class = "winner-a"
    else:
        overall_winner = "Tie"
        overall_class = "tie"

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A/B Evaluation Comparison</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}

        /* Header */
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 10px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
        .header .meta {{ opacity: 0.9; }}
        .header .timestamp {{ font-size: 14px; opacity: 0.8; }}

        /* Winner banner */
        .winner-banner {{ padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 30px; }}
        .winner-a {{ background: #fee2e2; color: #991b1b; border: 2px solid #ef4444; }}
        .winner-b {{ background: #dcfce7; color: #166534; border: 2px solid #22c55e; }}
        .tie {{ background: #f3f4f6; color: #4b5563; border: 2px solid #9ca3af; }}

        /* Summary cards */
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h3 {{ font-size: 14px; color: #6b7280; margin-bottom: 10px; }}
        .card .value {{ font-size: 32px; font-weight: bold; color: #1f2937; }}

        /* Table */
        .table-container {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #f9fafb; padding: 15px; text-align: left; font-weight: 600; color: #4b5563; }}
        td {{ padding: 15px; border-top: 1px solid #e5e7eb; }}
        tr:hover {{ background: #f9fafb; }}

        /* Metric colors */
        .metric-name {{ font-family: monospace; color: #4b5563; }}
        .value-a {{ color: #ef4444; }}
        .value-b {{ color: #22c55e; }}
        .diff-pos {{ color: #22c55e; font-weight: 600; }}
        .diff-neg {{ color: #ef4444; font-weight: 600; }}
        .diff-neutral {{ color: #6b7280; }}
        .winner-a {{ color: #ef4444; font-weight: 600; }}
        .winner-b {{ color: #22c55e; font-weight: 600; }}
        .winner-tie {{ color: #9ca3af; }}

        /* Footer */
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; }}

        /* Charts */
        .chart {{ margin: 30px 0; }}
        .bar-container {{ display: flex; align-items: center; margin: 10px 0; }}
        .bar-label {{ width: 200px; font-size: 12px; }}
        .bar-area {{ flex: 1; }}
        .bar {{ height: 30px; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; }}
        .bar-a {{ background: #ef4444; }}
        .bar-b {{ background: #22c55e; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>A/B Evaluation Comparison Report</h1>
            <div class="meta">Variant A vs Variant B</div>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="winner-banner {overall_class}">
            {overall_winner} Wins!
        </div>

        <div class="summary">
            <div class="card">
                <h3>Variant A Wins</h3>
                <div class="value">{wins_a}</div>
            </div>
            <div class="card">
                <h3>Variant B Wins</h3>
                <div class="value">{wins_b}</div>
            </div>
            <div class="card">
                <h3>Total Metrics</h3>
                <div class="value">{len(all_metrics)}</div>
            </div>
        </div>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Variant A</th>
                        <th>Variant B</th>
                        <th>Difference</th>
                        <th>Change</th>
                        <th>Winner</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>

        <div class="chart">
            <h2>Visual Comparison</h2>
            {_render_charts(metrics_a, metrics_b)}
        </div>

        <div class="footer">
            Generated by eval-harness. Questions? Contact the evaluation team.
        </div>
    </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def _render_row_metric(
    metric: str,
    val_a: float,
    val_b: float,
    diff: float,
    rel_change: float,
    winner: str
) -> str:
    """Render a metric comparison row."""
    if diff > 0:
        diff_class = "diff-pos"
        diff_sign = "+"
    elif diff < 0:
        diff_class = "diff-neg"
        diff_sign = ""
    else:
        diff_class = "diff-neutral"
        diff_sign = ""

    winner_class = f"winner-{winner}"
    winner_label = winner.upper()

    return f"""
    <tr>
        <td class="metric-name">{metric}</td>
        <td class="value-a">{val_a:.4f}</td>
        <td class="value-b">{val_b:.4f}</td>
        <td class="{diff_class}">{diff_sign}{diff:.4f}</td>
        <td class="{diff_class}">{diff_sign}{rel_change:.1f}%</td>
        <td class="{winner_class}">{winner_label}</td>
    </tr>"""


def _render_row_missing(metric: str, val_a: float | None, val_b: float | None) -> str:
    """Render a row with missing values."""
    val_a_str = "-" if val_a is None else f"{val_a:.4f}"
    val_b_str = "-" if val_b is None else f"{val_b:.4f}"

    return f"""
    <tr>
        <td class="metric-name">{metric}</td>
        <td class="value-a">{val_a_str}</td>
        <td class="value-b">{val_b_str}</td>
        <td colspan="3" style="color: #9ca3af;">Missing data</td>
    </tr>"""


def _render_charts(metrics_a: dict, metrics_b: dict) -> str:
    """Render bar charts for metrics."""
    charts = []

    for metric in sorted(metrics_a.keys()):
        if metric not in metrics_b:
            continue

        val_a = metrics_a[metric]
        val_b = metrics_b[metric]

        if val_a is None or val_b is None:
            continue

        # Normalize to 0-100%
        max_val = max(val_a, val_b, 0.01)  # Avoid div by zero

        width_a = (val_a / max_val) * 100
        width_b = (val_b / max_val) * 100

        charts.append(f"""
        <div class="bar-container">
            <div class="bar-label">{metric}</div>
            <div class="bar-area">
                <div class="bar bar-a" style="width: {width_a}%; margin-bottom: 5px;">A: {val_a:.3f}</div>
                <div class="bar bar-b" style="width: {width_b}%;">B: {val_b:.3f}</div>
            </div>
        </div>
        """)

    return "".join(charts)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HTML A/B comparison report"
    )
    parser.add_argument(
        "json_a",
        type=Path,
        help="Path to variant A JSON results"
    )
    parser.add_argument(
        "json_b",
        type=Path,
        help="Path to variant B JSON results"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output HTML path (default: comparison.html)"
    )

    args = parser.parse_args()

    # Load results
    with open(args.json_a) as f:
        result_a = json.load(f)
    with open(args.json_b) as f:
        result_b = json.load(f)

    # Default output
    if args.output is None:
        args.output = Path("comparison.html")

    # Generate report
    generate_html_report(result_a, result_b, args.output)
    print(f"[INFO] Generated HTML report: {args.output}")


if __name__ == "__main__":
    main()
```

**Add to `pyproject.toml`**:

```toml
[project.scripts]
eval-compare-html = "eval_harness.comparison.html_report:main"
```

**Usage**:

```bash
# Generate HTML report
uv run eval-compare-html \
    results/variant_a.json \
    results/variant_b.json \
    -o comparison.html

# Open in browser
open comparison.html
```

### Definition of Done

- [ ] HTML generation working
- [ ] Executive summary (which variant won?)
- [ ] Metric table with color coding
- [ ] Bar charts for visual comparison
- [ ] Professional styling
- [ ] Single self-contained file (email-able)

### What a adversarial Reviewer Will Ask

**Q**: "Why not use JavaScript charting library?"

**A**: Single file requirement. JS libraries require CDN or inline code. CSS-only charts are simpler and work everywhere.

**Q**: "Can I customize colors?"

**A**: Not in PBI-49. Future: `--template` flag for custom Jinja2 template.

**Q**: "What if I want to include statistical test results?"

**A**: PBI-49 is basic comparison. Combine with PBI-48 (statistical testing) for full report. Future: unified HTML with both.

---

## Summary Table

| PBI | Priority | Estimate | Category | Risk if Deferred |
|-----|----------|----------|----------|------------------|
| Built-in comparison tool | P1 | 4h | A/B → CLI | Manual comparison only |
| Statistical testing | P1 | 6h | A/B → Statistics | Can't distinguish signal from noise |
| Side-by-side HTML report | P2 | 4h | A/B → Reporting | Console output not shareable |

**Total P1**: 10 hours
**Total P2**: 4 hours
**Total all**: 14 hours

---

## Dependencies

```
PBI-47 (Comparison CLI) ──┐
                          ├──→ Independent, can parallelize
PBI-48 (Statistical) ─────┤
                          │
PBI-49 (HTML report) ─────┘→ Can build on PBI-47
```

**Sequence**:
1. PBI-47: Comparison CLI (4h)
2. PBI-48: Statistical testing (6h) - can parallelize with PBI-47
3. PBI-49: HTML report (4h) - depends on PBI-47

---

## Implementation Sequence

**Week 2 (P1)**:
1. PBI-47: Build comparison CLI (0.5 day)
2. PBI-48: Add statistical testing (1 day) - parallelize

**Week 3-4 (P2)**:
3. PBI-49: HTML report (0.5 day)

---

**Document version**: 1.0
**Last updated**: 2026-05-22
**For**: Team planning meeting
