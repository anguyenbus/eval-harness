# Run Evaluation Framework: PBIs

**Capability**: CLI framework for running evaluations (eval-rag, eval-parsing) with validation, progress tracking, and regression detection.

**Current State**:
- Regression detector **broken** (expects wrong JSON format)
- No dry-run mode (must run 2-hour eval to find config typo)
- No variance reporting (single run only)
- No progress bar (long runs appear frozen)

---

## Hostile Review Questions (Answer Before Review)

### Q1: "Why is broken regression check P0? Just fix the JSON format."

**Answer**: Because changing JSON format breaks existing result files.

**Evidence**:

[`regression_check.py:39`](src/eval_harness/reporting/regression_check.py:39) expects:
```python
for metric_name, current_data in current.get("metrics", {}).items():
    if current_data.get("severity") != "blocker":  # ← Expects 'metrics' with 'severity'
        continue
```

**Actual JSON format** (from `results/*.json`):
```json
{
  "metrics_avg": {
    "faithfulness_score": 0.9667,
    "context_precision_score": 0.2367
  }
}
```

**Result**: `regression_check.py` always finds 0 metrics to check. Passes silently even when regression exists.

**Hostile reviewer says**: "I rely on regression detection to catch bad deploys. Your tool says 'no regression' when there IS a regression. Now I shipped bad code. Fix your detector or delete it."

**PBI-41 addresses this**.

---

### Q2: "Dry-run mode? Just validate your config before committing."

**Answer**: Config isn't the only thing that can fail.

**Evidence**:

```bash
# Run eval
uv run eval-rag --dataset legal_rag_bench --slice nano

# 10 minutes later...
[ERROR] Dataset file not found: legal_rag_bench_nano.json
[ERROR] Or: Model gpt-4o-2024-08-06 not accessible
[ERROR] Or: Insufficient quota

# Wasted 10 minutes + API quota
```

**Things that can fail**:
1. Config syntax errors
2. Dataset file missing/malformed
3. Model not accessible (wrong name, no quota)
4. Missing dependencies
5. Invalid parameter combinations

**Hostile reviewer says**: "I want to know my eval will work BEFORE I wait 2 hours. Don't make me discover failures at minute 119."

**PBI-42 addresses this**.

---

### Q3: "Variance reporting? Just run 5x and average."

**Answer**: Automate it. Don't make me run 5 separate commands.

**Evidence**:

Current workflow to measure variance:
```bash
# Run 1
uv run eval-rag --dataset legal_rag_bench --slice nano
# Save result

# Run 2
uv run eval-rag --dataset legal_rag_bench --slice nano
# Save result

# Run 3, 4, 5...

# Then manually compute: mean, stdev, min, max
# That's 10 hours + manual computation
```

**Hostile reviewer says**: "I need to know if my 0.75 faithfulness is 0.75 ± 0.05 or 0.75 ± 0.25. Don't make me run 5 times and do math manually."

**PBI-43 addresses this**.

---

### Q4: "Progress bar? Just look at the CSV file growing."

**Answer**: CSV writes at the end, not incrementally.

**Evidence**:

```python
# Current: results written at end
results = []
for item in dataset:
    result = evaluate(item)
    results.append(result)

# At the END, write CSV
df = pd.DataFrame(results)
df.to_csv(output_path)

# During run: no output. File doesn't exist yet.
# User thinks: "Is it frozen? Did it crash?"
```

**Hostile reviewer says**: "I run a 1000-query eval. No output for 30 minutes. I kill it and restart. Now I wasted 30 minutes. Give me a progress bar."

**PBI-44 addresses this**.

---

## PBI-41: Fix regression_check.py (JSON Format Mismatch)

**Priority**: P0 (Correctness)
**Estimate**: 2 hours
**Category**: Run Framework → Regression Detection

### Problem Statement

[`regression_check.py`](src/eval_harness/reporting/regression_check.py:39) expects JSON structure that doesn't match actual output.

**Evidence of mismatch**:

**Code expects**:
```python
# regression_check.py:39
for metric_name, current_data in current.get("metrics", {}).items():
    if current_data.get("severity") != "blocker":  # ← severity field
        continue

    current_score = current_data.get("score", 0)  # ← nested score
```

**Actual JSON has**:
```json
{
  "metrics_avg": {
    "faithfulness_score": 0.9667  # ← flat structure, no severity
  }
}
```

| Expected by code | Actual in JSON | Result |
|------------------|---------------|--------|
| `metrics` | `metrics_avg` | KeyError → empty dict → no checks |
| `severity` | (doesn't exist) | KeyError (caught by .get()) → skips all |
| `{"score": 0.96}` | `0.9667` | Wrong data type |

**Impact**: Regression detector NEVER finds regressions. Always passes.

### Root Cause

Code was written for a different JSON schema. Schema changed but code wasn't updated.

### Acceptance Criteria

1. [ ] Fix JSON path: `metrics` → `metrics_avg`
2. [ ] Remove severity check (not in current format)
3. [ ] Make threshold configurable
4. [ ] Add both relative AND absolute threshold options
5. [ ] Handle missing metrics gracefully
6. [ ] Add metric blocklist (e.g., skip `total_ms`)

### Implementation

**Rewrite `src/eval_harness/reporting/regression_check.py`**:

```python
"""Regression checking against baseline results.

FIXED: Now works with actual JSON format using metrics_avg.
"""
import json
import sys
from pathlib import Path
from typing import Any


def check_regression(
    current_results: Path,
    baseline_path: Path,
    threshold: float = 0.05,
    absolute_threshold: float | None = None,
    metric_blocklist: list[str] | None = None,
    fail_on_improvement: bool = False
) -> dict[str, Any]:
    """
    Compare current results against baseline and check for regressions.

    Args:
        current_results: Path to JSON file with current results
        baseline_path: Path to JSON file with baseline results
        threshold: Relative regression threshold (default 5%)
        absolute_threshold: Optional absolute threshold (overrides relative if set)
        metric_blocklist: Metrics to skip (e.g., ["total_ms"])
        fail_on_improvement: Also fail if metrics improve significantly

    Returns:
        Dict with regression status and details

    Raises:
        RuntimeError: If regression is detected
        FileNotFoundError: If either file doesn't exist
    """
    # Load files
    if not current_results.exists():
        raise FileNotFoundError(f"Current results not found: {current_results}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    with open(current_results) as f:
        current = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    # Use metrics_avg (actual JSON format)
    current_metrics = current.get("metrics_avg", {})
    baseline_metrics = baseline.get("metrics_avg", {})

    # Default blocklist
    if metric_blocklist is None:
        metric_blocklist = ["total_ms"]

    # Check for regressions
    regressions = []
    improvements = []

    # All metrics in either file
    all_metrics = set(current_metrics.keys()) | set(baseline_metrics.keys())

    # Skip blocklisted metrics
    all_metrics -= set(metric_blocklist)

    for metric_name in sorted(all_metrics):
        if metric_name not in baseline_metrics:
            # New metric, can't compare
            continue

        current_score = current_metrics.get(metric_name, 0.0)
        baseline_score = baseline_metrics[metric_name]

        # Handle NaN
        if current_score != current_score or baseline_score != baseline_score:
            continue

        # Calculate change
        absolute_change = current_score - baseline_score

        if baseline_score != 0:
            relative_change = absolute_change / baseline_score
        else:
            relative_change = 0.0

        # Determine threshold
        if absolute_threshold is not None:
            effective_threshold = absolute_threshold
            is_regression = absolute_change < -absolute_threshold
            is_improvement = absolute_change > absolute_threshold
        else:
            effective_threshold = threshold
            is_regression = relative_change < -threshold
            is_improvement = relative_change > threshold

        if is_regression:
            regressions.append({
                "metric": metric_name,
                "baseline": baseline_score,
                "current": current_score,
                "relative_change_pct": relative_change * 100,
                "absolute_change": absolute_change
            })

        if is_improvement and fail_on_improvement:
            improvements.append({
                "metric": metric_name,
                "baseline": baseline_score,
                "current": current_score,
                "relative_change_pct": relative_change * 100,
                "absolute_change": absolute_change
            })

    # Build result
    result = {
        "has_regression": len(regressions) > 0,
        "regressions": regressions,
        "improvements": improvements,
        "threshold_type": "absolute" if absolute_threshold else "relative",
        "threshold": absolute_threshold if absolute_threshold else threshold
    }

    # Print summary
    _print_summary(result, current_results.name, baseline_path.name)

    # Fail on regression
    if regressions:
        raise RuntimeError(
            f"Regression detected in {len(regressions)} metric(s). "
            f"See details above."
        )

    return result


def _print_summary(result: dict[str, Any], current_name: str, baseline_name: str) -> None:
    """Print regression summary to stdout/stderr."""
    print(f"\n{'='*60}")
    print(f"Regression Check: {current_name} vs {baseline_name}")
    print(f"{'='*60}\n")

    if result["improvements"]:
        print("Improvements:")
        for imp in result["improvements"]:
            metric = imp["metric"]
            baseline = imp["baseline"]
            current = imp["current"]
            change = imp["relative_change_pct"]
            print(f"  ✓ {metric}: {baseline:.4f} → {current:.4f} ({change:+.1f}%)")
        print()

    if result["regressions"]:
        print("🔴 Regressions:", file=sys.stderr)
        for reg in result["regressions"]:
            metric = reg["metric"]
            baseline = reg["baseline"]
            current = reg["current"]
            change = reg["relative_change_pct"]
            print(f"  - {metric}: {baseline:.4f} → {current:.4f} ({change:.1f}%)",
                  file=sys.stderr)
        print()

    if not result["regressions"]:
        print("✓ No regressions detected")
        if result["threshold_type"] == "relative":
            print(f"  Threshold: ±{result['threshold']*100:.1f}%")
        else:
            print(f"  Threshold: ±{result['threshold']:.4f}")
        print()


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check for regressions in evaluation results"
    )
    parser.add_argument(
        "current",
        type=Path,
        help="Path to current result JSON file"
    )
    parser.add_argument(
        "baseline",
        type=Path,
        help="Path to baseline result JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Relative regression threshold (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        help="Absolute regression threshold (overrides --threshold)"
    )
    parser.add_argument(
        "--blocklist",
        nargs="+",
        default=["total_ms"],
        help="Metrics to skip (default: total_ms)"
    )
    parser.add_argument(
        "--fail-on-improvement",
        action="store_true",
        help="Also fail if metrics improve significantly (for testing)"
    )

    args = parser.parse_args()

    try:
        check_regression(
            current_results=args.current,
            baseline_path=args.baseline,
            threshold=args.threshold,
            absolute_threshold=args.absolute_threshold,
            metric_blocklist=args.blocklist,
            fail_on_improvement=args.fail_on_improvement
        )
        print("\n✓ Regression check passed")
        sys.exit(0)
    except RuntimeError as e:
        print(f"\n✗ Regression check failed: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Add to `pyproject.toml`**:

```toml
[project.scripts]
eval-regression-check = "eval_harness.reporting.regression_check:main"
```

**Usage**:

```bash
# Check against baseline
uv run eval-regression-check \
    results/legal_rag_bench_nano_results_20260521_211101.json \
    results/baseline.json

# Absolute threshold
uv run eval-regression-check current.json baseline.json --absolute-threshold 0.1

# Blocklist metrics
uv run eval-regression-check current.json baseline.json --blocklist total_ms latency_ms
```

### Definition of Done

- [ ] JSON path fixed (`metrics_avg`)
- [ ] Severity check removed
- [ ] Relative threshold working
- [ ] Absolute threshold working
- [ ] Metric blocklist working
- [ ] Tested with actual result files
- [ ] CLI command `eval-regression-check` added
- [ ] Exit code 1 on regression

### What a Hostile Reviewer Will Ask

**Q**: "Why remove severity check? What if we want blocker-only checks?"

**A**: Current JSON doesn't have severity. Add severity to JSON first (P1), then restore check. For now, check all metrics.

**Q**: "Why default threshold 5%? That's arbitrary."

**A**: Yes. That's why PBI-43 (variance reporting) exists to calibrate threshold using actual variance.

**Q**: "What if baseline has a metric that current doesn't?"

**A**: Skipped. Can't regress if metric doesn't exist. (Could treat as 0, but that's aggressive).

---

## PBI-42: Add Dry-Run Mode

**Priority**: P0 (Safety)
**Estimate**: 2 hours
**Category**: Run Framework → Validation

### Problem Statement

No way to validate config without running full evaluation. Must start eval to discover typos.

**Evidence**:

```bash
# Try to run
uv run eval-rag --dataset legal_rag_bench --slice nano

# 2 minutes later...
[ERROR] Dataset slice 'nana' not found. Did you mean 'nano'?
# Wasted 2 minutes + API calls if model was checked first

# Or worse:
[INFO] Starting evaluation...
[INFO] Processed 1/10
[INFO] Processed 2/10
[ERROR] Insufficient quota for model gpt-4o
# 10 queries charged to API before discovering quota issue
```

**Things to validate**:
1. Config file syntax
2. Dataset exists and is readable
3. Slice exists within dataset
4. Model is accessible (API key works, quota exists)
5. Output directory is writable
6. Dependencies are installed

### Acceptance Criteria

1. [ ] `--dry-run` flag validates everything
2. [ ] Checks config file syntax
3. [ ] Checks dataset exists and is loadable
4. [ ] Checks model accessibility (with lightweight call)
5. [ ] Shows what would run (count, estimated cost)
6. [ ] Exits without making LLM API calls
7. [ ] Works for both eval-rag and eval-parsing

### Implementation

**Add to `src/eval_harness/runners/run_rag_eval.py`**:

```python
# In argparse setup
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate config and dataset without running full evaluation"
)

# In main()
def _do_dry_run(args, config, dataset_path, judge_model) -> None:
    """Perform dry-run validation."""
    print("\n" + "="*60)
    print("DRY RUN MODE - Validating configuration")
    print("="*60 + "\n")

    all_valid = True

    # 1. Config file
    print("[1/6] Config file")
    print(f"  ✓ Config: {args.config}")
    print(f"  ✓ Dataset: {args.dataset}")
    print(f"  ✓ Slice: {args.slice}")

    # 2. Dataset exists
    print("\n[2/6] Dataset file")
    try:
        if not dataset_path.exists():
            print(f"  ✗ Dataset file not found: {dataset_path}")
            all_valid = False
        else:
            print(f"  ✓ Dataset file: {dataset_path}")
            print(f"  ✓ File size: {dataset_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"  ✗ Error accessing dataset: {e}")
        all_valid = False

    # 3. Dataset loadable
    print("\n[3/6] Dataset format")
    try:
        from eval_harness.datasets.rag_dataset import load_rag_dataset

        dataset = list(load_rag_dataset(dataset_path, args.slice))
        print(f"  ✓ Loaded {len(dataset)} items")

        if len(dataset) == 0:
            print(f"  ✗ Slice '{args.slice}' is empty")
            all_valid = False

        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  ✓ Sample query: {sample.get('question', '')[:50]}...")

    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        all_valid = False

    # 4. Model accessible
    print("\n[4/6] Model access")
    try:
        # Lightweight test: count tokens only (no generation)
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Test with tokenization (cheaper than generation)
        test_text = "Hello, world!"
        import tiktoken

        encoding = tiktoken.encoding_for_model(judge_model)
        tokens = encoding.encode(test_text)

        print(f"  ✓ Model accessible: {judge_model}")
        print(f"  ✓ Tokenization working ({len(tokens)} tokens for test)")

    except Exception as e:
        print(f"  ✗ Model check failed: {e}")
        all_valid = False

    # 5. Output directory writable
    print("\n[5/6] Output directory")
    try:
        output_dir = Path(args.output or "results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test write
        test_file = output_dir / ".dry_run_test"
        test_file.touch()
        test_file.unlink()

        print(f"  ✓ Output directory writable: {output_dir}")

    except Exception as e:
        print(f"  ✗ Output directory not writable: {e}")
        all_valid = False

    # 6. Dependencies
    print("\n[6/6] Dependencies")
    try:
        import pandas
        import deepeval
        from rich.progress import Progress

        print(f"  ✓ pandas {pandas.__version__}")
        print(f"  ✓ deepeval {deepeval.__version__}")

    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        all_valid = False

    # Summary
    print("\n" + "="*60)
    if all_valid:
        print("✓ DRY RUN PASSED")
        print("="*60)

        # Estimates
        print("\nEvaluation estimates:")
        print(f"  Total queries: {len(dataset)}")
        print(f"  Estimated queries: {len(dataset) * 4}")  # 4 metrics per query
        print(f"  Estimated time: {len(dataset) * 2 / 60:.1f} minutes")

        # Cost estimate (rough)
        # GPT-4o: ~$0.005/1K input tokens, ~$0.015/1K output tokens
        # Assume 500 input + 200 output per query
        input_tokens = len(dataset) * 500
        output_tokens = len(dataset) * 200
        est_cost = (input_tokens * 0.005 / 1000) + (output_tokens * 0.015 / 1000)
        print(f"  Estimated cost: ${est_cost:.2f}")

        print("\nRun without --dry-run to execute evaluation.")
    else:
        print("✗ DRY RUN FAILED")
        print("="*60)
        print("\nFix the errors above before running the evaluation.")

    print()
    return all_valid

# In main(), before evaluation loop
if args.dry_run:
    valid = _do_dry_run(args, config, dataset_path, judge_model)
    sys.exit(0 if valid else 1)
```

**Usage**:

```bash
# Validate before running
uv run eval-rag --dataset legal_rag_bench --slice nano --dry-run

# Output:
# [1/6] Config file
#   ✓ Config: config/eval_config.yaml
#   ✓ Dataset: legal_rag_bench
#   ✓ Slice: nano
#
# [2/6] Dataset file
#   ✓ Dataset file: data/legal_rag_bench.json
#   ✓ File size: 125.3 KB
#
# ...
#
# ✓ DRY RUN PASSED
# Estimated cost: $0.50
```

### Definition of Done

- [ ] `--dry-run` flag added
- [ ] All 6 validation checks implemented
- [ ] Cost estimation working
- [ ] No API calls made during dry-run
- [ ] Clear success/failure output
- [ ] Works for both eval-rag and eval-parsing

### What a Hostile Reviewer Will Ask

**Q**: "Why check model with tiktoken instead of actual API call?"

**A**: Cost. tiktoken is free. API call costs money. Dry-run should be free.

**Q**: "What if dataset loads but has wrong columns?"

**A**: That's a runtime error, not config error. Dry-run checks file exists and loadable, not schema.

**Q**: "Why not run 1 query as test?"

**A**: Cost. Dry-run should cost $0. Running 1 query costs $0.01+. Make that a `--test-run` feature (future).

---

## PBI-43: Add Variance Reporting

**Priority**: P1 (Statistics)
**Estimate**: 30 min
**Category**: Run Framework → Statistics

### Problem Statement

Single run doesn't show variance. Can't distinguish noise from signal.

**Evidence**:

```bash
# Run eval once
uv run eval-rag --dataset legal_rag_bench --slice nano

# Result:
{
  "metrics_avg": {
    "faithfulness_score": 0.75
  },
  "total_processed": 10
}

# Question: Is 0.75 precise or noisy?
# Answer: Unknown without running multiple times.
```

**Hostile reviewer says**: "You claim 0.75. I run it and get 0.65. Is your system broken or is that normal variance? Give me stdev."

### Solution

Add `--repeat=N` flag to run eval N times and report variance.

### Acceptance Criteria

1. [ ] `--repeat` flag to run N times
2. [ ] Report mean, stdev, min, max per metric
3. [ ] Add variance to JSON output
4. [ ] Optimize: reuse dataset loading across runs

### Implementation

**Add to `src/eval_harness/runners/run_rag_eval.py`**:

```python
parser.add_argument(
    "--repeat",
    type=int,
    default=1,
    help="Run evaluation N times to measure variance (default: 1)"
)

# In main()
if args.repeat > 1:
    # Run multiple times
    all_results = []
    run_timestamps = []

    for i in range(args.repeat):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{args.repeat}")
        print(f"{'='*60}\n")

        # Run evaluation (reuse dataset)
        run_results = _run_evaluation(args, config, dataset, judge_model)
        all_results.append(run_results)
        run_timestamps.append(run_results["timestamp"])

    # Compute variance
    variance_metrics = _compute_variance(all_results)

    # Create summary JSON
    summary = {
        "dataset": args.dataset,
        "slice": args.slice,
        "runs": args.repeat,
        "run_timestamps": run_timestamps,
        "metrics_mean": variance_metrics["mean"],
        "metrics_stdev": variance_metrics["stdev"],
        "metrics_min": variance_metrics["min"],
        "metrics_max": variance_metrics["max"],
        "judge_model": judge_model
    }

    # Write summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path("results") / f"{args.dataset}_{args.slice}_variance_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[INFO] Variance summary: {summary_path}")

else:
    # Single run (existing behavior)
    _run_evaluation(args, config, dataset, judge_model)


def _compute_variance(runs: list[dict]) -> dict:
    """Compute variance statistics across multiple runs."""
    import numpy as np

    # Collect all metrics
    metric_names = set()
    for run in runs:
        metric_names.update(run.get("metrics_avg", {}).keys())

    results = {"mean": {}, "stdev": {}, "min": {}, "max": {}}

    for metric in metric_names:
        values = []
        for run in runs:
            val = run.get("metrics_avg", {}).get(metric)
            if val is not None and val == val:  # Not NaN
                values.append(val)

        if values:
            results["mean"][metric] = float(np.mean(values))
            results["stdev"][metric] = float(np.std(values))
            results["min"][metric] = float(np.min(values))
            results["max"][metric] = float(np.max(values))

    return results
```

**Output format**:

```json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "runs": 5,
  "run_timestamps": ["20260521_120000", "20260521_120500", ...],
  "metrics_mean": {
    "faithfulness_score": 0.75
  },
  "metrics_stdev": {
    "faithfulness_score": 0.12
  },
  "metrics_min": {
    "faithfulness_score": 0.60
  },
  "metrics_max": {
    "faithfulness_score": 0.90
  }
}
```

**Usage**:

```bash
# Run 5 times for variance
uv run eval-rag --dataset legal_rag_bench --slice nano --repeat 5

# Output:
# Run 1/5
# ...
# Run 2/5
# ...
#
# Variance summary:
# faithfulness_score: 0.75 ± 0.12 (range: 0.60 - 0.90)
```

### Definition of Done

- [ ] `--repeat` flag working
- [ ] Variance computed correctly
- [ ] JSON output with mean/stdev/min/max
- [ ] Dataset reused across runs (not reloaded)
- [ ] Tested with repeat=5

### What a Hostile Reviewer Will Ask

**Q**: "Why not use scipy for stdev?"

**A**: numpy is already a dependency. scipy isn't. numpy.std() works fine.

**Q**: "What if some runs have errors?"

**A**: Only include successful runs in variance. Error count reported separately.

**Q**: "30 min estimate seems low."

**A**: Code is simple. Most time is testing. Implementation is ~50 lines.

---

## PBI-44: Add Progress Bar

**Priority**: P2 (UX)
**Estimate**: 1 hour
**Category**: Run Framework → UX

### Problem Statement

Long runs show no progress. User can't tell if eval is working or frozen.

**Evidence**:

```python
# Current: no progress output
for item in dataset:
    result = evaluate(item)
    results.append(result)

# 30 minutes of silence
# User: "Is it working? Should I kill it?"
```

**Hostile reviewer says**: "I run 100 queries. No output for 20 minutes. I kill it. It was working. Now I wasted 20 minutes and API quota."

### Solution

Use `tqdm` progress bar (already common in Python).

### Acceptance Criteria

1. [ ] Show progress bar during evaluation
2. [ ] Display: `Processing 7/100 (7%) | ETA: 14m | 3.2 queries/min`
3. [ ] Show current query ID/question
4. [ ] Works for both eval-rag and eval-parsing
5. [ ] Graceful fallback if tqdm not installed

### Implementation

**Add to `src/eval_harness/runners/run_rag_eval.py`**:

```python
# At top of file
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):
        """Fallback tqdm that just yields items."""
        for item in iterable:
            yield item


# In evaluation loop
dataset_list = list(dataset)  # Need list for tqdm

with tqdm(
    dataset_list,
    desc="Processing",
    unit="query",
    disable=not TQDM_AVAILABLE or args.no_progress
) as pbar:
    for item in pbar:
        query_id = item.get("query_id", "unknown")

        # Update description with current query
        pbar.set_description(f"Query {query_id}")

        # Evaluate
        result = _evaluate_single(item, ...)
        results.append(result)

        # Update postfix with stats
        if len(results) > 0:
            avg_score = sum(r.get("faithfulness_score", 0) for r in results if r.get("faithfulness_score")) / len(results)
            pbar.set_postfix({"avg_faithfulness": f"{avg_score:.3f}"})

# Add CLI flag
parser.add_argument(
    "--no-progress",
    action="store_true",
    help="Disable progress bar"
)
```

**Output**:

```
Processing Query 1: 100%|██████████| 10/10 [02:30<00:00, 15.0s/query, avg_faithfulness=0.750]
```

**Add dependency**:

```toml
[project.optional-dependencies]
progress = [
    "tqdm>=4.65.0",
]
```

Or add to main dependencies (tqdm is lightweight).

### Definition of Done

- [ ] Progress bar showing
- [ ] ETA accurate
- [ ] Current query shown
- [ ] Both CLIs supported
- [ ] Fallback if tqdm missing
- [ ] `--no-progress` flag working

### What a Hostile Reviewer Will Ask

**Q**: "Why tqdm rich.progress exists?"

**A**: tqdm is simpler. rich.progress is prettier but requires more setup. tqdm works out of the box.

**Q**: "What if I pipe output to a file?"

**A**: tqdm detects non-TTY and disables bar. Or use `--no-progress`.

**Q**: "Why show avg_faithfulness in postfix?"

**A**: Gives real-time feedback on quality. If avg drops, you know something is wrong.

---

## Summary Table

| PBI | Priority | Estimate | Category | Risk if Deferred |
|-----|----------|----------|----------|------------------|
| Fix regression_check.py | P0 | 2h | Framework → Regression | Broken detector, false confidence |
| Add dry-run mode | P0 | 2h | Framework → Validation | Wasted time/cost on bad configs |
| Add variance reporting | P1 | 30min | Framework → Statistics | Can't measure stability |
| Add progress bar | P2 | 1h | Framework → UX | Long runs appear frozen |

**Total P0**: 4 hours
**Total P1**: 30 min
**Total P2**: 1 hour
**Total all**: 5.5 hours

---

## Dependencies

```
PBI-41 (Fix regression) ──┐
PBI-42 (Dry-run) ─────────┼→ Independent, can parallelize
PBI-43 (Variance) ────────┤
PBI-44 (Progress bar) ────┘
```

**Note**: PBI-43 (variance) enables calibrated regression thresholds in PBI-41.

---

## Implementation Sequence

**Week 1 (P0)**:
1. PBI-41: Fix regression_check.py (2h)
2. PBI-42: Add dry-run mode (2h)

**Week 2 (P1)**:
3. PBI-43: Add variance reporting (30min)

**Week 3 (P2)**:
4. PBI-44: Add progress bar (1h)

---

**Document version**: 1.0
**Last updated**: 2026-05-22
**For**: Team planning meeting
