# Run Framework, Metrics Retrieval, Store, Comparison: PBIs

**Capabilities**: Run Evaluation Framework (CLI), Metrics Retrieval (output), Store Results (persistence), Comparison (regression).

**Current State**: CLI works but has broken tools (regression_check.py, html_summary.py). Results stored as flat files.

---

## PBI-26: Fix regression_check.py

**Priority**: P0 (Correctness)
**Estimate**: 2 hours
**Category**: Run Framework / Comparison → Regression

### Problem

[`regression_check.py:39`](src/eval_harness/reporting/regression_check.py:39) expects:

```python
for metric_name, current_data in current.get("metrics", {}).items():
    if current_data.get("severity") != "blocker":  # ← Expects 'metrics' dict with severity
        continue
```

But actual JSON uses `metrics_avg` without severity:

```json
{
  "metrics_avg": {
    "faithfulness_score": 0.9167,
    "context_precision_score": 0.1458
  }
}
```

**Impact**: No regression detection. Broken tool worse than no tool.

### Acceptance Criteria

1. [ ] Fix JSON path: `metrics` → `metrics_avg`
2. [ ] Remove severity check (not in JSON format)
3. [ ] Make threshold configurable
4. [ ] Add relative AND absolute threshold support
5. [ ] Test with actual result files

### Implementation Notes

**Rewrite `src/eval_harness/reporting/regression_check.py`**:

```python
"""
Regression checking against baseline results.

FIXED: Now works with actual JSON format using metrics_avg.
"""
import json
from pathlib import Path
from typing import Any

def check_regression(
    current_results: Path,
    baseline_path: Path,
    threshold: float = 0.05,
    absolute_threshold: float | None = None,
    metric_blocklist: list[str] | None = None
) -> None:
    """
    Compare current results against baseline and check for regressions.

    Args:
        current_results: Path to current JSON summary
        baseline_path: Path to baseline JSON summary
        threshold: Relative regression threshold (default 5%)
        absolute_threshold: Optional absolute threshold
        metric_blocklist: Metrics to skip (e.g., "total_ms")

    Raises:
        RuntimeError: If regression is detected
        FileNotFoundError: If either file doesn't exist
    """
    # Load files
    with open(current_results) as f:
        current = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    # Check for regressions
    regressions = []
    improvements = []

    # Use metrics_avg (actual JSON format)
    current_metrics = current.get("metrics_avg", {})
    baseline_metrics = baseline.get("metrics_avg", {})

    # All metrics in either file
    all_metrics = set(current_metrics.keys()) | set(baseline_metrics.keys())

    # Skip blocklisted metrics
    if metric_blocklist:
        all_metrics -= set(metric_blocklist)

    for metric_name in sorted(all_metrics):
        if metric_name not in baseline_metrics:
            continue  # New metric, can't compare

        current_score = current_metrics.get(metric_name, 0.0)
        baseline_score = baseline_metrics[metric_name]

        # Calculate relative change
        if baseline_score != 0:
            relative_change = (current_score - baseline_score) / baseline_score
        else:
            relative_change = 0.0

        absolute_change = current_score - baseline_score

        # Check regression (score decreased beyond threshold)
        is_regression = False
        is_improvement = False

        if absolute_threshold is not None:
            # Use absolute threshold
            if absolute_change < -absolute_threshold:
                is_regression = True
            elif absolute_change > absolute_threshold:
                is_improvement = True
        else:
            # Use relative threshold
            if relative_change < -threshold:
                is_regression = True
            elif relative_change > threshold:
                is_improvement = True

        if is_regression:
            regressions.append({
                "metric": metric_name,
                "baseline": baseline_score,
                "current": current_score,
                "relative_change": relative_change * 100,
                "absolute_change": absolute_change
            })

        if is_improvement:
            improvements.append({
                "metric": metric_name,
                "baseline": baseline_score,
                "current": current_score,
                "relative_change": relative_change * 100
            })

    # Print summary
    if improvements:
        print("\nImprovements:")
        for imp in improvements:
            print(f"  ✓ {imp['metric']}: {imp['baseline']:.4f} → {imp['current']:.4f} ({imp['relative_change']:+.1f}%)")

    if regressions:
        msg = "\n🔴 Regression detected:\n"
        for reg in regressions:
            msg += f"  - {reg['metric']}: {reg['baseline']:.4f} → {reg['current']:.4f} ({reg['relative_change']:.1f}%)\n"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    print("\n✓ No regressions detected")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python regression_check.py <current.json> <baseline.json>")
        sys.exit(1)

    current = Path(sys.argv[1])
    baseline = Path(sys.argv[2])

    try:
        check_regression(current, baseline, threshold=0.05, metric_blocklist=["total_ms"])
        print("\n✓ Regression check passed")
    except RuntimeError as e:
        print(f"\n✗ Regression check failed: {e}")
        sys.exit(1)
```

**Test with real files**:

```bash
# Should pass (no regression)
python src/eval_harness/reporting/regression_check.py \
    results/legal_rag_bench_nano_results_20260521_211101.json \
    results/legal_rag_bench_nano_results_20260520_223534.json

# Should fail if regression exists
echo $?
```

### Definition of Done

- [ ] JSON path fixed (`metrics_avg`)
- [ ] Severity check removed
- [ ] Relative and absolute thresholds
- [ ] Metric blocklist supported
- [ ] Tested with actual result files
- [ ] Exit code 1 on regression

---

## PBI-27: Fix html_summary.py

**Priority**: P0 (Correctness)
**Estimate**: 2 hours
**Category**: Metrics Retrieval → Reporting

### Problem

[`html_summary.py:24`](src/eval_harness/reporting/html_summary.py:24) expects:

```python
pass_count = len(df[df["label"] == "pass"])  # ← Expects 'label' column
score_mean = df["score"].mean()  # ← Expects 'score' column
```

But actual RAG CSV has different columns:

```csv
query_id,question,gold_answer,generated_answer,faithfulness_score,judge_verdict,...
```

**Impact**: HTML reports broken, can't visualize results.

### Acceptance Criteria

1. [ ] Read actual column names from CSV
2. [ ] Generate pass/fail from `judge_verdict` column
3. [ ] Handle both parsing and RAG CSV formats
4. [ ] Generate HTML with correct columns
5. [ ] Add metric selection UI

### Implementation Notes

**Rewrite `src/eval_harness/reporting/html_summary.py`**:

```python
"""
HTML summary report generator.

FIXED: Now works with actual CSV formats from eval-parsing and eval-rag.
"""
from pathlib import Path
from typing import Any
import pandas as pd

def generate_summary(
    results_path: Path,
    output_path: Path,
    metric_column: str | None = None
) -> None:
    """
    Generate HTML summary report from evaluation results CSV.

    Auto-detects CSV format (parsing vs RAG) and generates appropriate report.

    Args:
        results_path: Path to CSV file
        output_path: Path for HTML output
        metric_column: Primary metric to use (auto-detected if None)
    """
    df = pd.read_csv(results_path)

    # Detect format
    is_rag = "judge_verdict" in df.columns
    is_parsing = "nid" in df.columns

    if is_rag:
        html = _generate_rag_html(df, results_path, metric_column)
    elif is_parsing:
        html = _generate_parsing_html(df, results_path, metric_column)
    else:
        raise ValueError("Unknown CSV format")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

def _generate_rag_html(
    df: pd.DataFrame,
    results_path: Path,
    metric_column: str | None
) -> str:
    """Generate HTML for RAG evaluation results."""

    # Use faithfulness as default metric
    if metric_column is None:
        metric_column = "faithfulness_score"

    # Calculate pass/fail
    total = len(df)
    passed = len(df[df["judge_verdict"] == "PASS"])
    failed = len(df[df["judge_verdict"] == "NEEDS_REVIEW"])
    errored = len(df[df["error"] != ""])

    pass_rate = (passed / total * 100) if total > 0 else 0

    # Metric stats
    metrics = ["faithfulness_score", "context_precision_score",
              "context_recall_score", "answer_relevancy_score"]
    metric_stats = {}
    for m in metrics:
        if m in df.columns:
            valid = df[df[m].notna()]
            if len(valid) > 0:
                metric_stats[m] = {
                    "mean": valid[m].mean(),
                    "min": valid[m].min(),
                    "max": valid[m].max()
                }

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
        table {{ width: 100%%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>RAG Evaluation Summary</h1>
    <p><strong>File:</strong> {results_path.name}</p>

    <div class="summary">
        <div class="stat-box">
            <div class="stat-value {'' if pass_rate >= 80 else 'fail'}">{pass_rate:.1f}%%</div>
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
        </tr>
"""

    for m, stats in metric_stats.items():
        html += f"""
        <tr>
            <td>{m}</td>
            <td>{stats['mean']:.4f}</td>
            <td>{stats['min']:.4f}</td>
            <td>{stats['max']:.4f}</td>
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
            <th>Faithfulness</th>
            <th>Context Precision</th>
            <th>Context Recall</th>
            <th>Answer Relevancy</th>
        </tr>
"""

    for _, row in df.iterrows():
        verdict_class = "pass" if row.get("judge_verdict") == "PASS" else "fail"
        html += f"""
        <tr>
            <td>{row['query_id']}</td>
            <td>{row.get('question', '')[:50]}...</td>
            <td class="{verdict_class}">{row.get('judge_verdict', '')}</td>
            <td>{row.get('faithfulness_score', 0):.4f}</td>
            <td>{row.get('context_precision_score', 0):.4f}</td>
            <td>{row.get('context_recall_score', 0):.4f}</td>
            <td>{row.get('answer_relevancy_score', 0):.4f}</td>
        </tr>
"""

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
    # Similar implementation for parsing metrics
    # ...
    return ""  # Implementation omitted for brevity
```

**Add CLI command**:

```toml
[project.scripts]
eval-html = "eval_harness.reporting.html_summary:main"
```

**Usage**:

```bash
uv run eval-html results/legal_rag_bench_nano_results_20260521_211101.csv
# Output: results/legal_rag_bench_nano_results_20260521_211101.html
```

### Definition of Done

- [ ] Auto-detects CSV format
- [ ] Correct columns read from RAG CSV
- [ ] HTML generated with actual data
- [ ] Pass/fail from `judge_verdict`
- [ ] Parsing format also supported
- [ ] CLI command working

---

## PBI-28: Add Dry-Run Mode

**Priority**: P0 (Safety)
**Estimate**: 2 hours
**Category**: Run Framework → Validation

### Problem

No way to validate config and dataset without running full evaluation. Must start 2-hour eval to discover config typo.

### Acceptance Criteria

1. [ ] `--dry-run` flag validates everything
2. [ ] Checks config file syntax
3. [ ] Checks dataset exists and is readable
4. [ ] Checks model is accessible
5. [ ] Shows what would run (count, estimated cost)
6. [ ] Exits without making API calls

### Implementation Notes

```python
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate config and dataset without running evaluation"
)

# In main()
if args.dry_run:
    print("DRY RUN MODE - Validating configuration...")
    print()

    # Validate config
    print(f"✓ Config file: {args.config}")
    print(f"✓ Dataset: {args.dataset}")

    # Check dataset
    try:
        dataset = list(load_dataset(args.dataset, config))
        print(f"✓ Dataset loaded: {len(dataset)} items")
    except Exception as e:
        print(f"✗ Dataset load failed: {e}")
        sys.exit(1)

    # Check model
    try:
        verify_model_accessible(judge_model, api_key)
        print(f"✓ Model accessible: {judge_model}")
    except Exception as e:
        print(f"✗ Model check failed: {e}")
        sys.exit(1)

    # Estimate cost
    est_queries = len(dataset)
    est_cost = (est_queries * 4 * 0.01)  # Rough estimate
    print(f"✓ Estimated queries: {est_queries}")
    print(f"✓ Estimated cost: ${est_cost:.2f}")

    print()
    print("Dry run passed. Run without --dry-run to execute.")
    sys.exit(0)
```

### Definition of Done

- [ ] All validation checks implemented
- [ ] Cost estimation working
- [ ] No API calls made
- [ ] Clear success/failure output

---

## PBI-29: Add Confidence Intervals

**Priority**: P1 (Statistics)
**Estimate**: 4 hours
**Category**: Metrics Retrieval → Statistics

### Problem

Point estimates without CIs mislead. "Context precision: 0.25" on n=10 — is that 0.25±0.05 or 0.25±0.25?

### Acceptance Criteria

1. [ ] Compute 95% CI for each metric
2. [ ] Use bootstrap for continuous metrics
3. [ ] Use Clopper-Pearson for binary metrics
4. [ ] Include CI in JSON summary
5. [ ] Show CI in HTML report

### Definition of Done

- [ ] Bootstrap CI implemented
- [ ] Binary CI implemented
- [ ] JSON includes ci_lower/ci_upper
- [ ] HTML shows CIs
- [ ] Documented confidence level

---

## PBI-30: Add details.json Sample to Docs

**Priority**: P1 (Documentation)
**Estimate**: 1 hour
**Category**: Metrics Retrieval → Documentation

### Problem

details.json mentioned but not shown. Reviewer wants real example (Bob & Ted juror) to verify reasoning is useful.

### Acceptance Criteria

1. [ ] Run nano slice with reasoning enabled
2. [ ] Extract Bob & Ted query (query_id="1")
3. [ ] Add real output to docs
4. [ ] Annotate key fields
5. [ ] Explain reasoning structure (L1/L2/L3)

### Definition of Done

- [ ] Real details.json in docs
- [ ] Bob & Ted example included
- [ ] Annotations added
- [ ] Reasoning structure explained

---

## PBI-31: Add Database Option

**Priority**: P2 (Storage)
**Estimate**: 12 hours
**Category**: Store Results → Database

### Problem

Flat files don't scale. Can't query across runs, no time-series tracking.

### Acceptance Criteria

1. [ ] SQLite option for local storage
2. [ ] Postgres option for remote storage
3. [ ] Schema includes all CSV + JSON columns
4. [ ] `--db-url` flag for connection string
5. [ ] Fallback to files if DB unavailable

### Definition of Done

- [ ] Both DB backends working
- [ ] Schema defined
- [ ] Config flag working
- [ ] Graceful fallback
- [ ] Migration path documented

---

## PBI-32: Add Deduplication

**Priority**: P2 (Storage)
**Estimate**: 4 hours
**Category**: Store Results → Deduplication

### Problem

Identical runs create duplicate files. Waste of space and confusion.

### Acceptance Criteria

1. [ ] Hash of config + dataset + model version
2. [ ] Check for existing run with same hash
3. [ ] Skip if exists (unless `--force`)
4. [ ] Symlink to previous result if skipped

### Definition of Done

- [ ] Hash computation working
- [ ] Existing run detection
- [ ] Skip or symlink logic
- [ ] `--force` flag implemented

---

## PBI-33: Add Retention Policy

**Priority**: P2 (Operations)
**Estimate**: 2 hours
**Category**: Store Results → Cleanup

### Problem

Old result files accumulate forever. No cleanup policy.

### Acceptance Criteria

1. [ ] `--retention-days` config option
2. [ ] Auto-delete files older than threshold
3. [ ] Dry-run mode for cleanup
4. [ ] Keep latest N runs regardless of age

### Definition of Done

- [ ] Retention config working
- [ ] Auto-cleanup implemented
- [ ] Dry-run mode
- [ ] Safety checks (keep latest N)

---

## PBI-34: Document S3 Security Model

**Priority**: P2 (Security)
**Estimate**: 2 hours
**Category**: Store Results → Security

### Problem

[`phoenix_adapter.py:538`](src/eval_harness/observability/phoenix_adapter.py:538) has `upload_parquet_to_s3()` but no security context.

**Where are credentials? What IAM role? What bucket?**

### Acceptance Criteria

1. [ ] Document IAM role requirements
2. [ ] Document bucket policy
3. [ ] Document retention
4. [ ] Document failure handling
5. [ ] Add security warnings to docs

### Implementation Notes

**Create `docs/operations/s3-upload-security.md`**:

```markdown
# S3 Upload Security Model

## IAM Requirements

### Minimum Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::eval-harness-traces",
        "arn:aws:s3:::eval-harness-traces/*"
      ]
    }
  ]
}
```

### Credential Source

Priority order:
1. AWS profile (`AWS_PROFILE`)
2. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
3. IAM role (when running on EC2/ECS)
4. Default credential chain

## Bucket Policy

### Required Bucket Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": "arn:aws:s3:::eval-harness-traces/*",
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    }
  ]
}
```

**Enforces**: SSL/TLS only

### Encryption

- Server-side encryption: AES256 (SSE-S3)
- Bucket key: Enabled

## Retention

- Parquet traces: 30 days
- Lifecycle rule: Transition to Glacier after 7 days
- Delete after 30 days

## Failure Handling

If S3 upload fails:
1. Log error with full details
2. Keep local Parquet file
3. Do NOT fail evaluation run
4. Return `False` from `upload_parquet_to_s3()`

## Security Checklist

- [ ] IAM role uses least privilege
- [ ] Bucket enforces SSL/TLS
- [ ] Server-side encryption enabled
- [ ] No credentials in code
- [ ] Credentials from secure source
- [ ] Failure doesn't expose credentials in logs
```

### Definition of Done

- [ ] Security doc created
- [ ] IAM requirements documented
- [ ] Bucket policy documented
- [ ] Retention policy documented
- [ ] Failure handling documented
- [ ] Security checklist added

---

## PBI-35: Add Progress Bar

**Priority**: P2 (UX)
**Estimate**: 1 hour
**Category**: Run Framework → UX

### Acceptance Criteria

1. [ ] Show progress bar during evaluation
2. [ ] Display: `Processing 7/100 (7%) | ETA: 14m`
3. [ ] Show current query ID
4. [ ] Works with both eval-parsing and eval-rag

### Implementation

```python
from tqdm import tqdm

# Wrap dataset iterator
for item in tqdm(dataset, desc="Processing", unit="query"):
    # Process item
```

### Definition of Done

- [ ] Progress bar showing
- [ ] ETA accurate
- [ ] Current query shown
- [ ] Both CLIs supported

---

## Dependencies

```
PBI-26 (Fix regression_check) ────┐
PBI-27 (Fix html_summary)    ├──→ Core reporting fixed
PBI-28 (Dry-run) ──────────────┘

PBI-29 (CI) ──┐
PBI-30 (docs) ├──→ Enhanced reporting
              │
PBI-31 (DB) ──┤
              ├──→ Storage improvements
PBI-32 (dedup)┤
              │
PBI-33 (retention)┘

PBI-34 (S3 docs) ────┐
                     ├──→ Security & operations
PBI-35 (progress bar)┘
```

## Summary Table

| PBI | Priority | Estimate | Category |
|-----|----------|----------|----------|
| Fix regression_check.py | P0 | 2h | Framework / Comparison |
| Fix html_summary.py | P0 | 2h | Metrics Retrieval |
| Add dry-run mode | P0 | 2h | Framework |
| Add confidence intervals | P1 | 4h | Metrics Retrieval |
| Add details.json sample | P1 | 1h | Metrics Retrieval |
| Add database option | P2 | 12h | Store Results |
| Add deduplication | P2 | 4h | Store Results |
| Add retention policy | P2 | 2h | Store Results |
| Document S3 security | P2 | 2h | Store Results |
| Add progress bar | P2 | 1h | Framework |

**Total P0**: 6 hours
**Total P1**: 5 hours
**Total P2**: 22 hours
**Total all**: 33 hours

---

## Combined Effort Summary

| Category | P0 | P1 | P2 | Total |
|----------|-----|-----|-----|-------|
| Load Evaluation | 2h | 5.5h | 6h | 13.5h |
| Call Orchestration | 3h | 2h | 6h | 11h |
| Offline Evaluation | 2h | 5h | 8h | 15h |
| A/B Routing & Shadow | 0h | 10h | 24h | 34h |
| Collect Metrics & Citations | 20h | 5h | 12h | 37h |
| Framework / Retrieval / Store | 6h | 5h | 22h | 33h |
| **TOTAL** | **33h** | **32.5h** | **78h** | **143.5h** |

**Sprint planning**:
- P0 only: ~1 week
- P0 + P1: ~2 weeks
- All PBIs: ~4 weeks

**Priority for adversarial review**: P0 items block honest result interpretation. Do them first.
