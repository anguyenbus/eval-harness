# Load Evaluation: PBIs

**Capability**: Load evaluation — running evaluations on datasets of varying sizes to measure performance under load.

**Current State**: Implemented for dataset slicing and truncation. Missing: variance measurement, error handling rigor, concurrency.

---

## PBI-1: Fix Error-as-Zero Bug

**Priority**: P0 (Correctness)
**Estimate**: 2 hours
**Category**: Load Evaluation → Error Handling

### Problem

When LLM judge calls fail, scores default to `0.0` ([`deepeval_adapter.py:196`](src/eval_harness/adapters/deepeval_adapter.py:196)):

```python
except Exception as e:
    print(f"[ERROR] faithfulness failed: {e}", file=sys.stderr)
    results["faithfulness"] = 0.0  # ← Silent zero
```

This corrupts averages. If 3 of 10 queries fail silently, your "average 0.75" is actually 3 failures + 7 good runs.

### Acceptance Criteria

1. [ ] Failed metric scores use `math.nan` instead of `0.0`
2. [ ] CSV writes empty string for NaN values (not "nan")
3. [ ] JSON summary excludes NaN values from averages
4. [ ] Error count reported separately from valid results
5. [ ] `--fail-on-error` flag added to CLI (optional, exits on first error)

### Implementation Notes

**Files to modify**:
- `src/eval_harness/adapters/deepeval_adapter.py` — change error handling to return `nan`
- `src/eval_harness/runners/run_rag_eval.py` — update CSV writer to handle NaN
- `src/eval_harness/runners/run_parsing_eval.py` — same for parsing

**Code change**:
```python
# Before
results["faithfulness"] = 0.0

# After
import math
results["faithfulness"] = math.nan
```

**CSV handling**:
```python
# Convert NaN to empty string for CSV
def safe_float(x):
    return "" if math.isnan(x) else round(x, 4)
```

**JSON summary**:
```python
# Filter out NaN when calculating averages
valid_scores = [s for s in scores if not math.isnan(s)]
if valid_scores:
    avg = sum(valid_scores) / len(valid_scores)
else:
    avg = math.nan
```

### Definition of Done

- [ ] All metric failures return NaN
- [ ] CSV files show empty cells (not "0.0") for failed metrics
- [ ] JSON summary `metrics_avg` excludes NaN from calculation
- [ ] Unit tests added for NaN handling
- [ ] Documentation updated

---

## PBI-2: Run Nano Slice 5x for Judge Variance

**Priority**: P1 (Rigor)
**Estimate**: 30 minutes execution + 1 hour analysis
**Category**: Load Evaluation → Statistical Rigor

### Problem

No measurement of judge variance. When metric changes from 0.75 to 0.70, is it:
- Signal (system regressed)?
- Noise (judge variance)?

Unknown without variance measurement.

### Acceptance Criteria

1. [ ] Execute `uv run eval-rag --slice nano` 5 times
2. [ ] Collect all 5 JSON summaries
3. [ ] Calculate mean ± stdev for each metric
4. [ ] Document variance in `docs/judge-variance-report.md`
5. [ ] Update regression threshold from 5% to "2σ of judge variance"

### Execution Plan

```bash
# Run 5 times
for i in {1..5}; do
  uv run eval-rag --slice nano --rag stub-local
  sleep 60  # Rate limit avoidance
done

# Extract metrics
python scripts/analyze_variance.py results/legal_rag_bench_nano_results_*.json
```

### Variance Analysis Script

Create `scripts/analyze_variance.py`:

```python
import json
import sys
from pathlib import Path
import numpy as np

def analyze_variance(json_files):
    metrics = {}
    for f in json_files:
        data = json.load(open(f))
        for name, value in data["metrics_avg"].items():
            metrics.setdefault(name, []).append(value)

    results = {}
    for name, values in metrics.items():
        arr = np.array(values)
        results[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "cv": float(np.std(arr) / np.mean(arr))  # Coefficient of variation
        }
    return results

if __name__ == "__main__":
    files = list(Path("results").glob("legal_rag_bench_nano_results_*.json"))
    results = analyze_variance(files)
    print(json.dumps(results, indent=2))
```

### Expected Output Format

```json
{
  "faithfulness_score": {
    "mean": 0.89,
    "std": 0.03,
    "min": 0.85,
    "max": 0.92,
    "cv": 0.034
  },
  "context_precision_score": {
    "mean": 0.25,
    "std": 0.08,
    "min": 0.15,
    "max": 0.35,
    "cv": 0.32
  }
}
```

### Definition of Done

- [ ] 5 runs completed successfully
- [ ] Variance report generated
- [ ] Regression threshold updated based on 2σ
- [ ] Documented in `docs/judge-variance-report.md`

---

## PBI-3: Add Concurrency Control

**Priority**: P1 (Performance)
**Estimate**: 4 hours
**Category**: Load Evaluation → Concurrency

### Problem

Current execution is sequential ([`run_rag_eval.py:528`](src/eval_harness/runners/run_rag_eval.py:528)). Full slice (100 queries) with DeepEval takes ~2 hours due to sequential LLM calls.

Async evaluator exists ([`deepeval_adapter.py:237`](src/eval_harness/adapters/deepeval_adapter.py:237)) but runner doesn't use it.

### Acceptance Criteria

1. [ ] Add `--concurrent` flag to `eval-rag` CLI
2. [ ] Use `asyncio.gather()` with semaphore for concurrent queries
3. [ ] Respect `max_concurrent` from config (default 10)
4. [ ] Maintain CSV write order (write rows in query_id order)
5. [ ] Add progress indicator showing active/completed/remaining

### Implementation Notes

**Files to modify**:
- `src/eval_harness/runners/run_rag_eval.py` — add async main
- `src/eval_harness/adapters/deepeval_adapter.py` — already has `async_batch_compute_metrics`

**Architecture**:
```python
async def process_query_async(query_data, rag_adapter, evaluator):
    # Single query processing
    output = rag_adapter.query(...)
    result = await evaluator.async_compute_metrics_with_reasoning(...)
    return result

async def main_async():
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(query):
        async with semaphore:
            return await process_query_async(query, ...)

    tasks = [process_with_semaphore(q) for q in dataset]
    results = await asyncio.gather(*tasks)

    # Sort by query_id to maintain CSV order
    results.sort(key=lambda x: x["query_id"])
    for r in results:
        writer.writerow(r)
```

**CLI addition**:
```python
parser.add_argument(
    "--concurrent",
    type=int,
    default=1,
    help="Number of concurrent queries (default: 1 for sequential)"
)
```

### Trade-offs

| Concurrent | Speed | Cost | Risk |
|------------|-------|------|------|
| 1 | 2 hours | Baseline | None |
| 5 | 24 min | Same | Rate limit |
| 10 | 12 min | Same | Rate limit |
| 20 | 6 min | Same | High rate limit risk |

### Definition of Done

- [ ] `--concurrent` flag working
- [ ] Semaphore respects max_concurrent config
- [ ] CSV rows written in query_id order
- [ ] Progress indicator functional
- [ ] Tested with 1, 5, 10 concurrent values

---

## PBI-4: Add Resource Monitoring

**Priority**: P2 (Observability)
**Estimate**: 4 hours
**Category**: Load Evaluation → Monitoring

### Problem

No tracking of CPU/memory during evaluation. Can't answer:
- Did evaluation hit memory limits?
- Was CPU saturated?
- What's the per-query resource cost?

### Acceptance Criteria

1. [ ] Track peak memory usage per evaluation run
2. [ ] Track CPU utilization (percentage)
3. [ ] Track wall-clock time per query
4. [ ] Write resource summary to JSON
5. [ ] Alert if memory > 80% of available

### Implementation Notes

**Use `psutil` library**:
```python
import psutil
import time

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss
        self.peak_memory = self.start_memory
        self.start_time = time.time()

    def update(self):
        current_mem = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_mem)
        cpu_percent = self.process.cpu_percent(interval=0.1)

    def summary(self):
        return {
            "peak_memory_mb": self.peak_memory / 1024 / 1024,
            "cpu_percent_avg": self.process.cpu_percent(),
            "duration_seconds": time.time() - self.start_time
        }
```

**Integration in runner**:
```python
monitor = ResourceMonitor()
for query in dataset:
    monitor.update()
    # ... process query ...

json_summary["resources"] = monitor.summary()
```

### JSON Output Format

```json
{
  "resources": {
    "peak_memory_mb": 2048.5,
    "cpu_percent_avg": 45.2,
    "duration_seconds": 7234.1
  }
}
```

### Definition of Done

- [ ] Resource tracking in both eval-parsing and eval-rag
- [ ] JSON summary includes resource section
- [ ] Memory alert if > 80%
- [ ] Tested with large dataset (full slice)

---

## PBI-5: Add Deterministic Random Seed Control

**Priority**: P2 (Reproducibility)
**Estimate**: 2 hours
**Category**: Load Evaluation → Reproducibility

### Problem

Randomness not controlled. LLM temperature=0 helps, but:
- Embedding models may have nondeterminism
- ChromaDB retrieval may have ties broken randomly
- Any third-party code may use randomness

### Acceptance Criteria

1. [ ] Add `--seed` flag to CLI
2. [ ] Set `PYTHONHASHSEED` environment variable
3. [ ] Configure random seeds for numpy, torch if used
4. [ ] Document seed in JSON summary

### Implementation

```python
# CLI
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility"
)

# In main()
import os
import random
import numpy as np

seed = args.seed
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

# JSON summary
json_summary["reproducibility"] = {"random_seed": seed}
```

### Definition of Done

- [ ] Seed parameter working
- [ ] Reproducible runs verified (same seed = same results)
- [ ] Documented in JSON

---

## Dependencies

```
PBI-1 (Fix error-as-zero) → BLOCKS → PBI-2 (Variance measurement)
                                   ↓
PBI-2 (Variance) → INFORMS → PBI-3 (Concurrency limits)
```

**Execution order**:
1. PBI-1 first (unblocks accurate variance measurement)
2. PBI-2 second (informs safe concurrency level)
3. PBI-3 third (apply variance findings)
4. PBI-4, PBI-5 can run in parallel

---

## Summary Table

| PBI | Priority | Estimate | Dependencies | Risk |
|-----|----------|----------|--------------|------|
| Fix error-as-zero | P0 | 2h | None | Low |
| Run nano 5x variance | P1 | 1.5h | PBI-1 | Medium (API costs) |
| Add concurrency | P1 | 4h | PBI-2 | Medium (rate limits) |
| Resource monitoring | P2 | 4h | None | Low |
| Random seed control | P2 | 2h | None | Low |

**Total P0-P1 effort**: 7.5 hours
**Total all PBIs**: 13.5 hours
