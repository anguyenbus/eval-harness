# Call Orchestration: PBIs

**Capability**: Call Orchestration — coordinating calls to external systems (parsers, RAG, LLM judges).

**Current State**: Sequential per-query execution. Missing: retry logic, timeout enforcement, token tracking, resume capability.

---

## PBI-6: Add Retry Logic with Exponential Backoff

**Priority**: P0 (Reliability)
**Estimate**: 2 hours
**Category**: Call Orchestration → Error Handling

### Problem

LLM API calls fail transiently (rate limits, network blips, server errors). Current code: no retry. One transient error = entire query fails with score=0.0.

From [`deepeval_adapter.py:196`](src/eval_harness/adapters/deepeval_adapter.py:196):

```python
try:
    metric.measure(test_case)
    results["faithfulness"] = float(metric.score)
except Exception as e:
    results["faithfulness"] = 0.0  # ← No retry
```

### Acceptance Criteria

1. [ ] Add retry wrapper for LLM calls
2. [ ] Use exponential backoff: 1s, 2s, 4s, 8s
3. [ ] Max 3 retries before failing
4. [ ] Log retry attempts to stderr
5. [ ] Respect `Retry-After` header if present
6. [ ] Configurable via `--max-retries` flag

### Implementation Notes

**Create `src/eval_harness/observability/retry.py`**:

```python
import time
import random
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    jitter: bool = True
):
    """
    Retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Add random jitter to avoid thundering herd
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random())  # 0.5x to 1.5x

                    print(
                        f"[WARN] {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}",
                        file=sys.stderr
                    )
                    print(f"[WARN] Retrying in {delay:.1f}s...", file=sys.stderr)
                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator
```

**Apply to DeepEval adapter**:

```python
from eval_harness.observers.retry import retry_with_backoff

class DeepEvalEvaluator:
    @beartype
    def compute_metrics(self, ...):
        for metric_name, metric in self._metrics.items():
            try:
                # Wrap the measure call with retry
                if metric_name == "faithfulness":
                    self._retry_metric(lambda: metric.measure(test_case))
                    results["faithfulness"] = float(metric.score)
```

**CLI addition**:

```python
parser.add_argument(
    "--max-retries",
    type=int,
    default=3,
    help="Max retry attempts for LLM calls (default: 3)"
)
```

### Retry Behavior Table

| Attempt | Delay (with jitter) | Total wait so far |
|---------|---------------------|-------------------|
| 1 (fail) | 0.75-1.5s | ~1s |
| 2 (fail) | 1.5-3s | ~3s |
| 3 (fail) | 3-6s | ~8s |
| 4 (final) | — | — |

**Total worst-case wait**: ~8 seconds per metric

### Definition of Done

- [ ] Retry wrapper implemented
- [ ] Applied to all LLM metric calls
- [ ] Logs show retry attempts
- [ ] `--max-retries` flag working
- [ ] Unit tests for retry logic
- [ ] Tested with simulated failures

---

## PBI-7: Add Timeout Enforcement

**Priority**: P0 (Correctness)
**Estimate**: 1 hour
**Category**: Call Orchestration → Timeouts

### Problem

LLM calls can hang indefinitely. OpenAI timeout is 600s default, but network issues can cause longer hangs. Current code: no timeout.

**Impact**: Single hung query = entire evaluation stuck.

### Acceptance Criteria

1. [ ] Add timeout for each LLM metric call
2. [ ] Default timeout: 120 seconds per metric
3. [ ] Configurable via `--metric-timeout` flag
4. [ ] Timeout raises clear exception with metric name
5. [ ] Timeout doesn't count toward retry limit

### Implementation Notes

**Extend retry wrapper with timeout**:

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

def with_timeout(func: Callable[..., T], timeout_seconds: float) -> Callable[..., T]:
    """
    Wrap function to timeout after specified seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                raise TimeoutError(
                    f"{func.__name__} exceeded {timeout_seconds}s timeout"
                )
    return wrapper
```

**Combine with retry**:

```python
def retry_with_backoff_and_timeout(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    timeout_seconds: float = 120.0
):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func_with_timeout = with_timeout(func, timeout_seconds)
        return retry_with_backoff(max_retries, base_delay, max_delay)(func_with_timeout)
    return decorator
```

**CLI addition**:

```python
parser.add_argument(
    "--metric-timeout",
    type=float,
    default=120.0,
    help="Timeout in seconds for each metric computation (default: 120)"
)
```

### Timeout Scenarios

| Scenario | Without timeout | With timeout |
|----------|-----------------|--------------|
| Network hang | Wait forever | Fail after 120s |
| API overload | Wait 600s | Fail after 120s |
| Normal query | Complete in 5s | Complete in 5s |

### Definition of Done

- [ ] Timeout wrapper implemented
- [ ] Combined with retry logic
- [ ] `--metric-timeout` flag working
- [ ] Clear error message on timeout
- [ ] Tested with actual timeout scenarios

---

## PBI-8: Track Token Usage

**Priority**: P1 (Cost Management)
**Estimate**: 2 hours
**Category**: Call Orchestration → Observability

### Problem

No token tracking. Can't answer:
- How much did this eval run cost?
- Which metric is most expensive?
- Are we being efficient?

### Acceptance Criteria

1. [ ] Wrap LLM calls to count input/output tokens
2. [ ] Report per-metric token counts
3. [ ] Add total token count to JSON summary
4. [ ] Add estimated cost in USD
5. [ ] Support both OpenAI and Anthropic pricing

### Implementation Notes

**Create `src/eval_harness/observability/token_tracker.py`**:

```python
from collections import defaultdict
from typing import Dict

# Pricing as of 2025-05 (USD per 1M tokens)
PRICING = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    }
}

class TokenTracker:
    def __init__(self):
        self.tokens: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )

    def record(self, metric_name: str, input_tokens: int, output_tokens: int):
        self.tokens[metric_name]["input"] += input_tokens
        self.tokens[metric_name]["output"] += output_tokens

    def summarize(self, model: str, provider: str = "openai") -> Dict:
        total_input = sum(m["input"] for m in self.tokens.values())
        total_output = sum(m["output"] for m in self.tokens.values())

        pricing = PRICING[provider][model]
        cost = (
            (total_input / 1_000_000) * pricing["input"] +
            (total_output / 1_000_000) * pricing["output"]
        )

        return {
            "total_tokens": total_input + total_output,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "estimated_cost_usd": round(cost, 4),
            "per_metric": {
                name: {
                    "input_tokens": m["input"],
                    "output_tokens": m["output"],
                    "total": m["input"] + m["output"]
                }
                for name, m in self.tokens.items()
            }
        }
```

**Integration with DeepEval**:

DeepEval doesn't expose token counts directly. Need to wrap the LLM client:

```python
class TokenTrackingWrapper:
    def __init__(self, base_llm, tracker: TokenTracker, metric_name: str):
        self._base = base_llm
        self._tracker = tracker
        self._metric_name = metric_name

    def generate(self, prompt: str, *args, **kwargs) -> str:
        # Call base LLM
        response = self._base.generate(prompt, *args, **kwargs)

        # Try to extract token usage
        # OpenAI includes usage in response
        if hasattr(response, 'usage'):
            self._tracker.record(
                self._metric_name,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )

        return response
```

**JSON output format**:

```json
{
  "token_usage": {
    "total_tokens": 125000,
    "input_tokens": 100000,
    "output_tokens": 25000,
    "estimated_cost_usd": 1.1250,
    "per_metric": {
      "faithfulness": {
        "input_tokens": 30000,
        "output_tokens": 5000,
        "total": 35000
      },
      "context_precision": {...}
    }
  }
}
```

**Pricing reference table** (keep updated):

```python
# As of 2025-05
# GPT-4o: $2.50 input / $10.00 output per 1M tokens
# GPT-4o-mini: $0.15 input / $0.60 output per 1M tokens
# Claude 3.5 Sonnet: $3.00 input / $15.00 output per 1M tokens
```

### Definition of Done

- [ ] Token tracking implemented
- [ ] Per-metric breakdown in JSON
- [ ] Cost estimation working
- [ ] Pricing table documented
- [ ] Tested with real eval run

---

## PBI-9: Add Resume Capability

**Priority**: P2 (User Experience)
**Estimate**: 6 hours
**Category**: Call Orchestration → Resilience

### Problem

No checkpointing. If evaluation crashes at query 97/100:
- Lose all progress
- Must restart from query 1
- Wasted API costs

### Acceptance Criteria

1. [ ] Write checkpoint after each successful query
2. [ ] Checkpoint file: `{timestamp}_checkpoint.csv`
3. [ ] `--resume-from` flag to continue from checkpoint
4. [ ] Skip already-completed query_ids
5. [ ] Merge checkpoint with final results

### Implementation Notes

**Checkpoint format** (CSV):

```csv
query_id,status,timestamp
1,completed,20260521_123045
2,completed,20260521_123046
3,failed,20260521_123047
4,completed,20260521_123048
```

**Resume logic**:

```python
def load_completed_ids(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()

    completed = set()
    with open(checkpoint_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "completed":
                completed.add(row["query_id"])
    return completed

def main():
    # ...
    if args.resume_from:
        completed_ids = load_completed_ids(Path(args.resume_from))
        print(f"Resuming from checkpoint, skipping {len(completed_ids)} completed queries")
    else:
        completed_ids = set()
        checkpoint_path = output_path.with_suffix('.checkpoint.csv')

    # Initialize checkpoint file
    if not args.resume_from:
        with open(checkpoint_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "status", "timestamp"])

    for query in dataset:
        if query["query_id"] in completed_ids:
            print(f"Skipping {query['query_id']} (already completed)")
            continue

        try:
            result = process_query(query)
            writer.writerow([query["query_id"], "completed", datetime.now().isoformat()])
            csv_file.flush()
        except Exception as e:
            writer.writerow([query["query_id"], "failed", datetime.now().isoformat()])
            csv_file.flush()
            raise
```

**CLI addition**:

```python
parser.add_argument(
    "--resume-from",
    type=Path,
    help="Path to checkpoint file to resume from"
)
```

### Recovery Scenarios

| Scenario | Without resume | With resume |
|----------|----------------|-------------|
| Crash at 97/100 | Restart from 1, pay for 97 again | Resume at 98 |
| Ctrl+C at 50/100 | Lose all progress | Resume at 51 |
| Network outage 30min | Timeout, restart | Resume after fix |

### Definition of Done

- [ ] Checkpoint file created
- [ ] Resume from checkpoint working
- [ ] Skipped queries logged
- [ ] Checkpoint merged with final results
- [ ] `--resume-from` flag working
- [ ] Tested crash/recovery scenarios

---

## Dependencies

```
PBI-6 (Retry) ─┐
PBI-7 (Timeout)├→ Enables robust token tracking (PBI-8)
               │
               └→ Makes resume less critical (PBI-9 becomes P2)
```

**Execution order**:
1. PBI-6 and PBI-7 first (block reliability)
2. PBI-8 second (builds on stable calls)
3. PBI-9 third (nice-to-have after reliability)

---

## Summary Table

| PBI | Priority | Estimate | Dependencies | Risk |
|-----|----------|----------|--------------|------|
| Add retry logic | P0 | 2h | None | Low |
| Add timeout enforcement | P0 | 1h | None | Low |
| Track token usage | P1 | 2h | PBI-6, PBI-7 | Medium (API changes) |
| Add resume capability | P2 | 6h | None | Low |

**Total P0-P1 effort**: 5 hours
**Total all PBIs**: 11 hours

---

## Combined with Load Evaluation PBIs

| Category | P0 | P1 | P2 | Total |
|----------|-----|-----|-----|-------|
| Load Evaluation | 2h | 5.5h | 6h | 13.5h |
| Call Orchestration | 3h | 2h | 6h | 11h |
| **Combined** | **5h** | **7.5h** | **12h** | **24.5h** |

**Sprint capacity** (40 hours): Can complete P0 + P1 + 60% of P2 in one sprint.
