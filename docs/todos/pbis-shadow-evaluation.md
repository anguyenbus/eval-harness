# Shadow Evaluation: PBIs

**Capability**: Mirror production traffic to shadow variant for comparison without user impact.

**Current State**:
- Offline shadow evaluation exists (run same dataset on two variants)
- **No live shadow mode** — cannot mirror real-time traffic
- **No automatic result sync** — manual comparison only

---

## adversarial Review Questions (Answer Before Review)

### Q1: "Why do we need shadow mode? Can't we just AB test?"

**Answer**: Shadow mode tests safely. AB test risks users.

**Evidence**:

**AB Test** (risky):
- 50% of users get variant A (production)
- 50% of users get variant B (new code)
- Variant B has bug → 50% of users affected
- Production incident

**Shadow Mode** (safe):
- 100% of users get variant A (production)
- Variant B (shadow) receives read-only copy of traffic
- Variant B has bug → 0 users affected
- Compare results offline

**adversarial reviewer says**: "I want to test a new embedding model on production traffic. If it's bad, I don't want users to see bad answers. Shadow mode or I can't deploy."

**PBI-45 addresses this**.

---

### Q2: "Shadow mode is just logging requests. Why 16 hours?"

**Answer**: Not just logging. Full requirements:

1. **Request mirroring**: Intercept production requests, forward to shadow
2. **Async processing**: Shadow shouldn't slow production
3. **Result matching**: Match shadow responses to production by query ID
4. **Evaluation**: Run metrics on shadow vs production
5. **Storage**: Store shadow results for comparison
6. **Safety**: Shadow failures must not affect production

**Evidence**: None of this exists. Need to build from scratch.

**adversarial reviewer says**: "16 hours seems long until you list requirements. That's a full server. Not a simple logger."

---

### Q3: "Why sync results? Can't we just query shadow separately?"

**Answer**: You can, but you're missing automatic comparison.

**Evidence**:

Current workflow:
```bash
# Get production results
uv run eval-rag --dataset production_january --output prod.json

# Get shadow results (same queries)
uv run eval-rag --dataset production_january --shadow --output shadow.json

# Manually compare
# diff prod.json shadow.json (doesn't work - different timestamps)
# Write script to match query_id and compare metrics
# That's 2 hours every time
```

**adversarial reviewer says**: "I want to know: did shadow beat production? Don't make me write comparison scripts. Give me a command."

**PBI-46 addresses this**.

---

## PBI-45: Build Live Shadow Mode Server

**Priority**: P2 (Operations)
**Estimate**: 16 hours
**Category**: Shadow Evaluation → Infrastructure

### Problem Statement

Current eval system only supports offline evaluation. Cannot test against live production traffic safely.

**Evidence**:

From [`evaluation-system-walkthrough-verified.md:214`](docs/evaluation-system-walkthrough-verified.md:214):
> **Live shadow mode** does NOT exist:
> - No request mirroring from production traffic
> - No shadow evaluation endpoint
> - No production/shadow result pairing

**Impact**:
- Can't test new models on real queries
- Can't measure production-traffic performance before deploy
- Risk of deploying untested changes to production

### Architecture

```
┌─────────────────┐
│  Production API │  ← 100% of user traffic
└────────┬────────┘
         │
         │ 1. Receive request
         │
         ▼
    ┌─────────────┐
    │  Process    │  ← Generate response
    └──────┬──────┘
           │
           │ 2. Log request/response
           ▼
    ┌─────────────┐
    │   Queue     │  ← Async mirror (don't block)
    └──────┬──────┘
           │
           │ 3. Forward to shadow
           ▼
┌─────────────────┐
│ Shadow Server   │  ← Read-only copy
│  (Variant B)    │
└────────┬────────┘
         │
         │ 4. Generate shadow response
         ▼
    ┌─────────────┐
    │  Storage    │  ← Store for comparison
    └─────────────┘
```

### Acceptance Criteria

1. [ ] Shadow server accepts evaluation requests
2. [ ] Async request forwarding (non-blocking to production)
3. [ ] Query ID matching between production and shadow
4. [ ] Graceful degradation (shadow failure → no production impact)
5. [ ] Storage of shadow results
6. [ ] CLI to start shadow server
7. [ ] Health check endpoint

### Implementation

**Create `src/eval_harness/shadow/server.py`**:

```python
"""Shadow mode server for live evaluation.

Receives mirrored production requests and processes them through shadow variant.
Results are stored for later comparison with production.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

from eval_harness.stubs.rag.chromadb_query import query


# Request/response models
class ShadowRequest(BaseModel):
    """Request from production to be evaluated by shadow."""

    query_id: str = Field(description="Unique query identifier (matches production)")
    question: str = Field(description="User question")
    corpus_dir: str = Field(description="Path to document corpus")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ShadowResponse(BaseModel):
    """Shadow evaluation response."""

    query_id: str
    question: str
    generated_answer: str
    retrieved_chunks: list[dict[str, Any]]
    faithfulness_score: float | None = None
    context_precision_score: float | None = None
    context_recall_score: float | None = None
    answer_relevancy_score: float | None = None
    judge_verdict: str | None = None
    processing_time_ms: int
    shadow_timestamp: str
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float


# FastAPI app
app = FastAPI(
    title="Shadow Evaluation Server",
    description="Live shadow mode for safe production testing",
    version="0.1.0"
)

# Global state
_results_dir: Path = Path("results/shadow")
_results_dir.mkdir(parents=True, exist_ok=True)

_start_time = datetime.now()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    uptime = (datetime.now() - _start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=uptime
    )


@app.post("/evaluate", response_model=ShadowResponse)
async def evaluate_shadow(request: ShadowRequest) -> ShadowResponse:
    """
    Evaluate a query through the shadow variant.

    This endpoint processes the request through the shadow RAG system
    and returns results. It's called by the production API's shadow middleware.

    Processing is async and does not block production requests.
    """
    start_time = datetime.now()

    try:
        # Run shadow evaluation
        result = query(
            question=request.question,
            corpus_dir=Path(request.corpus_dir),
            top_k=request.top_k,
            phoenix_trace_id=f"shadow_{request.query_id}"
        )

        # Calculate processing time
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Extract metrics
        metrics = result.get("metrics", {})

        response = ShadowResponse(
            query_id=request.query_id,
            question=request.question,
            generated_answer=result.get("answer", ""),
            retrieved_chunks=result.get("retrieved_chunks", []),
            faithfulness_score=metrics.get("faithfulness_score"),
            context_precision_score=metrics.get("context_precision_score"),
            context_recall_score=metrics.get("context_recall_score"),
            answer_relevancy_score=metrics.get("answer_relevancy_score"),
            judge_verdict=metrics.get("judge_verdict"),
            processing_time_ms=processing_time_ms,
            shadow_timestamp=datetime.now().isoformat()
        )

        # Store result for comparison
        _store_result(response)

        return response

    except Exception as e:
        # Log error but don't fail (shadow failures shouldn't block)
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        response = ShadowResponse(
            query_id=request.query_id,
            question=request.question,
            generated_answer="",
            retrieved_chunks=[],
            processing_time_ms=processing_time_ms,
            shadow_timestamp=datetime.now().isoformat(),
            error=str(e)
        )

        _store_result(response)

        # Still return 200 (shadow failure is not a failure)
        return response


def _store_result(response: ShadowResponse) -> None:
    """Store shadow result for later comparison."""
    # Create file per query_id (append if exists)
    result_file = _results_dir / f"shadow_{datetime.now().strftime('%Y%m%d')}.jsonl"

    with open(result_file, "a") as f:
        f.write(json.dumps(response.model_dump()) + "\n")


@app.get("/results/{date}")
async def get_results(date: str) -> dict[str, Any]:
    """Get all shadow results for a date."""
    result_file = _results_dir / f"shadow_{date}.jsonl"

    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"No results for date {date}")

    results = []
    with open(result_file) as f:
        for line in f:
            results.append(json.loads(line))

    return {"date": date, "count": len(results), "results": results}


def main(
    host: str = "0.0.0.0",
    port: int = 8001,
    workers: int = 1
) -> None:
    """
    Start the shadow server.

    Args:
        host: Bind address
        port: Bind port
        workers: Number of worker processes
    """
    uvicorn.run(
        "eval_harness.shadow.server:app",
        host=host,
        port=port,
        workers=workers
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shadow evaluation server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes")

    args = parser.parse_args()

    main(host=args.host, port=args.port, workers=args.workers)
```

**Create production middleware** (`src/eval_harness/shadow/middleware.py`):

```python
"""Production API middleware for shadow mode.

This middleware mirrors production requests to the shadow server.
"""
import asyncio
from typing import Any
import httpx

SHADOW_SERVER_URL = "http://localhost:8001"


class ShadowMiddleware:
    """Middleware to mirror requests to shadow server."""

    def __init__(self, shadow_url: str = SHADOW_SERVER_URL):
        self.shadow_url = shadow_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def mirror_to_shadow(
        self,
        query_id: str,
        question: str,
        corpus_dir: str,
        top_k: int,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Mirror request to shadow server (non-blocking).

        Args:
            query_id: Unique query identifier
            question: User question
            corpus_dir: Path to document corpus
            top_k: Number of chunks to retrieve
            metadata: Additional metadata

        Note:
            This runs in background and does not block the production response.
            Shadow failures are logged but do not affect production.
        """
        request_data = {
            "query_id": query_id,
            "question": question,
            "corpus_dir": corpus_dir,
            "top_k": top_k,
            "metadata": metadata or {}
        }

        try:
            # Async call to shadow (don't await)
            asyncio.create_task(self._send_to_shadow(request_data))
        except Exception as e:
            # Log but don't fail
            print(f"[WARN] Failed to queue shadow request: {e}")

    async def _send_to_shadow(self, request_data: dict[str, Any]) -> None:
        """Send request to shadow server."""
        try:
            response = await self.client.post(
                f"{self.shadow_url}/evaluate",
                json=request_data
            )
            response.raise_for_status()
        except Exception as e:
            # Shadow failure is not a production failure
            print(f"[WARN] Shadow evaluation failed: {e}")
```

**Add to `pyproject.toml`**:

```toml
[project.optional-dependencies]
shadow = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "httpx>=0.24.0",
]

[project.scripts]
eval-shadow-server = "eval_harness.shadow.server:main"
```

**Usage**:

```bash
# Terminal 1: Start shadow server
uv run eval-shadow-server --port 8001

# Terminal 2: Production API with shadow middleware
# (pseudo-code for production API)
from eval_harness.shadow.middleware import ShadowMiddleware

shadow = ShadowMiddleware()

@app.post("/query")
async def handle_query(request: QueryRequest):
    query_id = str(uuid.uuid4())

    # 1. Process production request
    production_result = await process_query(request)

    # 2. Mirror to shadow (non-blocking)
    await shadow.mirror_to_shadow(
        query_id=query_id,
        question=request.question,
        corpus_dir="/data/corpus",
        top_k=5
    )

    # 3. Return production result (immediately)
    return production_result
```

### Safety Features

1. **Async processing**: Shadow requests don't block production
2. **Timeout enforcement**: Shadow requests timeout after 30s
3. **Graceful degradation**: Shadow failures don't affect production
4. **Separate storage**: Shadow results stored separately
5. **Health checks**: `/health` endpoint for monitoring

### Definition of Done

- [ ] Shadow server accepts evaluation requests
- [ ] Async request forwarding (non-blocking)
- [ ] Query ID matching working
- [ ] Shadow failure doesn't affect production
- [ ] Results stored in JSONL format
- [ ] CLI command `eval-shadow-server` working
- [ ] Health check endpoint
- [ ] Tested with mock production traffic

### What a adversarial Reviewer Will Ask

**Q**: "What if shadow server is down?"

**A**: Production middleware logs error and continues. Shadow is optional.

**Q**: "What if shadow is slower than production?"

**A**: Async. Production returns immediately. Shadow runs in background.

**Q**: "What about data consistency? What if corpus changes between production and shadow?"

**A**: Same corpus_dir. Shadow queries same data. If data changes, that's what you're testing (production vs shadow on current data).

**Q**: "Why FastAPI instead of Flask?"

**A**: Async support. Flask is sync. Need async to avoid blocking production.

---

## PBI-46: Add Automatic Result Sync

**Priority**: P2 (Operations)
**Estimate**: 4 hours
**Category**: Shadow Evaluation → Comparison

### Problem Statement

Shadow and production results are stored separately. No automatic comparison.

**Evidence**:

Current storage:
```
results/
├── production_20260521.json
└── shadow_20260521.jsonl  ← Different format!
```

**Problem**:
- Different formats (JSON vs JSONL)
- Different timestamps (production at time T, shadow at T+1)
- No automatic matching by query_id
- Manual comparison required

**adversarial reviewer says**: "I ran shadow for a day. Now tell me: did shadow beat production? Don't make me write Python to find out."

### Solution

Add `eval-shadow-compare` CLI command to automatically match and compare results.

### Acceptance Criteria

1. [ ] `eval-shadow-compare` CLI command
2. [ ] Load production and shadow results
3. [ ] Match by query_id
4. [ ] Compare metrics (faithfulness, etc.)
5. [ ] Generate comparison report
6. [ ] Statistical testing (is shadow significantly better?)
7. [ ] Handle missing queries

### Implementation

**Create `src/eval_harness/shadow/compare.py`**:

```python
"""Compare production and shadow evaluation results."""
import json
from pathlib import Path
from typing import Any
import pandas as pd

from scipy import stats


def load_shadow_results(results_path: Path) -> dict[str, dict[str, Any]]:
    """Load shadow results from JSONL file."""
    results = {}

    with open(results_path) as f:
        for line in f:
            data = json.loads(line)
            query_id = data["query_id"]
            results[query_id] = data

    return results


def load_production_results(results_path: Path) -> dict[str, dict[str, Any]]:
    """Load production results from CSV."""
    df = pd.read_csv(results_path)

    results = {}
    for _, row in df.iterrows():
        query_id = str(row["query_id"])
        results[query_id] = row.to_dict()

    return results


def compare_results(
    production_path: Path,
    shadow_path: Path,
    output_path: Path | None = None
) -> dict[str, Any]:
    """
    Compare production and shadow results.

    Args:
        production_path: Path to production CSV results
        shadow_path: Path to shadow JSONL results
        output_path: Optional path to write comparison report

    Returns:
        Comparison results dict
    """
    # Load results
    prod_results = load_production_results(production_path)
    shadow_results = load_shadow_results(shadow_path)

    # Find common query_ids
    prod_ids = set(prod_results.keys())
    shadow_ids = set(shadow_results.keys())

    common_ids = prod_ids & shadow_ids
    only_prod = prod_ids - shadow_ids
    only_shadow = shadow_ids - prod_ids

    # Collect metrics for comparison
    metrics = ["faithfulness_score", "context_precision_score",
               "context_recall_score", "answer_relevancy_score"]

    comparisons = {}

    for metric in metrics:
        prod_values = []
        shadow_values = []

        for query_id in common_ids:
            prod_val = prod_results[query_id].get(metric)
            shadow_val = shadow_results[query_id].get(metric)

            if pd.notna(prod_val) and pd.notna(shadow_val):
                prod_values.append(prod_val)
                shadow_values.append(shadow_val)

        if len(prod_values) == 0:
            continue

        # Calculate statistics
        prod_mean = sum(prod_values) / len(prod_values)
        shadow_mean = sum(shadow_values) / len(shadow_values)

        # Paired t-test (same queries)
        t_stat, p_value = stats.ttest_rel(shadow_values, prod_values)

        # Is shadow better?
        is_better = shadow_mean > prod_mean and p_value < 0.05

        comparisons[metric] = {
            "production_mean": prod_mean,
            "shadow_mean": shadow_mean,
            "difference": shadow_mean - prod_mean,
            "relative_change": (shadow_mean - prod_mean) / prod_mean if prod_mean != 0 else 0,
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "is_better": is_better,
            "n_compared": len(prod_values)
        }

    # Build result
    result = {
        "common_queries": len(common_ids),
        "production_only": len(only_prod),
        "shadow_only": len(only_shadow),
        "metrics": comparisons
    }

    # Print summary
    _print_summary(result)

    # Write report if requested
    if output_path:
        _write_report(result, output_path)

    return result


def _print_summary(result: dict[str, Any]) -> None:
    """Print comparison summary."""
    print("\n" + "="*60)
    print("SHADOW vs PRODUCTION COMPARISON")
    print("="*60 + "\n")

    print(f"Common queries: {result['common_queries']}")
    print(f"Production only: {result['production_only']}")
    print(f"Shadow only: {result['shadow_only']}")
    print()

    if not result["metrics"]:
        print("No metrics to compare.")
        return

    print("Metric Comparison:")
    print("-" * 60)

    for metric, comp in result["metrics"].items():
        prod_mean = comp["production_mean"]
        shadow_mean = comp["shadow_mean"]
        diff = comp["difference"]
        rel_change = comp["relative_change"] * 100
        p_val = comp["p_value"]
        is_sig = comp["is_significant"]
        is_better = comp["is_better"]

        status = "✓" if is_better else ("✗" if shadow_mean < prod_mean else "=")
        sig_mark = "*" if is_sig else ""

        print(f"\n{metric}:")
        print(f"  Production: {prod_mean:.4f}")
        print(f"  Shadow:     {shadow_mean:.4f}")
        print(f"  Difference: {diff:+.4f} ({rel_change:+.1f}%)")
        print(f"  P-value:    {p_val:.4f}{sig_mark}")
        print(f"  Status:     {status} {'Significant' if is_sig else 'Not significant'}")

    print("\n" + "="*60)


def _write_report(result: dict[str, Any], output_path: Path) -> None:
    """Write comparison report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[INFO] Comparison report: {output_path}")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare production and shadow evaluation results"
    )
    parser.add_argument(
        "production",
        type=Path,
        help="Path to production CSV results"
    )
    parser.add_argument(
        "shadow",
        type=Path,
        help="Path to shadow JSONL results"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path for comparison report (JSON)"
    )

    args = parser.parse_args()

    compare_results(args.production, args.shadow, args.output)


if __name__ == "__main__":
    main()
```

**Add to `pyproject.toml`**:

```toml
[project.scripts]
eval-shadow-compare = "eval_harness.shadow.compare:main"
```

**Usage**:

```bash
# Compare results
uv run eval-shadow-compare \
    results/production_20260521.csv \
    results/shadow/shadow_20260521.jsonl

# Output:
# ========================================================================
# SHADOW vs PRODUCTION COMPARISON
# ========================================================================
#
# Common queries: 100
# Production only: 5
# Shadow only: 2
#
# Metric Comparison:
# ------------------------------------------------------------
#
# faithfulness_score:
#   Production: 0.7500
#   Shadow:     0.8200
#   Difference: +0.0700 (+9.3%)
#   P-value:    0.0023*
#   Status:     ✓ Significant
#
# context_precision_score:
#   Production: 0.2500
#   Shadow:     0.2800
#   Difference: +0.0300 (+12.0%)
#   P-value:    0.1234
#   Status:     = Not significant
```

### Definition of Done

- [ ] `eval-shadow-compare` CLI working
- [ ] Loads production (CSV) and shadow (JSONL) results
- [ ] Matches by query_id
- [ ] Compares all metrics
- [ ] Statistical testing (paired t-test)
- [ ] Report generation (JSON + stdout)
- [ ] Handles missing queries gracefully

### What a adversarial Reviewer Will Ask

**Q**: "Why paired t-test instead of Wilcoxon?"

**A**: Paired t-test is standard for matched samples (same query_id). Wilcoxon is non-parametric alternative. Use Wilcoxon if data isn't normal. Add as option if needed.

**Q**: "What if production has 100 queries but shadow only 50?"

**A**: Compare only the 50 common. Report counts separately.

**Q**: "Why different formats (CSV vs JSONL)?"

**A**: Production uses existing CSV format. Shadow uses JSONL for streaming (write as results come in). Unify formats in future PBI.

**Q**: "Why p < 0.05 threshold?"

**A**: Standard for statistical significance. Make configurable if needed.

---

## Summary Table

| PBI | Priority | Estimate | Category | Risk if Deferred |
|-----|----------|----------|----------|------------------|
| Build live shadow mode | P2 | 16h | Shadow → Infrastructure | Can't test on production traffic safely |
| Add automatic result sync | P2 | 4h | Shadow → Comparison | Manual comparison only |

**Total P2**: 20 hours

---

## Dependencies

```
PBI-45 (Shadow server) ──┐
                         ├──→ PBI-45 first, then PBI-46
PBI-46 (Result sync)   ──┘
```

**Sequence**:
1. PBI-45: Build shadow server (16h)
2. PBI-46: Add comparison tool (4h)

---

## Implementation Sequence

**Week 3-4 (P2)**:
1. PBI-45: Build shadow server (2 days)
2. PBI-46: Add result sync (0.5 day)

---

## Operational Notes

### Deployment

```
Production Server:
- Main API on port 8000
- Shadow middleware installed
- Mirrors to shadow server

Shadow Server:
- Separate host (or container)
- Port 8001
- Not exposed to public internet
- Internal network only

Monitoring:
- Health checks: `/health` endpoint
- Metrics: shadow request rate, error rate
- Logs: shadow processing time
```

### Safety

1. **Isolation**: Shadow on separate host/container
2. **No user impact**: Shadow failures don't affect production
3. **Async**: Shadow processing doesn't block production
4. **Timeouts**: Shadow requests timeout after 30s
5. **Rate limiting**: Optional rate limiter on shadow server

---

**Document version**: 1.0
**Last updated**: 2026-05-22
**For**: Team planning meeting
