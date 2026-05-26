# Evaluation System Improvement Roadmap

**Source**: Hostile engineer review of verified walkthrough
**Date**: 2026-05-21
**Status**: Organized by capability category with priority levels

---

## Priority Legend

- **P0**: Correctness — blocks honest interpretation of results
- **P1**: Rigor — needed for scientific/statistical validity
- **P2**: Trust — needed for external stakeholders

---

## 1. Load Evaluation

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P0 | Fix error-as-zero bug | Judge failures default to 0.0. Use NaN or fail-loud instead. Silent zeros corrupt averages. | 2h |
| P1 | Run nano 5x for variance | Execute nano slice 5 times, report stdev per metric. Calibrates regression threshold. | 30min |
| P1 | Add concurrency control | Current sequential execution. Add parallel query option with semaphore control. | 4h |
| P2 | Add resource monitoring | Track CPU/memory during load evaluation. | 4h |

**Why P0 matters**: If 3 of 10 queries fail silently with score=0.0, your "average 0.75" is actually 3 failures + 7 good runs. That's misleading.

---

## 2. Call Orchestration

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P0 | Add retry logic | LLM calls fail transiently. Implement exponential backoff retry. | 2h |
| P0 | Add timeout enforcement | LLM calls can hang indefinitely. Add per-metric timeout. | 1h |
| P1 | Track token usage | Wrap LLM calls to count input/output tokens. | 2h |
| P2 | Add resume capability | Checkpoint after each query. Resume from checkpoint on restart. | 6h |

---

## 3. Offline Evaluation

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P0 | Pin model snapshots | Judge uses `gpt-4o` alias. Pin to `gpt-4o-2024-08-06` or similar dated snapshot. | 1h |
| P0 | Resolve config contradiction | Config shows `gpt-4o` but `deepeval_config.py` defaults to `gpt-4o-mini`. | 1h |
| P1 | Capture dataset SHA | Record HuggingFace dataset commit SHA in JSON metadata. | 2h |
| P1 | Capture environment metadata | Python version, OS, uv.lock hash in JSON. | 3h |
| P2 | Add streaming mode | Support incremental result publication before full completion. | 8h |

---

## 4. A/B Routing

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P1 | Add built-in comparison tool | Auto-compare two JSON summaries, highlight differences. | 4h |
| P1 | Add statistical testing | Fisher exact test for binary metrics, bootstrap for continuous. | 6h |
| P2 | Add side-by-side report | Generate combined A/B view in HTML. | 4h |

---

## 5. Shadow Evaluation

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P2 | Build live shadow mode | Server that mirrors production traffic to shadow variant. | 16h |
| P2 | Add automatic result sync | Match shadow/production results by query ID for comparison. | 4h |

---

## 6. Collect Metrics

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P0 | Build citation correctness metric | Verify cited chunk actually supports the claim. Critical for legal RAG. | 12h |
| P0 | Fix citation gap | Schema defines `citation_spans_support_claims` but it's unused. Implement it. | 8h |
| P1 | Add cost reporting | Track tokens in/out per metric. Add dollar estimate to JSON. | 3h |
| P1 | Add metric threshold enforcement | Config has thresholds but they're not enforced. Fail on threshold breach. | 2h |
| P2 | Add custom metric registration | Allow user-defined metrics without code changes. | 8h |
| P2 | Add metric caching | Cache LLM-judge results to avoid recomputation. | 4h |

**Why citation correctness is P0 for legal RAG**: A lawyer cannot use a system that cites the wrong statute. Extraction + validation ≠ correctness.

---

## 7. Run Evaluation Framework

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P0 | Fix regression_check.py | Expects `metrics` with `severity`, JSON uses `metrics_avg`. Make it work. | 2h |
| P0 | Add dry-run mode | Validate config and dataset without running full evaluation. | 2h |
| P1 | Add variance reporting | Run nano slice 5x, report stdev per metric. | 30min |
| P2 | Add progress bar | Show query progress during long runs. | 1h |

---

## 8. Metrics Retrieval

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P0 | Fix html_summary.py | Expects `label`/`score` columns, CSV has different names. Fix it. | 2h |
| P1 | Add confidence intervals | Report CI alongside point estimates for small samples. | 4h |
| P1 | Add details.json sample | Document real example (Bob & Ted juror) with full reasoning. | 1h |
| P2 | Add time-series tracking | Track metric trends over time in a database. | 12h |
| P2 | Add visualization | Generate charts from results without external tools. | 8h |

**Why regression check is P0**: A broken regression detector gives false confidence. "We have regression detection" when you don't is worse than admitting you don't have it.

---

## 9. Citation Check

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P0 | Implement citation evaluation | Move from extraction to correctness checking. | 12h |
| P0 | Use schema citation question | Implement `citation_spans_support_claims` from eval_questions schema. | 8h |
| P2 | Support more citation formats | Beyond `[chunk_id]` pattern matching. | 6h |

---

## 10. Store Results

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P2 | Add database option | Store results in Postgres/SQLite instead of just flat files. | 12h |
| P2 | Add deduplication | Don't create new files for identical runs. | 4h |
| P2 | Add retention policy | Auto-clean old result files. | 2h |
| P2 | Document S3 security model | IAM roles, bucket policy, retention, failure handling. | 2h |

---

## 11. Comparison

| Priority | Task | Description | Est. Effort |
|----------|------|-------------|-------------|
| P0 | Fix regression_check.py format | Make it work with current `metrics_avg` JSON structure. | 2h |
| P1 | Calibrate threshold to variance | Use judge variance from 5x nano runs instead of magic 5%. | 2h |
| P1 | Add automated comparison tool | Compare any two JSON summaries with diff view. | 4h |
| P2 | Add statistical tests | Fisher exact, bootstrap, Mann-Whitney U. | 6h |
| P2 | Add diff visualization | Side-by-side A/B HTML report. | 4h |

---

## Cross-Cutting Concerns

### Reproducibility Audit

| Element | Current State | Fix | Priority |
|---------|---------------|-----|----------|
| Judge model snapshot | Alias (`gpt-4o`) | Pin to dated snapshot | P0 |
| Embedding model version | Name only | Capture version hash | P1 |
| DeepEval version | ✅ Pinned in JSON | Maintain | — |
| Dataset version | Not captured | Record HF commit SHA | P1 |
| Python/OS versions | Not captured | Add to metadata | P1 |
| uv.lock | Not stated | Commit or reference | P1 |
| Random seeds | Not stated | Document seed behavior | P2 |
| Judge variance | Not measured | Run 5x nano slice | P1 |

### Human Calibration Loop

| Task | Description | Priority |
|------|-------------|----------|
| SME review sample | Have lawyer review 10% of judge verdicts | P2 |
| Calibrate to ground truth | Adjust prompts based on SME feedback | P2 |
| Document calibration process | Write how calibration was performed | P2 |

### Security & Operations

| Task | Description | Priority |
|------|-------------|----------|
| Document S3 model | IAM, bucket, retention, failure handling | P2 |
| Add secrets audit | Ensure no credentials in code | P0 |
| Add cost controls | Budget limits on API spend | P1 |

---

## Focused Sprint: One Week Priority

If I had to pick what to do this week:

**Monday-Tuesday (P0 Correctness)**
1. Fix error-as-zero bug (2h)
2. Pin judge model to snapshot (1h)
3. Fix regression_check.py format (2h)
4. Fix html_summary.py columns (2h)
5. Resolve config contradiction (1h)

**Wednesday-Thursday (P1 Rigor)**
1. Run nano 5x, report variance (30min)
2. Calibrate regression threshold to variance (2h)
3. Add cost reporting (3h)
4. Capture dataset SHA and environment metadata (5h)

**Friday (P2 Trust Foundation)**
1. Add details.json sample to docs (1h)
2. Document S3 security model (2h)
3. Plan citation correctness metric (design session)

**Total**: ~22 hours — achievable in one focused sprint.

---

## What a Hostile Reviewer Would Ask

After completing P0-P1, you should be able to answer:

| Question | Current Answer | Target Answer |
|----------|----------------|---------------|
| "What's the confidence interval on context_precision=0.1458?" | "I don't know" | "±0.08 (n=10, 5-run variance)" |
| "Are any scores silent failures?" | "Can't tell" | "No — failures are NaN and excluded" |
| "What gpt-4o snapshot did this use?" | "The alias" | "gpt-4o-2024-08-06 (pinned)" |
| "Has a lawyer reviewed these verdicts?" | "No" | "Yes, 10% sample, see CALIBRATION.md" |
| "Show me regression detection" | "It's broken" | "Here's the last alert (link)" |
| "What did this run cost?" | "Not tracked" | "$12.37 (4.2M tokens)" |
| "Can you reproduce last month's results?" | "Approximately" | "Exactly (SHA: abc123)" |

---

## Meta-Pattern

The walkthrough has moved from "marketing feature list" to "honest verified reference." Next step: "scientifically rigorous benchmark."

**For internal use**: Current state is fine.
**For external action** (vendor selection, publication, procurement): P0-P1 gaps must be addressed.

---

## Quick Reference: P0 Backlog

```
[P0] Fix error-as-zero bug (use NaN, not 0.0)
[P0] Pin judge model to dated snapshot
[P0] Fix regression_check.py JSON format
[P0] Fix html_summary.py column mismatch
[P0] Resolve gpt-4o vs gpt-4o-mini contradiction
[P0] Build citation correctness metric
[P0] Implement citation_spans_support_claims question
[P0] Add retry logic for LLM calls
[P0] Add timeout for LLM calls
```

**Estimated P0 total**: ~30 hours

---

**Document version**: 1.0
**Last updated**: 2026-05-21
**Maintainer**: eval-harness team
