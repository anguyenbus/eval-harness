# Shadow Evaluation: Rewritten PBIs

**Status**: Istio verified in production. Rewritten for actual stack.

**Key changes from original**:
- Istio VirtualService.mirror instead of FastAPI server
- Wilcoxon signed-rank default, not t-test
- Replay-first architecture, live mirroring as enhancement
- Compliance-grade S3 storage included

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Istio Ingress Gateway                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ VirtualService.mirror
                            │
           ┌────────────────┴────────────────┐
           │                                 │
           ▼                                 ▼
┌──────────────────┐              ┌──────────────────┐
│  Production (v1) │              │  Shadow (v2)     │
│  100% traffic    │              │  Mirrored copy   │
└────────┬─────────┘              └────────┬─────────┘
         │                                 │
         │ 1. Response to user             │ 2. Process + store
         ▼                                 ▼
    ┌─────────┐                      ┌──────────────┐
    │  User   │                      │   S3 Bucket  │
    └─────────┘                      │  (eval data) │
                                     └──────────────┘
```

**Storage tiers**:
- Hot: S3 Standard (30 days)
- Warm: S3 Standard-IA (30-180 days)
- Cold: Glacier Flexible Retrieval (180 days - 7 years)
- SSE-KMS encryption, Object Lock Compliance mode

---

## Phase 1: Foundation

### PBI-A: Compliance-Grade Replay Payload Capture

**Priority**: P0 (Blocking)
**Estimate**: 1-2 weeks engineering + 2-4 weeks compliance review
**Risk**: Security/legal signoff required

#### Problem

Production emits Phoenix traces but adapter-boundary payloads not captured for replay.

#### Acceptance Criteria

1. [ ] Payload schema with `payload_schema_version` field
2. [ ] Schema validation at write time
3. [ ] Dead-letter routing for invalid payloads
4. [ ] S3 bucket in separate compliance account
5. [ ] SSE-KMS with customer-managed key
6. [ ] Object Lock Compliance mode (7-year retention)
7. [ ] Lifecycle policy (Standard → IA → Glacier)
8. [ ] CloudTrail + S3 access logs to audit bucket
9. [ ] Cross-account read role for eval-harness
10. [ ] Staged rollout (10% → 50% → 100%)

#### Implementation

**Payload schema** (`payload_schema_v1`):
```json
{
  "schema_version": "v1",
  "query_id": "uuid",
  "timestamp": "ISO8601",
  "mode": "rfi|chat",
  "input": {
    "question": "string",
    "case_id": "string",
    "document_refs": ["s3://..."]
  },
  "output": {
    "answer": "string",
    "retrieved_chunks": [{"chunk_id", "score", "content"}],
    "metrics": {...}
  },
  "metadata": {
    "adapter_version": "git-sha",
    "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
  }
}
```

**S3 structure**:
```
s3://eval-replay-payloads-prod/
  ├── hot/           (days 0-30, S3 Standard)
  ├── warm/          (days 30-180, Standard-IA)
  └── cold/          (days 180+, Glacier)
```

**Terraform for bucket**:
```hcl
resource "aws_s3_bucket" "replay_payloads" {
  bucket = "eval-replay-payloads-prod"

  lifecycle_rule {
    id      = "hot-to-warm"
    enabled = true

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 180
      storage_class = "GLACIER"
    }

    expiration {
      days = 2555  # 7 years
    }
  }
}

resource "aws_s3_bucket_versioning" "replay_payloads" {
  bucket = aws_s3_bucket.replay_payloads.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_object_lock_configuration" "replay_payloads" {
  bucket = aws_s3_bucket.replay_payloads.id

  object_lock_enabled = "Enabled"

  rule {
    default_retention {
      mode  = "COMPLIANCE"
      years = 7
    }
  }
}
```

#### Definition of Done

- [ ] Schema validated against real payloads
- [ ] Bucket created with compliance mode
- [ ] Cross-account IAM role working
- [ ] Dead-letter routing tested
- [ ] Security review approved
- [ ] Legal retention confirmed

---

## Phase 2: Core Evaluation

### PBI-B: Production Replay Harness

**Priority**: P1
**Estimate**: 4-5 days

#### Problem

No CLI to replay captured payloads against candidate adapter.

#### Acceptance Criteria

1. [ ] `eval-replay` CLI command
2. [ ] Accepts `--adapter` (candidate image) and `--since` (time range)
3. [ ] Reads from hot/warm S3 tier
4. [ ] Runs baseline + candidate on same payloads
5. [ ] Writes results to S3 in stable schema
6. [ ] Controls for judge drift (baseline re-run)

#### Implementation

```python
# src/eval_harness/runners/run_replay.py

@click.command()
@click.argument("candidate_adapter", type=str)
@click.option("--since", type=str, default="7d", help="Time range: 7d, 30d, 90d")
@click.option("--sample", type=int, default=100, help="Max payloads to replay")
@click.option("--output-bucket", type=str, default="s3://eval-results/")
def main(candidate_adapter: str, since: str, sample: int, output_bucket: str):
    """Replay captured production payloads against candidate adapter.

    Runs both baseline and candidate on same payloads to control for judge drift.
    """
    # 1. Load payloads from S3
    payloads = load_payloads_since(since, limit=sample)

    # 2. Run baseline
    baseline_results = run_eval(payloads, adapter="baseline")

    # 3. Run candidate
    candidate_results = run_eval(payloads, adapter=candidate_adapter)

    # 4. Store paired results
    write_results(output_bucket, {
        "baseline": baseline_results,
        "candidate": candidate_results,
        "paired": pair_by_query_id(baseline_results, candidate_results)
    })
```

#### Definition of Done

- [ ] `eval-replay` command working
- [ ] Reads from S3 hot/warm tiers
- [ ] Baseline + candidate execution
- [ ] Paired results output

---

### PBI-C: Public Benchmark CI Runner

**Priority**: P1
**Estimate**: 2-3 days

#### Problem

Public benchmarks integrated but not CI-callable.

#### Acceptance Criteria

1. [ ] `eval-benchmark` CLI with `--adapter` flag
2. [ ] Fast-subset mode for PR validation
3. [ ] Full mode for nightly baseline
4. [ ] Output to S3 in comparison-ready schema

#### Definition of Done

- [ ] OmniDocBench, DP-Bench, LegalBench-RAG working
- [ ] Fast subset (< 5 min) for PRs
- [ ] Full results to S3

---

### PBI-D: Differential Diff Utility

**Priority**: P2
**Estimate**: 2-3 days

#### Problem

No tool to compare parse/retrieval outputs on same input.

#### Acceptance Criteria

1. [ ] Parse diff: token-level, TEDS for tables
2. [ ] Retrieval diff: Jaccard overlap, Kendall tau
3. [ ] Flag unexplained changes
4. [ ] Markdown report output

#### Definition of Done

- [ ] `eval-diff` command
- [ ] Parse + retrieval comparison
- [ ] Change flagging

---

## Phase 3: Operationalize

### PBI-E: Statistical Comparison Layer

**Priority**: P1
**Estimate**: 3-4 days

#### Problem

No paired comparison, no significance testing, no variance calibration.

#### Acceptance Criteria

1. [ ] **Wilcoxon signed-rank default** for continuous metrics
2. [ ] McNemar's test for binary (when RFI corpus exists)
3. [ ] Effect size (Cliff's δ) alongside p-values
4. [ ] Variance calibration from repeated runs
5. [ ] Composite pass/fail rule
6. [ ] Single boolean output

#### Implementation

```python
# src/eval_harness/reporting/statistics.py

def compare_metrics(
    baseline: List[float],
    candidate: List[float],
    metric_type: str = "continuous"
) -> ComparisonResult:
    """Compare metrics with appropriate statistical test.

    Defaults to Wilcoxon for continuous (LLM judge scores).
    Uses McNemar for binary classification.
    """
    if metric_type == "continuous":
        # Wilcoxon signed-rank (non-parametric, paired)
        stat, p_value = scipy.stats.wilcoxon(baseline, candidate)
        effect_size = cliffs_delta(baseline, candidate)
    else:
        # McNemar for binary
        stat, p_value = mcnemar_test(baseline, candidate)
        effect_size = odds_ratio(baseline, candidate)

    return ComparisonResult(
        test="wilcoxon" if metric_type == "continuous" else "mcnemar",
        p_value=p_value,
        effect_size=effect_size,
        significant=p_value < 0.05,
        better=effect_size > 0  # positive = candidate better
    )
```

#### Definition of Done

- [ ] Wilcoxon default for judge scores
- [ ] Effect size reported
- [ ] Variance calibration implemented
- [ ] Composite rule defined

---

### PBI-F: CI Integration with Eval Gates

**Priority**: P1
**Estimate**: 2-3 days

#### Problem

No blocking on eval regressions.

#### Acceptance Criteria

1. [ ] GitHub Actions workflow on PR to parser/retriever/chat
2. [ ] Build + push candidate container to ECR
3. [ ] Submit eval Jobs to EKS in parallel
4. [ ] Fetch comparison report from PBI-E
5. [ ] Post summary to PR as comment
6. [ ] Block merge on regression
7. [ ] Override mechanism (lead approval)

#### Implementation

```yaml
# .github/workflows/eval-gate.yml

name: Eval Gate

on:
  pull_request:
    paths:
      - 'src/parsers/**'
      - 'src/retrieval/**'
      - 'src/chat/**'

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build candidate image
        run: |
          docker build -t eval-candidate:${{ github.sha }} .
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO
          docker push $ECR_REPO/eval-candidate:${{ github.sha }}

      - name: Submit eval jobs
        run: |
          kubectl config use-context eval-harness
          kubectl apply -f - <<EOF
          apiVersion: batch/v1
          kind: Job
          metadata:
            name: eval-pr-${{ github.number }}
          spec:
            template:
              spec:
                containers:
                - name: eval
                  image: $ECR_REPO/eval-candidate:${{ github.sha }}
                  command: ["eval-replay", "--since", "7d"]
          EOF

      - name: Wait for results
        run: ./scripts/wait-for-eval.sh eval-pr-${{ github.number }}

      - name: Post to PR
        if: always()
        run: |
          RESULTS=$(aws s3 cp s3://eval-results/pr-${{ github.number }}/report.json -)
          ./scripts/post-pr-comment.sh "$RESULTS"
```

#### Definition of Done

- [ ] CI workflow triggered on code paths
- [ ] Eval jobs submitted to EKS
- [ ] Comparison report posted to PR
- [ ] Merge blocked on regression
- [ ] Override mechanism working

---

## Phase 4: Istio Live Mirroring

### PBI-G: Istio VirtualService Mirror Configuration

**Priority**: P2
**Estimate**: 1-2 days

#### Problem

Replay is primary. Live mirroring via Istio for specific use cases.

#### Use Cases

- Pre-promotion validation when traffic distribution shifted faster than replay corpus
- Testing against current traffic for candidates deployed later

#### Acceptance Criteria

1. [ ] Istio VirtualService with `mirror` field
2. [ ] Shadow deployment labeled `version: shadow`
3. [ ] Mirror percentage configurable (0-100%)
4. [ ] Shadow results written to eval bucket
5. [ ] No impact on production response

#### Implementation

```yaml
# istio/virtualservice-mirror.yaml

apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: case-assistant-mirror
spec:
  hosts:
  - case-assistant.prod.svc.cluster.local
  http:
  - route:
    - destination:
        host: case-assistant
        subset: production
      weight: 100
    mirror:
      host: case-assistant
      subset: shadow
    mirrorPercentage:
      value: 10.0  # Configurable
```

```yaml
# istio/destinationrule-shadow.yaml

apiVersion: networking.istio.io/v1
kind: DestinationRule
metadata:
  name: case-assistant-shadow
spec:
  host: case-assistant
  subsets:
  - name: production
    labels:
      version: v1
  - name: shadow
    labels:
      version: shadow
```

#### Shadow Service Modifications

```python
# Shadow service writes to eval bucket (not local disk)

@app.post("/query")
async def handle_shadow_query(request: QueryRequest):
    result = await process_query(request)

    # Write to S3 instead of local file
    s3_client.put_object(
        Bucket="eval-replay-payloads-prod",
        Key=f"shadow/{date}/{request.query_id}.json",
        Body=json.dumps(result),
        ObjectLockMode="COMPLIANCE",
        ObjectLockRetainUntilDate=retention_date
    )

    return result
```

#### Definition of Done

- [ ] VirtualService.mirror configured
- [ ] Shadow subset labeled
- [ ] Mirror writes to S3 eval bucket
- [ ] Production response unaffected
- [ ] Mirror percentage adjustable

---

## Dependencies

```
PBI-A (Storage) ────────────────────────────────┐
                                                   ├──→ PBI-E (Stats) ──→ PBI-F (CI)
PBI-B (Replay) ────┐                              │
                   ├──→ PBI-D (Diff) ──────────────┘
PBI-C (Benchmarks)─┘                              │
                                                   │
PBI-G (Istio Mirror) ──────────────────────────────┘ (Enhancement)
```

**Critical path**: A → B/C/D → E → F

**Parallelizable**: B, C, D (after A)
**Deferred**: G (Phase 4)

---

## Timeline

| Phase | PBIs | Estimate | Calendar |
|-------|------|----------|----------|
| 1: Foundation | A | 1-2w eng + 2-4w compliance | ~6 weeks total |
| 2: Core eval | B, C, D | 8-11 days | 2 weeks |
| 3: Operationalize | E, F | 5-7 days | 1-2 weeks |
| 4: Istio mirror | G | 1-2 days | Deferred |

**Total active work**: ~4-6 weeks engineering + compliance dependency

---

## Changes from Original

| Aspect | Original | Rewritten |
|--------|----------|-----------|
| Shadow server | FastAPI on port 8001 | Istio VirtualService.mirror |
| Statistical test | Paired t-test default | Wilcoxon signed-rank default |
| Storage | Local JSONL files | S3 with 7-year compliance retention |
| Primary mechanism | Live mirroring | Replay first, live enhancement |
| Estimate | 20h total | 4-6 weeks + compliance review |

---

## What Was Thrown Out

- FastAPI shadow server (replaced by Istio)
- `ShadowMiddleware` with `httpx.AsyncClient` (replaced by mesh-level mirroring)
- Local JSONL storage (replaced by S3 compliance archive)
- Paired t-test as default (replaced by Wilcoxon)
- 16h + 4h estimate (replaced by realistic timeline including compliance)

---

**Document version**: 2.0 (Rewritten)
**Last updated**: 2026-05-23
**For**: Implementation planning
