# Collect Metrics & Citation Check: PBIs

**Capabilities**: Collect Metrics (metric computation) + Citation Check (citation correctness).

**Current State**: 9 parsing metrics + 4 DeepEval LLM-judge metrics. Citations extracted but not evaluated for correctness.

---

## PBI-20: Build Citation Correctness Metric

**Priority**: P0 (Legal RAG Critical)
**Estimate**: 12 hours
**Category**: Collect Metrics → Citations

### Problem

For legal RAG, citation correctness is non-negotiable. Current code:
- Extracts citations: ✅ ([`citations.py:15`](src/eval_harness/stubs/rag/citations.py:15))
- Validates chunk_id exists: ✅ ([`schema_conformance.py:106`](src/eval_harness/stubs/rag/schema_conformance.py:106))
- **Checks if cited chunk supports claim**: ❌ NOT IMPLEMENTED

**Why this matters**: A lawyer cannot trust a system that cites the wrong statute. Extraction ≠ Correctness.

### Acceptance Criteria

1. [ ] LLM-judge checks if cited chunk contains evidence for claim
2. [ ] Per-citation score (0-1) for evidence quality
3. [ ] Overall citation correctness score (average)
4. [ ] Reasoning for each citation judgment
5. [ ] Integrated into RAG evaluation pipeline

### Implementation Notes

**Create `src/eval_harness/metrics/citations/correctness.py`**:

```python
"""
Citation correctness metric.

Verifies that cited chunks actually support the claims they're attached to.
"""
from typing import Any, Dict, List
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CitationCorrectnessMetric(BaseMetric):
    """
    Measures whether citations actually support the claims.

    For each citation in the answer:
    1. Extract the claim (text span)
    2. Check if cited chunk contains supporting evidence
    3. Score 0-1 based on evidence quality
    """

    def __init__(self, model: Any = None, include_reasoning: bool = True):
        super().__init__()
        self.model = model
        self.include_reasoning = include_reasoning
        self.verdicts: List[Dict] = []

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure citation correctness.

        Returns:
            Overall score (0-1) averaging all citation judgments
        """
        from eval_harness.stubs.rag.citations import extract_citations

        # Extract citations from answer
        # Note: This requires the actual output to have citations
        # For now, we'll evaluate a passed-in structure
        citations = test_case.additional_metadata.get("citations", [])
        retrieved_chunks = test_case.retrieval_context or []

        if not citations:
            # No citations to evaluate
            return 0.0

        self.verdicts = []

        for citation in citations:
            verdict = self._evaluate_citation(citation, retrieved_chunks)
            self.verdicts.append(verdict)

        # Average all citation scores
        scores = [v["score"] for v in self.verdicts]
        self.score = sum(scores) / len(scores) if scores else 0.0

        return self.score

    def _evaluate_citation(
        self,
        citation: Dict[str, Any],
        retrieved_chunks: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a single citation.

        Args:
            citation: Dict with claim_span, chunk_ids
            retrieved_chunks: List of retrieved chunk texts

        Returns:
            Verdict dict with score, reason, evidence_quality
        """
        # Get the claim text
        # In practice, need to extract from answer using claim_span
        # For now, assume claim is passed separately
        claim = citation.get("claim", "")
        chunk_ids = citation.get("chunk_ids", [])

        # Get the cited chunks
        cited_chunks = [
            chunk for i, chunk in enumerate(retrieved_chunks)
            if f"chunk_{i:05d}" in chunk_ids or str(i) in chunk_ids
        ]

        if not cited_chunks:
            return {
                "score": 0.0,
                "reason": "Cited chunks not found in retrieval context",
                "evidence_quality": "missing"
            }

        # Use LLM to evaluate evidence quality
        evidence_quality = self._llm_evaluate_evidence(claim, cited_chunks)

        # Map evidence quality to score
        score_map = {
            "full_support": 1.0,
            "partial_support": 0.5,
            "no_support": 0.0,
            "contradiction": 0.0  # Cited chunk contradicts claim
        }

        score = score_map.get(evidence_quality, 0.0)

        return {
            "score": score,
            "reason": f"Claim '{claim[:50]}...': {evidence_quality}",
            "evidence_quality": evidence_quality,
            "cited_chunks": len(cited_chunks)
        }

    def _llm_evaluate_evidence(
        self,
        claim: str,
        cited_chunks: List[str]
    ) -> str:
        """
        Use LLM to evaluate if cited chunks support the claim.

        Returns:
            One of: full_support, partial_support, no_support, contradiction
        """
        # Build prompt
        chunks_text = "\n\n".join(
            f"[Chunk {i+1}] {chunk}"
            for i, chunk in enumerate(cited_chunks)
        )

        prompt = f"""You are evaluating whether cited sources support a legal claim.

Claim: {claim}

Cited sources:
{chunks_text}

Evaluate: Do these cited sources provide evidence for the claim?
Respond with EXACTLY ONE of:
- FULL_SUPPORT: Sources directly support the claim
- PARTIAL_SUPPORT: Sources partially support but have gaps
- NO_SUPPORT: Sources do not address the claim
- CONTRADICTION: Sources contradict the claim

Your response:"""

        # Call LLM
        if self.model:
            response = self.model.generate(prompt)
            result = response.strip().upper()
        else:
            # Fallback for testing
            result = "NO_SUPPORT"

        # Validate response
        valid = {"FULL_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT", "CONTRADICTION"}
        return result if result in valid else "NO_SUPPORT"

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)
```

**Integrate into DeepEval adapter**:

```python
# In create_deepeval_metrics()
metrics = {
    "faithfulness": ...,
    "context_precision": ...,
    "context_recall": ...,
    "answer_relevancy": ...,
    "citation_correctness": CitationCorrectnessMetric(
        model=llm,
        include_reasoning=True
    )
}
```

**Update RAG adapter to pass citations**:

```python
# In RagAdapter.query()
output = self._query(question, corpus_dir, embedder=self._embedder)

# Extract and validate citations
from eval_harness.stubs.rag.citations import extract_citations
citations = extract_citations(
    output.get("answer", {}).get("text", ""),
    output.get("retrieved_chunks", [])
)
output["answer"]["citations"] = citations
```

**JSON output format**:

```json
{
  "citation_correctness_score": 0.75,
  "citation_correctness_reasoning": {
    "reason": "3 of 4 citations fully support their claims",
    "verdicts": [
      {
        "claim": "The judge is required to excuse Bob",
        "score": 1.0,
        "evidence_quality": "full_support",
        "cited_chunks": 2
      },
      {
        "claim": "Jurors can be trusted to follow directions",
        "score": 0.5,
        "evidence_quality": "partial_support",
        "cited_chunks": 1
      }
    ]
  }
}
```

### Definition of Done

- [ ] LLM-based citation evaluation implemented
- [ ] Per-citation scoring working
- [ ] Overall correctness score computed
- [ ] Integrated into RAG pipeline
- [ ] Reasoning captured in details.json
- [ ] Tested with legal questions

---

## PBI-21: Implement Schema Citation Question

**Priority**: P0 (Legal RAG Critical)
**Estimate**: 8 hours
**Category**: Citation Check → Schema

### Problem

Schema defines `citation_spans_support_claims` evaluation question ([`contracts/eval_questions.schema.json`](contracts/eval_questions.schema.json)) but it's never used.

### Acceptance Criteria

1. [ ] Use schema-defined evaluation question for citations
2. [ ] Load question from schema at runtime
3. [ ] Pass to LLM judge with query context
4. [ ] Parse structured response from judge
5. [ ] Store verdict in evaluation results

### Implementation Notes

**Schema format** (from `eval_questions.schema.json`):

```json
{
  "evaluation_questions": [
    {
      "id": "citation_spans_support_claims",
      "question": "Do the cited passages support the claims in the answer?",
      "type": "llm_judge",
      "rubric": {
        "pass": "All citations directly support their associated claims",
        "fail": "One or more citations do not support their claims or contradict them"
      }
    }
  ]
}
```

**Create `src/eval_harness/metrics/citations/schema_question.py`**:

```python
"""
Schema-driven citation evaluation.

Uses eval_questions.schema.json to define and execute citation evaluation.
"""
import json
from pathlib import Path
from typing import Any, Dict

def load_evaluation_questions() -> Dict[str, Any]:
    """Load evaluation questions from schema."""
    schema_path = Path("contracts/eval_questions.schema.json")

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with open(schema_path) as f:
        schema = json.load(f)

    return schema.get("evaluation_questions", [])

def get_citation_question() -> Dict[str, Any]:
    """Get the citation evaluation question."""
    questions = load_evaluation_questions()

    for q in questions:
        if q.get("id") == "citation_spans_support_claims":
            return q

    raise ValueError("citation_spans_support_claims question not found in schema")

class SchemaCitationEvaluator:
    """
    Evaluates citations using schema-defined question.
    """

    def __init__(self, llm_model: Any):
        self.model = llm_model
        self.question_spec = get_citation_question()

    def evaluate(
        self,
        answer: str,
        citations: list[dict[str, Any]],
        retrieved_chunks: list[str]
    ) -> dict[str, Any]:
        """
        Evaluate citations using schema question.

        Returns:
            Verdict dict with score, reasoning, per-citation breakdown
        """
        # Build prompt from schema
        prompt = self._build_prompt(answer, citations, retrieved_chunks)

        # Get LLM judgment
        response = self.model.generate(prompt)

        # Parse response
        verdict = self._parse_response(response)

        return verdict

    def _build_prompt(
        self,
        answer: str,
        citations: list[dict[str, Any]],
        retrieved_chunks: list[str]
    ) -> str:
        """Build evaluation prompt from schema question."""

        # Extract claim texts
        claims = []
        for citation in citations:
            span = citation.get("claim_span", [0, 0])
            claim_text = answer[span[0]:span[1]]
            claims.append(claim_text)

        # Build chunk references
        chunk_texts = []
        for citation in citations:
            chunk_ids = citation.get("chunk_ids", [])
            for chunk_id in chunk_ids:
                # Find chunk by ID
                # ... extract chunk text
                chunk_texts.append(f"[{chunk_id}] {chunk_text}")

        # Use schema question template
        question_text = self.question_spec["question"]
        rubric = self.question_spec["rubric"]

        prompt = f"""{question_text}

Answer:
{answer}

Citations made:
{chr(10).join(f'- "{c}"' for c in claims)}

Cited passages:
{chr(10).join(chunk_texts)}

Evaluation criteria:
PASS: {rubric["pass"]}
FAIL: {rubric["fail"]}

Evaluate each citation and provide an overall verdict (PASS/FAIL)."""

        return prompt

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response into structured verdict."""
        # Parse for PASS/FAIL
        if "PASS" in response.upper():
            overall = "pass"
        elif "FAIL" in response.upper():
            overall = "fail"
        else:
            overall = "needs_review"

        # Try to extract per-citation judgments
        # This depends on LLM output format
        # ...

        return {
            "verdict": overall,
            "reasoning": response,
            "per_citation": {}  # Parsed if possible
        }
```

### Definition of Done

- [ ] Schema question loaded at runtime
- [ ] Prompt built from question template
- [ ] LLM judgment executed
- [ ] Response parsed into verdict
- [ ] Verdict stored in details.json
- [ ] Compatible with existing schema format

---

## PBI-22: Add Cost Reporting (Duplicate PBI-8 from Call Orchestration)

**Priority**: P1 (Cost Management)
**Estimate**: 3 hours
**Category**: Collect Metrics → Observability

*Note: This PBI is defined in `pbis-call-orchestration.md` (PBI-8). Implement once.*

**Key addition for citations**: Track cost per citation evaluation (additional LLM calls).

---

## PBI-23: Add Metric Threshold Enforcement

**Priority**: P1 (Quality Gates)
**Estimate**: 2 hours
**Category**: Collect Metrics → Enforcement

### Problem

Config defines metric thresholds ([`eval_config.yaml:37`](eval_config.yaml:37)):

```yaml
metrics:
  text_fidelity:
    threshold: 0.95
  table_teds:
    threshold: 0.85
```

But thresholds are never enforced. Results below threshold still pass.

### Acceptance Criteria

1. [ ] Load metric thresholds from config
2. [ ] Check each metric against threshold
3. [ ] Fail evaluation if threshold breached
4. [ ] Report which metrics failed
5. [ ] `--fail-on-threshold` flag (default: true)

### Implementation Notes

**Create `src/eval_harness/metrics/thresholds.py`**:

```python
"""
Metric threshold enforcement.
"""
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ThresholdCheck:
    """Result of threshold check."""
    metric_name: str
    threshold: float
    actual: float
    passed: bool
    severity: str  # "blocker", "warning"

def check_thresholds(
    metrics: Dict[str, float],
    config: Dict[str, Any]
) -> List[ThresholdCheck]:
    """
    Check metrics against configured thresholds.

    Returns:
        List of threshold check results
    """
    checks = []
    threshold_config = config.get("metrics", {})

    for metric_name, actual_value in metrics.items():
        if metric_name not in threshold_config:
            continue

        threshold_spec = threshold_config[metric_name]
        threshold_value = threshold_spec.get("threshold", 0.0)
        severity = threshold_spec.get("severity", "warning")

        passed = actual_value >= threshold_value

        checks.append(ThresholdCheck(
            metric_name=metric_name,
            threshold=threshold_value,
            actual=actual_value,
            passed=passed,
            severity=severity
        ))

    return checks

def enforce_thresholds(
    checks: List[ThresholdCheck],
    fail_on_breach: bool = True
) -> None:
    """
    Enforce threshold checks.

    Args:
        checks: List of threshold check results
        fail_on_breach: If True, raise exception on blocker failures

    Raises:
        RuntimeError: If any blocker threshold failed and fail_on_breach=True
    """
    failures = [c for c in checks if not c.passed and c.severity == "blocker"]
    warnings = [c for c in checks if not c.passed and c.severity == "warning"]

    if failures:
        msg = "Blocker threshold failures:\n"
        for f in failures:
            msg += f"  - {f.metric_name}: {f.actual:.4f} < {f.threshold:.4f}\n"

        print(f"[ERROR] {msg}", file=sys.stderr)

        if fail_on_breach:
            raise RuntimeError(msg)

    if warnings:
        msg = "Warning threshold failures:\n"
        for w in warnings:
            msg += f"  - {w.metric_name}: {w.actual:.4f} < {w.threshold:.4f}\n"
        print(f"[WARN] {msg}", file=sys.stderr)
```

**Integrate into runners**:

```python
# After computing metrics
threshold_checks = check_thresholds(metric_scores, config)

# Print summary
for check in threshold_checks:
    status = "✓" if check.passed else "✗"
    print(f"  {status} {check.metric_name}: {check.actual:.4f} (threshold: {check.threshold:.4f})")

# Enforce (or just warn)
enforce_thresholds(threshold_checks, fail_on_breach=args.fail_on_threshold)
```

### Definition of Done

- [ ] Thresholds loaded from config
- [ ] Each metric checked against threshold
- [ ] Failures logged with severity
- [ ] Blocker failures raise exception
- [ ] Warnings printed but don't fail
- [ ] `--fail-on-threshold` flag working

---

## PBI-24: Add Custom Metric Registration

**Priority**: P2 (Extensibility)
**Estimate**: 8 hours
**Category**: Collect Metrics → Plugins

### Problem

Adding new metrics requires code changes. Users can't define custom metrics without modifying source.

### Acceptance Criteria

1. [ ] Plugin system for metric registration
2. [ ] Load metrics from `metrics/` directory
3. [ ] Validate metric interface
4. [ ] Register custom metrics in config
5. [ ] Custom metrics treated same as built-in

### Definition of Done

- [ ] Plugin system functional
- [ ] Metrics load from directory
- [ ] Interface validation working
- [ ] Config-based registration
- [ ] Documentation and examples

---

## PBI-25: Add Metric Caching

**Priority**: P2 (Performance)
**Estimate**: 4 hours
**Category**: Collect Metrics → Caching

### Problem

LLM-judge metrics recomputed every run. Same query + same answer = repeated API calls. Wasted money and time.

### Acceptance Criteria

1. [ ] Cache metric results by hash(query + answer + model)
2. [ ] Check cache before computing
3. [ ] Store cache in local SQLite DB
4. [ ] `--clear-cache` flag to reset
5. [ ] Cache key logged for debugging

### Implementation Notes

```python
import hashlib
import sqlite3
from pathlib import Path

class MetricCache:
    """Cache for LLM-judge metric results."""

    def __init__(self, cache_path: Path = Path(".metric_cache.db")):
        self.cache_path = cache_path
        self._init_db()

    def _init_db(self):
        """Initialize cache database."""
        conn = sqlite3.connect(self.cache_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                metric_name TEXT,
                query_hash TEXT,
                answer_hash TEXT,
                model TEXT,
                result_json TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    def get(self, metric_name: str, query: str, answer: str, model: str) -> dict | None:
        """Get cached result if available."""
        cache_key = self._compute_key(query, answer, model)

        conn = sqlite3.connect(self.cache_path)
        cursor = conn.execute(
            "SELECT result_json FROM cache WHERE cache_key = ? AND metric_name = ?",
            (cache_key, metric_name)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def put(self, metric_name: str, query: str, answer: str, model: str, result: dict):
        """Store result in cache."""
        cache_key = self._compute_key(query, answer, model)

        conn = sqlite3.connect(self.cache_path)
        conn.execute(
            """INSERT OR REPLACE INTO cache
               (cache_key, metric_name, query_hash, answer_hash, model, result_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            (cache_key, metric_name, self._hash(query), self._hash(answer), model,
             json.dumps(result))
        )
        conn.commit()
        conn.close()

    def _compute_key(self, query: str, answer: str, model: str) -> str:
        """Compute cache key from inputs."""
        combined = f"{model}:{query}:{answer}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _hash(self, s: str) -> str:
        """Hash a string."""
        return hashlib.sha256(s.encode()).hexdigest()[:16]
```

### Definition of Done

- [ ] Cache DB initialized
- [ ] Cache checked before API call
- [ ] Results stored after computation
- [ ] `--clear-cache` flag working
- [ ] Cache hit rate logged

---

## Dependencies

```
PBI-20 (Citation correctness) ────┐
                                 ├──→ Core citation functionality
PBI-21 (Schema citation) ────────┘
                                 ↓
                        PBI-23 (Threshold enforcement)
                                 ↓
                        PBI-24 (Custom metrics)
                                 ↓
                        PBI-25 (Caching)

PBI-22 (Cost reporting) ───────────┘ (Independent, from Call Orchestration)
```

## Summary Table

| PBI | Priority | Estimate | Dependencies |
|-----|----------|----------|--------------|
| Citation correctness metric | P0 | 12h | None |
| Schema citation question | P0 | 8h | None |
| Cost reporting | P1 | 3h | See PBI-8 |
| Threshold enforcement | P1 | 2h | None |
| Custom metric registration | P2 | 8h | None |
| Metric caching | P2 | 4h | None |

**Total P0-P1**: 25 hours
**Total all**: 37 hours

---

## Legal RAG Context

For legal domain users, **citation correctness (PBI-20)** is the single most important missing feature. Current system:
- ✅ Extracts citations from answer
- ✅ Validates chunk_id exists in retrieved set
- ❌ Does NOT verify chunk content supports claim

**Gap**: A system that extracts citations but doesn't verify correctness is dangerous for legal work. Lawyers will not (and should not) trust it.
