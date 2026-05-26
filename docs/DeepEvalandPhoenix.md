# Phoenix + DeepEval Integration Guide

**Video walkthrough:** [Experiments Walkthrough](https://www.youtube.com/watch?v=DZqgvcJ2t2o&feature=youtu.be)

This project wraps [DeepEval](https://github.com/confident-ai/deepeval) LLM-as-judge metrics as [Arize Phoenix](https://arize.com/phoenix/) evaluators. You keep DeepEval's judgment quality (faithfulness, context precision/recall, answer relevancy) and gain Phoenix's experiment UI, cost tracking, dataset versioning, and run-to-run comparison.

## Quick start

```python
from eval_harness.experiments.runner import run_phoenix_experiment

experiment = run_phoenix_experiment(
    rag_adapter=rag_adapter,
    corpus_dir=Path("/path/to/corpus"),
    endpoint="http://localhost:6006",
    slice_name="pico",            # pico (2 Qs), nano (~10), or full
    judge_model="gpt-4o-mini",
)
```

Results stream to the Phoenix UI at the configured endpoint and are also exported to CSV and Parquet under `results/`. Use the `pico` slice first — it runs in seconds and catches most config errors.

## Architecture

```
                          ┌─────────────────────────────┐
                          │      Phoenix server         │
                          │  (datasets, experiments,    │
                          │   traces, UI at :6006)      │
                          └──────────────┬──────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │   run_phoenix_experiment    │
                          │   (orchestrator)            │
                          └──────┬───────────────┬──────┘
                                 │               │
                ┌────────────────▼─┐   ┌─────────▼─────────────┐
                │  RAG task        │   │  DeepEval evaluators  │
                │  (your pipeline) │   │  • faithfulness       │
                │                  │   │  • context_precision  │
                │  returns:        │   │  • context_recall     │
                │   answer +       │   │  • answer_relevancy   │
                │   retrieval_ctx  │   └─────────┬─────────────┘
                └──────────────────┘             │
                                                 ▼
                                   ┌───────────────────────────┐
                                   │  Export: CSV + Parquet    │
                                   │  with per-metric verdicts │
                                   │  and cost breakdown       │
                                   └───────────────────────────┘
```

Phoenix's `run_experiment()` is responsible for:

1. **Dataset management** — versions the test questions so runs are comparable.
2. **Task execution** — invokes your RAG pipeline once per example.
3. **Evaluation** — applies the wrapped DeepEval judges to each task result.
4. **Aggregation** — collects scores, labels, explanations, and costs into a single experiment record.

## DeepEval metrics

| Metric | Question it answers | Required inputs |
|---|---|---|
| **Faithfulness** | Are the answer's claims supported by the retrieved context? (hallucination check) | `actual_output`, `retrieval_context` |
| **Context Precision** | Are the retrieved chunks ranked by relevance? (signal-to-noise) | `retrieval_context`, `expected_output` |
| **Context Recall** | Does the retrieved context cover what the expected answer needs? | `retrieval_context`, `expected_output` |
| **Answer Relevancy** | Does the answer address the question? | `input`, `actual_output` |

### The verdict system (`yes` / `no` / `idk`)

DeepEval scores each metric by decomposing the output into atomic claims, then having a judge LLM classify each claim. All four metrics use the same three-state verdict scheme:

| Verdict | Meaning | Effect on score |
|---|---|---|
| `yes` | Statement passes judgment | counts as pass |
| `no` | Statement fails judgment | counts as fail |
| `idk` | Ambiguous or not contradicted by context | counts as pass |

**Score formula:** `(yes_count + idk_count) / total_statements`

The critical insight is that **`idk` is treated as a pass, not a partial credit**. This is deliberate — it stops the judge from being forced into a binary call on borderline material. The downside is that a score of `1.0` made entirely of `idk` verdicts means "nothing was clearly wrong," which is not the same as "everything was clearly right." Always inspect the verdict breakdown for high scores that feel surprising.

All four built-in metrics emit `yes`/`no`/`idk` verdicts. The class names differ — `FaithfulnessVerdict`, `ContextualPrecisionVerdict`, `VerdictWithExpectedOutput` (recall), `AnswerRelevancyVerdict` — but the value range is the same.

## Implementation

### Evaluator wrapper pattern

Each DeepEval metric is wrapped behind a Phoenix `@create_evaluator` decorator. The wrapper translates Phoenix's task output into a DeepEval `LLMTestCase`, runs the metric, and returns a Phoenix `Score`:

```python
from phoenix.evals import create_evaluator, Score
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

@create_evaluator(name="faithfulness", kind="llm")
def faithfulness_evaluator(output, expected=None, input=None):
    # 1. Pull what we need out of the task output
    actual_answer = output.get("answer", "")
    retrieval_context = output.get("retrieval_context", [])

    # 2. Skip cleanly if a required input is missing
    if not retrieval_context or not actual_answer:
        return {"score": 0.0, "label": "skipped", "explanation": "missing inputs"}

    # 3. Instantiate the metric *per call* (see thread-safety note below)
    metric = FaithfulnessMetric(model=judge_model, include_reason=True)

    # 4. Run the judge
    test_case = LLMTestCase(
        input=input_str,
        actual_output=actual_answer,
        retrieval_context=retrieval_context,
    )
    metric.measure(test_case)

    # 5. Pack metadata — keep the full verdict trail, not just pass/fail
    metadata = {
        "threshold": metric.threshold,
        "success": metric.success,
        "evaluation_cost": getattr(metric, "evaluation_cost", None),
        "verdicts": _serialize_verdicts(metric.verdicts),
    }

    return Score(
        name="faithfulness",
        score=float(metric.score),
        label="faithful" if metric.success else "unfaithful",
        explanation=metric.reason or "",
        metadata=metadata,
    )
```

### Design decisions worth understanding

**Instantiate metrics per call, not at module scope.** Phoenix runs evaluators concurrently. A DeepEval metric stores `score`, `reason`, and `verdicts` as instance attributes that get overwritten on each `measure()` call, so a shared instance creates a race condition where the last-finishing call overwrites earlier results.

```python
# WRONG — race condition under Phoenix's concurrent execution
metric = FaithfulnessMetric(model=judge_model)  # module-level

@create_evaluator(...)
def evaluator(...):
    metric.measure(test_case)   # mutates shared state
    return metric.score         # may belong to a different call

# CORRECT — fresh instance per invocation
@create_evaluator(...)
def evaluator(...):
    metric = FaithfulnessMetric(model=judge_model)
    metric.measure(test_case)
    return metric.score
```

**Use lowercase `kind="llm"`.** Phoenix's `create_evaluator` signature is `create_evaluator(name, source=None, direction="maximize", kind=None)`, and `kind` accepts only `"human"`, `"llm"`, or `"code"`. Uppercase `"LLM"` is silently accepted but causes evaluators to display as `CODE` in the Experiments UI.

**Keep verdicts for both passes and fails.** A high score made of `idk` verdicts is qualitatively different from one made of `yes` verdicts, and you can't tell them apart from the aggregate score alone. Serialize the full verdict list every time:

```python
if hasattr(metric, "verdicts"):    # no `metric.success` gate
    metadata["verdicts"] = _serialize_verdicts(metric.verdicts)
```

**Suppress nested judge spans.** DeepEval's internal LLM calls would otherwise create child spans inside the task's trace, cluttering the UI:

```python
with _suppress_tracing_if_available():
    metric.measure(test_case)
```

## Cost tracking

Each metric exposes `evaluation_cost` after `measure()` returns. The wrapper forwards it into Score metadata, and the exporter aggregates costs into the result file:

| Column | What it is |
|---|---|
| `app_cost_usd` | RAG pipeline cost (from Phoenix `task_run.cost`) |
| `judge_faithfulness_cost_usd` | Faithfulness judge LLM calls |
| `judge_context_precision_cost_usd` | Context Precision judge LLM calls |
| `judge_context_recall_cost_usd` | Context Recall judge LLM calls |
| `judge_answer_relevancy_cost_usd` | Answer Relevancy judge LLM calls |
| `judge_cost_usd` | Sum of all four judge costs |
| `total_cost_usd` | `app_cost_usd + judge_cost_usd` |

Use `getattr(metric, "evaluation_cost", None)` rather than direct attribute access — some DeepEval versions don't populate the field, and a missing attribute should produce `None`, not an `AttributeError`.

## Export format

Each row of the export covers one query. Per-metric columns follow a consistent pattern:

| Column suffix | Contents |
|---|---|
| `_score` | Numeric 0.0–1.0 |
| `_label` | DeepEval pass/fail label (e.g. `faithful` / `unfaithful`) |
| `_verdicts` | JSON array of individual verdicts with reasons |
| `_cost_usd` | LLM cost for that judge |

Prefer Parquet over CSV for verdict analysis — nested JSON survives the round-trip cleanly and column types are preserved.

## Reading the results

### Top-line summary

```json
{
  "metrics_avg": {
    "faithfulness_score": 0.85,
    "context_precision_score": 0.42,
    "context_recall_score": 0.72,
    "answer_relevancy_score": 0.91
  },
  "total_processed": 50,
  "errors": 0
}
```

### Drilling into verdicts

```python
import json
import polars as pl

df = pl.read_parquet("results/experiment-RXhwZXJpbWVudDo3_results.parquet")

row = df.filter(pl.col("query_id") == "2").row(0, named=True)
for v in json.loads(row["faithfulness_verdicts"]):
    print(f"{v['verdict']}: {v['reason']}")
```

### Score patterns and what they usually mean

| Pattern | Likely cause | Where to look first |
|---|---|---|
| High faithfulness, low context precision | Answers are factual but built from noisy context | Retrieval (chunking, reranking, top-k) |
| Low faithfulness, high context precision | Good context, but generation invents claims | Generation prompt; consider lowering temperature |
| High answer relevancy, lots of "I don't know" answers | Evasive answers score well because they don't contradict the question | Add a custom G-Eval to penalize non-answers |
| Score 1.0 dominated by `idk` verdicts | Ambiguous claims; nothing wrong but nothing clearly right | Re-examine question phrasing and prompt; the metric isn't telling you what you think |

## References

- [Phoenix Experiments docs](https://arize.com/docs/phoenix/datasets-and-experiments/how-to-experiments)
- [Phoenix `create_evaluator` reference](https://arize-phoenix.readthedocs.io/projects/evals/)
- [DeepEval on GitHub](https://github.com/confident-ai/deepeval)
- [DeepEval verdict system (internal doc)](deepeval-verdict-system.md)