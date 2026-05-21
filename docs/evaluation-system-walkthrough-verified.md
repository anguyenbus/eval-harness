# Evaluation System: Verified Technical Reference

**Purpose**: Document what actually exists in the eval-harness codebase for accurate technical reference.

**Verification Method**: Source code analysis of `/src/eval_harness/` modules, confirmed against actual result files in `/results/`.

**Scope**: This document covers the 11 evaluation capabilities requested. Gaps and unimplemented features are explicitly noted.

---

## Table of Contents

1. [Load Evaluation](#1-load-evaluation)
2. [Call Orchestration](#2-call-orchestration)
3. [Offline Evaluation](#3-offline-evaluation)
4. [A/B Routing](#4-ab-routing)
5. [Shadow Evaluation](#5-shadow-evaluation)
6. [Collect Metrics](#6-collect-metrics)
7. [Run Evaluation Framework](#7-run-evaluation-framework)
8. [Metrics Retrieval](#8-metrics-retrieval)
9. [Citation Check](#9-citation-check)
10. [Store Results](#10-store-results)
11. [Comparison](#11-comparison)

---

## 1. Load Evaluation

**Definition**: Running evaluations on datasets of varying sizes to measure performance under load.

### What Exists (Verified)

#### Dataset Slicing
- **Legal RAG Bench** ([`datasets/legal_rag_bench.py`](src/eval_harness/datasets/legal_rag_bench.py:93)):
  - `slice="nano"`: 10 questions for rapid development iteration
  - `slice="full"`: 100 questions (complete test split)
  - Implementation: `_get_slice_limit()` returns limit or None ([line 70](src/eval_harness/datasets/legal_rag_bench.py:70))

- **OmniDocBench** ([`datasets/omnidocbench.py`](src/eval_harness/datasets/omnidocbench.py:43)):
  - English-only filter applied at loader level
  - Document type filter (academic_literature, research_report, exam_paper, etc.)
  - No explicit slice parameter—full filtered dataset is yielded

- **DP-Bench** ([`datasets/dp_bench.py`](src/eval_harness/datasets/dp_bench.py)):
  - Full dataset (1,052 documents)
  - No slice parameter

#### Arbitrary Truncation
- Parsing CLI accepts `--limit` flag ([`run_parsing_eval.py:187`](src/eval_harness/runners/run_parsing_eval.py:187)) for truncating any dataset

#### Retrieval Depth Control
- RAG CLI accepts `--top-k` parameter ([`run_rag_eval.py:291`](src/eval_harness/runners/run_rag_eval.py:291)) for controlling retrieval load (default: 5)

### Key Files

| File | Verified Function |
|------|-------------------|
| `src/eval_harness/datasets/legal_rag_bench.py` | HuggingFace loader with slice support |
| `src/eval_harness/datasets/omnidocbench.py` | English-only filtered loader |
| `src/eval_harness/datasets/dp_bench.py` | DP-Bench loader |

### Gaps / Limitations

1. **No concurrent load testing**: Evaluations run sequentially, not under concurrent load
2. **No resource monitoring**: CPU/memory usage during evaluation is not tracked
3. **OmniDocBench size**: Not configurable via slice parameter; must filter manually

---

## 2. Call Orchestration

**Definition**: How the evaluation pipeline coordinates calls to external systems (parsers, RAG, LLM judges).

### What Exists (Verified)

#### Two Independent Pipelines

**Parsing Pipeline** ([`run_parsing_eval.py`](src/eval_harness/runners/run_parsing_eval.py:153-465)):
```
CLI args → load_config() → load_dataset() → get_parser() →
  for each document:
    ParserAdapter.parse(pdf_path)
      → schema validation (parser_output.schema.json)
    parser_output_to_markdown(output)
    compute all metrics (NID, TEDS, MHS, ARD, BLEU, METEOR)
    write CSV row (incremental flush)
  → calculate averages → write JSON summary
```

**RAG Pipeline** ([`run_rag_eval.py`](src/eval_harness/runners/run_rag_eval.py:250-642)):
```
CLI args → load_config() → load_dataset() → create shared embedder →
  initialize DeepEval evaluator → get RAG adapter →
  for each query:
    RagAdapter.query(question, corpus_dir)
      → schema validation (rag_query_output.schema.json)
    check relevant_passage_retrieved (deterministic)
    DeepEvalEvaluator.compute_metrics_with_reasoning()
      → Faithfulness, ContextualPrecision, ContextualRecall, AnswerRelevancy
    determine verdict (PASS/NEEDS_REVIEW)
    write CSV row (incremental flush)
  → export Phoenix traces (if enabled)
  → calculate averages → write JSON summary + details.json
```

#### Adapter Pattern

- **ParserAdapter** ([`adapters/parser_adapter.py`](src/eval_harness/adapters/parser_adapter.py:20)): Wraps `(pdf_path: Path) -> dict` callable, validates against `contracts/parser_output.schema.json`
- **RagAdapter** ([`adapters/rag_adapter.py`](src/eval_harness/adapters/rag_adapter.py:20)): Wraps `(question: str, corpus_dir: Path) -> dict` callable, validates against `contracts/rag_query_output.schema.json`, optionally passes shared embedder

#### Phoenix Observability (Optional)

When `--enable-phoenix` is set, spans are created via [`observability/phoenix_adapter.py`](src/eval_harness/observability/phoenix_adapter.py:42):

- `eval_run` (root span) - groups all queries
- `rag_query` (CHAIN span) - per query
- `retrieval` (RETRIEVER span) - chunk retrieval
- `generation` (LLM span) - answer generation
- `evaluator` (EVALUATOR span) - DeepEval metrics

### Gaps / Limitations

1. **No parallel query execution**: RAG queries run sequentially; async evaluator exists but runner processes one query at a time
2. **No retry logic**: Failed LLM calls are not retried
3. **No timeout enforcement**: LLM calls can hang indefinitely

---

## 3. Offline Evaluation

**Definition**: Running evaluations on pre-collected datasets without real-time user traffic.

### What Exists (Verified)

The entire framework is designed for offline batch evaluation:

1. **Static benchmark datasets**:
   - OmniDocBench (English-only filtered subset)
   - DP-Bench (1,052 documents)
   - Legal RAG Bench (100 questions from HuggingFace `isaacus/legal-rag-bench`)

2. **Deterministic metrics** (computed from ground truth):
   - Parsing: NID, NID-S, TEDS, TEDS-S, MHS, MHS-S, ARD, BLEU, METEOR
   - RAG: Relevant Passage Retrieved (binary hit@k), latency (ms)

3. **LLM-judge metrics** via **DeepEval** (not RAGAS):
   - Faithfulness: Hallucination detection
   - ContextualPrecision: Signal-to-noise in retrieved context
   - ContextualRecall: Coverage of relevant information
   - AnswerRelevancy: Directness of response

4. **Reproducible configuration**: [`eval_config.yaml`](eval_config.yaml) pins judge model (`gpt-4o`), temperature (`0.0`), embedding model

### Key Design Patterns

- **Iterator pattern**: Datasets yield items one at a time ([`legal_rag_bench.py:164`](src/eval_harness/datasets/legal_rag_bench.py:164))—constant memory
- **Incremental CSV writes**: Each result flushed immediately ([`run_rag_eval.py:140`](src/eval_harness/runners/run_rag_eval.py:140))—crash-safe
- **Schema contracts**: JSON Schema validation before metric computation

### Gaps / Limitations

1. **No online evaluation mode**: Cannot evaluate against live traffic
2. **No streaming evaluation**: All queries must complete before final summary

---

## 4. A/B Routing

**Definition**: Routing evaluation queries to different system variants for comparison.

### What Exists (Verified)

Parser selection via `--parser` flag ([`run_parsing_eval.py:168`](src/eval_harness/runners/run_parsing_eval.py:168)):
- `stub`: Minimal reference implementation
- `fast`: pypdf-based parser for digital PDFs
- `docling`: Full parsing with OCR support

RAG selection via `--rag` flag ([`run_rag_eval.py:264`](src/eval_harness/runners/run_rag_eval.py:264)):
- `stub-local`: ChromaDB-backed reference implementation

Each run produces independently timestamped CSV+JSON pairs in `results/`.

### Example Comparison Workflow

```bash
# Run parser A
uv run eval-parsing --dataset dp_bench --parser fast

# Run parser B
uv run eval-parsing --dataset dp_bench --parser docling

# Compare the JSON summaries manually
```

### Gaps / Limitations

1. **No built-in comparison tool**: Must manually compare JSON summaries
2. **No statistical significance testing**: No p-values or confidence intervals
3. **No side-by-side report**: No combined view of A vs B results

---

## 5. Shadow Evaluation

**Definition**: Running a system variant in parallel without affecting the primary pipeline.

### What Exists (Verified)

**Offline shadow evaluation** is supported:
1. Same dataset can be run through multiple systems
2. Reference implementations (stubs) serve as baselines
3. Deterministic reproducibility via pinned config

**Live shadow mode** does NOT exist:
- No server implementation
- No request mirroring from production traffic

### Reference Implementations

- **ChromaDB RAG stub** ([`stubs/rag/chromadb_query.py`](src/eval_harness/stubs/rag/chromadb_query.py:186)): Complete RAG pipeline for shadow comparison
- **Parser stubs**: stub_parser, digital_pdf_parser, docling_parser

### Gaps / Limitations

1. **No production shadow mode**: Cannot shadow live traffic
2. **No automatic result sync**: Must manually match results across runs

---

## 6. Collect Metrics

**Definition**: Metrics computed for evaluating system performance.

### What Exists (Verified)

#### Parsing Metrics ([`metrics/parsing/`](src/eval_harness/metrics/parsing/))

| Metric | File | Verified Location |
|--------|------|-------------------|
| **NID** | `nid.py` | [`metrics/parsing/nid.py`](src/eval_harness/metrics/parsing/nid.py) |
| **TEDS** | `table_teds.py` | [`metrics/parsing/table_teds.py`](src/eval_harness/metrics/parsing/table_teds.py) |
| **MHS** | `mhs.py` | [`metrics/parsing/mhs.py`](src/eval_harness/metrics/parsing/mhs.py) |
| **ARD** | `reading_order.py` | [`metrics/parsing/reading_order.py`](src/eval_harness/metrics/parsing/reading_order.py) |
| **BLEU** | `text_similarity.py` | [`metrics/parsing/text_similarity.py`](src/eval_harness/metrics/parsing/text_similarity.py) |
| **METEOR** | `text_similarity.py` | [`metrics/parsing/text_similarity.py`](src/eval_harness/metrics/parsing/text_similarity.py) |
| **Layout mAP** | `layout_map.py` | [`metrics/parsing/layout_map.py`](src/eval_harness/metrics/parsing/layout_map.py) |
| **Text Fidelity** | `text_fidelity.py` | [`metrics/parsing/text_fidelity.py`](src/eval_harness/metrics/parsing/text_fidelity.py) |
| **Structure Recall** | `structure_recall.py` | [`metrics/parsing/structure_recall.py`](src/eval_harness/metrics/parsing/structure_recall.py) |

#### RAG Metrics

**Deterministic** (inline in [`run_rag_eval.py:85-87`](src/eval_harness/runners/run_rag_eval.py:85-87)):
- `relevant_passage_retrieved`: Binary check if gold passage ID in retrieved chunks
- `total_ms`: Timing from RAG system output

**LLM-Judge via DeepEval** ([`adapters/deepeval_adapter.py`](src/eval_harness/adapters/deepeval_adapter.py:78)):
- **Faithfulness**: Hallucination detection via claim decomposition
- **ContextualPrecision**: Signal-to-noise in chunk ranking
- **ContextualRecall**: Coverage of gold answer in retrieved context
- **AnswerRelevancy**: Directness of response (cosine similarity)

### Important Note: RAGAS vs DeepEval

- **Current implementation**: Uses DeepEval ([`run_rag_eval.py:382`](src/eval_harness/runners/run_rag_eval.py:382))
- **RAGAS code exists**: [`metrics/ragas_config.py`](src/eval_harness/metrics/ragas_config.py) but is marked deprecated in config
- **Config note**: Line 30 in [`eval_config.yaml`](eval_config.yaml:30) states "RAGAS configuration is deprecated"

### LLM Backend Configuration

From [`metrics/deepeval_config.py`](src/eval_harness/metrics/deepeval_config.py:94):
- Default model: `gpt-4o-mini` (configurable)
- Provider: OpenAI (Bedrock planned)
- Temperature: 0.0
- Max concurrent: 10

### Gaps / Limitations

1. **No custom metric registration**: Cannot add user-defined metrics without code changes
2. **No metric caching**: LLM-judge metrics recomputed on every run
3. **No metric threshold enforcement**: Thresholds in config are not enforced during evaluation

---

## 7. Run Evaluation Framework

**Definition**: Top-level machinery tying datasets, adapters, metrics, and results together.

### What Exists (Verified)

#### CLI Entry Points ([`pyproject.toml:53-55`](pyproject.toml:53-55))

```toml
[project.scripts]
eval-parsing = "eval_harness.runners.run_parsing_eval:main"
eval-rag = "eval_harness.runners.run_rag_eval:main"
```

#### Startup Sequence

**Parsing** ([`run_parsing_eval.py:153-465`](src/eval_harness/runners/run_parsing_eval.py:153-465)):
1. Parse CLI args
2. Load config from `eval_config.yaml`
3. Load dataset via iterator
4. Create parser adapter
5. Open incremental CSV
6. Loop: parse → validate schema → compute metrics → write row
7. Calculate averages
8. Write JSON summary

**RAG** ([`run_rag_eval.py:250-642`](src/eval_harness/runners/run_rag_eval.py:250-642)):
1. Parse CLI args
2. Load config
3. Initialize Phoenix (optional)
4. Create shared embedder
5. Initialize DeepEval evaluator
6. Create RAG adapter
7. Open incremental CSV
8. Optionally wrap in Phoenix eval_run_span
9. Loop: query → validate → check retrieval → compute DeepEval metrics → write row
10. Export Phoenix traces
11. Calculate averages
12. Write JSON summary + details.json (with reasoning)

#### Error Handling

- Parse/query errors: Write error row with all scores = 0.0 ([`run_rag_eval.py:144`](src/eval_harness/runners/run_rag_eval.py:144))
- Schema validation: Raises `SchemaValidationError` ([`adapters/schema_validator.py`](src/eval_harness/adapters/schema_validator.py))
- LLM judge failures: Catch exception, default to 0.0 ([`deepeval_adapter.py:196`](src/eval_harness/adapters/deepeval_adapter.py:196))

### Gaps / Limitations

1. **No resume capability**: Cannot resume interrupted runs from last checkpoint
2. **No dry-run mode**: Cannot validate config without running full evaluation

---

## 8. Metrics Retrieval

**Definition**: How computed metrics are aggregated, accessed, and reported.

### What Exists (Verified)

#### Per-Query CSV Results

**Parsing CSV columns** ([`run_parsing_eval.py:223-235`](src/eval_harness/runners/run_parsing_eval.py:223-235)):
```
query_id, error, nid, nid_s, teds, teds_s, mhs, mhs_s, ard, bleu, meteor
```

**RAG CSV columns** ([`run_rag_eval.py:439-456`](src/eval_harness/runners/run_rag_eval.py:439-456)):
```
query_id, question, gold_answer, generated_answer, relevant_passage_retrieved,
faithfulness_score, context_precision_score, context_recall_score,
answer_relevancy_score, judge_verdict, total_ms, error,
framework_version, metric_computation_time_ms, llm_judge_model
```

#### Aggregated JSON Summaries

Example from actual result file ([`results/legal_rag_bench_nano_results_*.json`](results/legal_rag_bench_nano_results_20260521_211101.json)):
```json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "timestamp": "20260521_211101",
  "csv_file": "legal_rag_bench_nano_results_20260521_211101.csv",
  "metrics_avg": {
    "relevant_passage_retrieved": 0.75,
    "faithfulness_score": 0.9167,
    "context_precision_score": 0.1458,
    "context_recall_score": 0.4375,
    "answer_relevancy_score": 0.8302,
    "total_ms": 4911.43
  },
  "total_processed": 10,
  "errors": 0,
  "top_k": 5,
  "evaluation_framework": "deepeval",
  "framework_version": "4.0.3",
  "judge_model": "gpt-4o"
}
```

#### Phoenix Trace Export

When enabled, traces exported via OTLP gRPC or buffered to Parquet ([`phoenix_adapter.py:481`](src/eval_harness/observability/phoenix_adapter.py:481))

#### Details.json (Reasoning)

New output file with LLM reasoning per query ([`run_rag_eval.py:621-635`](src/eval_harness/runners/run_rag_eval.py:621-635)):
```json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "timestamp": "20260521_211101",
  "total_queries": 10,
  "evaluation_framework": "deepeval",
  "framework_version": "4.0.3",
  "judge_model": "gpt-4o",
  "queries": [
    {
      "query_id": "1",
      "question": "...",
      "reasoning": {
        "faithfulness": {
          "reason": "The generated answer contains...",
          "verdicts": [...],
          "claims": [...],
          "truths": [...]
        },
        "context_precision": {...}
      }
    }
  ]
}
```

### Gaps / Limitations

1. **No built-in visualization**: Must use external tools for charts
2. **No time-series tracking**: Cannot track metric trends over time
3. **HTML summary not integrated**: [`html_summary.py`](src/eval_harness/reporting/html_summary.py) expects different column format

---

## 9. Citation Check

**Definition**: Verifying claims in generated answers trace back to retrieved source chunks.

### What Exists (Verified)

#### Citation Extraction ([`stubs/rag/citations.py:15`](src/eval_harness/stubs/rag/citations.py:15))

```python
def extract_citations(answer: str, retrieved_chunks: list[dict]) -> list[dict]:
```

**Behavior**:
1. Finds `[chunk_id]` pattern in answer text
2. Expands to full sentence containing citation
3. Returns list with `claim_span` (char offsets) and `chunk_ids`

#### Citation Validation ([`stubs/rag/schema_conformance.py:106`](src/eval_harness/stubs/rag/schema_conformance.py:106))

```python
def _validate_citations(output: dict) -> None:
```

**Behavior**:
1. Collects valid chunk_ids from `retrieved_chunks`
2. Verifies every `chunk_id` in `answer.citations` exists
3. Raises `SchemaValidationError` for invalid references

#### Citation Schema

From [`contracts/rag_query_output.schema.json`](contracts/rag_query_output.schema.json):
```json
{
  "answer": {
    "citations": [
      {
        "claim_span": [0, 85],
        "chunk_ids": ["doc1_chunk_00000"]
      }
    ]
  }
}
```

#### Integration in RAG Pipeline

ChromaDB stub calls `extract_citations()` ([`chromadb_query.py:186`](src/eval_harness/stubs/rag/chromadb_query.py:186)) after generation.

### Gaps / Limitations

1. **Pattern-based only**: Only detects explicit `[chunk_id]` format
2. **Not actively evaluated**: Citations validated but no metric score generated
3. **Schema-defined but unused**: `eval_questions.schema.json` defines `citation_spans_support_claims` evaluation question but current pipeline uses DeepEval metrics instead

---

## 10. Store Results

**Definition**: How evaluation outputs are persisted.

### What Exists (Verified)

#### Incremental CSV Files

Location: `results/` directory

Format: `legal_rag_bench_{slice}_results_{timestamp}.csv`

Key design ([`run_rag_eval.py:459`](src/eval_harness/runners/run_rag_eval.py:459)):
- Open in append mode `"a"`
- Flush after every row ([`run_rag_eval.py:140`](src/eval_harness/runners/run_rag_eval.py:140))
- Crash-safe and progress-visible

#### JSON Summaries

Accompanies each CSV with same base filename: `.json`

Contains: `metrics_avg`, `metadata`, `configuration`

#### Phoenix Trace Storage

- **Connected**: OTLP gRPC to Phoenix server
- **Disconnected**: Parquet files at `/tmp/phoenix_traces/` ([`phoenix_adapter.py:516`](src/eval_harness/observability/phoenix_adapter.py:516))

#### S3 Upload (Optional)

[`phoenix_adapter.py:538`](src/eval_harness/observability/phoenix_adapter.py:538) provides `upload_parquet_to_s3()` for buffered traces.

### Gaps / Limitations

1. **No database storage**: All results in flat files
2. **No deduplication**: Re-runs create new files
3. **No retention policy**: Old files not automatically cleaned up

---

## 11. Comparison

**Definition**: Comparing evaluation results across runs and system variants.

### What Exists (Verified)

#### Regression Check Function ([`reporting/regression_check.py:7`](src/eval_harness/reporting/regression_check.py:7))

```python
def check_regression(current_results: Path, baseline_path: Path, threshold: float = 0.05)
```

**Behavior**:
1. Loads two JSON result files
2. Checks for metrics with `severity: "blocker"`
3. Flags relative decrease > 5%

**⚠️ CONFIG MISMATCH**: 
- Function expects `metrics` dict with `severity` field ([line 39](src/eval_harness/reporting/regression_check.py:39))
- Actual JSON summaries use `metrics_avg` without severity ([run_rag_eval.py:599](src/eval_harness/runners/run_rag_eval.py:599))
- **This function will not work with current JSON output format**

#### Multi-Variant Comparison

Since each run produces independent CSV+JSON:
1. Run variant A
2. Run variant B
3. Manually compare `metrics_avg` fields

#### Diagnostic Matrix (Document)

[`docs/legal-rag-bench-guide.md`](docs/legal-rag-bench-guide.md) provides metric interpretation guidance (documented but not verified to exist).

### Gaps / Limitations

1. **Regression check incompatible**: Does not work with current JSON format
2. **No automated comparison tool**: Must manually compare files
3. **No diff visualization**: No side-by-side view
4. **No statistical tests**: No significance testing

---

## Summary of Gaps

| Capability | Status | Notes |
|------------|--------|-------|
| Load evaluation | ✅ Implemented | Slicing, truncation, top-k control |
| Call orchestration | ✅ Implemented | Sequential per-query with adapter pattern |
| Offline evaluation | ✅ Implemented | Primary mode of operation |
| A/B routing | ✅ Implemented | CLI flags for system selection |
| Shadow evaluation | ⚠️ Partial | Offline only; no live shadow mode |
| Collect metrics | ✅ Implemented | Parsing (9 deterministic) + RAG (DeepEval) |
| Run framework | ✅ Implemented | Two CLI entry points |
| Metrics retrieval | ✅ Implemented | CSV + JSON + details.json |
| Citation check | ⚠️ Partial | Extraction/validation exists, no score metric |
| Store results | ✅ Implemented | Incremental CSV + JSON + Parquet |
| Comparison | ⚠️ Broken | Regression check incompatible with JSON format |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Entry Points                         │
│  eval-parsing                          eval-rag                 │
└────────┬───────────────────────────────────┬────────────────────┘
         │                                   │
┌────────▼────────┐              ┌───────────▼───────────┐
│  ParserAdapter   │              │     RagAdapter         │
│  (schema valid.) │              │  (schema validation)   │
└────────┬────────┘              └───────────┬───────────┘
         │                                   │
┌────────▼────────┐              ┌───────────▼───────────┐
│  Parsers         │              │  RAG Systems           │
│  - stub          │              │  - stub-local (ChromaDB)│
│  - fast (pypdf)  │              │                        │
│  - docling       │              │                        │
└────────┬────────┘              └───────────┬───────────┘
         │                                   │
┌────────▼───────────────────────────────────▼──────────────────┐
│                      Metrics Layer                             │
│  Parsing: NID, TEDS, MHS, ARD, BLEU, METEOR, TextF1, mAP     │
│  RAG:      DeepEval (Faithfulness, ContextualPrecision/Recall,│
│            AnswerRelevancy) + deterministic (hit@k, latency)  │
└────────┬───────────────────────────────────┬──────────────────┘
         │                                   │
┌────────▼────────┐              ┌───────────▼───────────┐
│  CSV + JSON      │              │  Phoenix Observability │
│  (incremental)   │              │  (OTLP spans/traces)   │
└────────┬────────┘              └───────────────────────┘
         │
┌────────▼────────┐
│  Details.json    │
│  (LLM reasoning) │
└─────────────────┘
```

---

## File Reference

### Core Implementation

| File | Purpose |
|------|---------|
| `src/eval_harness/runners/run_parsing_eval.py` | Parsing evaluation CLI |
| `src/eval_harness/runners/run_rag_eval.py` | RAG evaluation CLI (DeepEval) |
| `src/eval_harness/adapters/parser_adapter.py` | Parser wrapper with schema validation |
| `src/eval_harness/adapters/rag_adapter.py` | RAG wrapper with schema validation |
| `src/eval_harness/adapters/deepeval_adapter.py` | DeepEval metrics evaluator |

### Datasets

| File | Purpose |
|------|---------|
| `src/eval_harness/datasets/legal_rag_bench.py` | HuggingFace loader with slice support |
| `src/eval_harness/datasets/omnidocbench.py` | English-only filtered loader |
| `src/eval_harness/datasets/dp_bench.py` | DP-Bench loader |

### Metrics

| File | Purpose |
|------|---------|
| `src/eval_harness/metrics/deepeval_config.py` | DeepEval LLM-judge configuration |
| `src/eval_harness/metrics/ragas_config.py` | RAGAS (deprecated) |
| `src/eval_harness/metrics/parsing/` | All parsing metric implementations |

### Reporting

| File | Purpose |
|------|---------|
| `src/eval_harness/reporting/regression_check.py` | Regression comparison (⚠️ format mismatch) |
| `src/eval_harness/reporting/html_summary.py` | HTML generation (⚠️ column mismatch) |

### Observability

| File | Purpose |
|------|---------|
| `src/eval_harness/observability/phoenix_adapter.py` | Phoenix span creation and export |

---

## Verification Notes

This document was created by:
1. Reading all source files in `src/eval_harness/`
2. Confirming CLI entry points in `pyproject.toml`
3. Verifying actual result file formats in `results/`
4. Cross-referencing schema contracts in `contracts/`

**Date**: 2026-05-21
**Codebase Commit**: deepeval branch
**Method**: Static analysis + runtime output verification
