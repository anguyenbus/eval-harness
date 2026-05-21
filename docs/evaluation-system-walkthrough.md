# eval-harness: Evaluation System Walkthrough

This document explains in detail what was built in this project, organized by the key evaluation capabilities: load evaluation, call orchestration, offline evaluation, A/B routing, shadow evaluation, metrics collection, evaluation framework execution, metrics retrieval, citation checking, result storage, and comparison/regression.

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

Load evaluation refers to the ability to run evaluations against datasets of varying sizes under controlled conditions, measuring system performance as data volume changes.

### What was built

The project supports **dataset slicing** for controlled load testing:

- **Legal RAG Bench** offers two slices: `nano` (10 questions) for rapid development iteration and `full` (100 questions) for complete evaluation runs.
- **OmniDocBench** is filtered from 1,651 pages down to 593 English-only pages using a configurable filter (`src/eval_harness/datasets/omnidocbench.py:43`).
- A `--limit` CLI flag allows arbitrary truncation of any dataset for quick test runs (`run_parsing_eval.py:188`).
- The `--top-k` parameter controls retrieval depth (default 5), enabling evaluation of how retrieval load affects metrics (`run_rag_eval.py:262`).

### Key files

| File | Role |
|------|------|
| `src/eval_harness/datasets/legal_rag_bench.py` | Dataset loader with slice support (`nano`/`full`) |
| `src/eval_harness/datasets/omnidocbench.py` | English-only filtered loader |
| `src/eval_harness/datasets/dp_bench.py` | DP-Bench loader (1,052 docs) |
| `eval_config.yaml` | `k_values: [5, 10, 20]` for retrieval depth |

### How it works

1. `load_legal_rag_bench()` accepts a `slice` parameter that maps to a row limit (`SLICE_NANO = 10` or `None` for full).
2. The HuggingFace `datasets` library downloads and caches the `isaacus/legal-rag-bench` QA split.
3. Items are yielded as an iterator (constant memory), so dataset size does not affect memory footprint.
4. OmniDocBench filtering applies language (`english`), document type (`academic_literature`, `research_report`, etc.), and tags noisy pages (`fuzzy_scan`, `watermark`) for stratified analysis without exclusion.

---

## 2. Call Orchestration

Call orchestration refers to how the evaluation pipeline coordinates calls to external systems (parsers, RAG pipelines, LLM judges) and manages the sequence of operations.

### What was built

The project implements a **sequential per-query orchestration** pattern with two independent pipelines:

#### Parsing Pipeline (`eval-parsing` CLI)

Orchestration flow in `src/eval_harness/runners/run_parsing_eval.py`:

```
CLI args → load_config() → load_dataset() → get_parser() →
  for each document:
    ParserAdapter.parse(pdf_path)
      → user's parse function (stub/fast/docling)
      → schema validation (parser_output.schema.json)
    parser_output_to_markdown(output)
    compute all metrics (NID, TEDS, MHS, ARD, BLEU, METEOR)
    write CSV row (incremental flush)
  → calculate averages
  → write JSON summary
```

#### RAG Pipeline (`eval-rag` CLI)

Orchestration flow in `src/eval_harness/runners/run_rag_eval.py`:

```
CLI args → load_config() → load_dataset() → create shared embedder →
  initialize RAGAS evaluator → get RAG adapter →
  for each query:
    RagAdapter.query(question, corpus_dir)
      → user's RAG function (stub-local/ChromaDB/custom)
      → schema validation (rag_query_output.schema.json)
    check relevant_passage_retrieved
    RagasEvaluator.compute_metrics(output, gold_answer)
      → transform to RAGAS SingleTurnSample
      → compute Faithfulness (async)
      → compute ContextPrecision (async)
      → compute ContextRecall (async)
      → compute AnswerRelevancy (async)
    determine verdict (PASS/NEEDS_REVIEW)
    write CSV row (incremental flush)
  → calculate averages
  → write JSON summary
  → export Phoenix traces (if enabled)
```

### Adapter pattern

Both pipelines use the **adapter pattern** to decouple user systems from the framework:

- **ParserAdapter** (`src/eval_harness/adapters/parser_adapter.py`): Wraps a `(pdf_path: Path) -> dict` callable, validates output against `parser_output.schema.json`.
- **RagAdapter** (`src/eval_harness/adapters/rag_adapter.py`): Wraps a `(question: str, corpus_dir: Path) -> dict` callable, validates output against `rag_query_output.schema.json`, optionally passes a shared embedder instance.

### Phoenix observability (optional)

When `--enable-phoenix` is set, the RAG pipeline wraps all queries in OpenTelemetry spans:

- `eval_run` (root span) groups all queries
- `rag_query` (CHAIN span) per query
- `retrieval` (RETRIEVER span) for chunk retrieval
- `generation` (LLM span) for answer generation
- `evaluator` (EVALUATOR span) for RAGAS metrics

Span hierarchy is managed via context managers in `src/eval_harness/observability/phoenix_adapter.py`.

---

## 3. Offline Evaluation

Offline evaluation means running evaluations on pre-collected datasets without real-time user traffic. This is the primary evaluation mode of eval-harness.

### What was built

The entire framework is designed for **offline batch evaluation**:

1. **Static benchmark datasets**: OmniDocBench, DP-Bench, Legal RAG Bench — all pre-downloaded and cached locally.
2. **Deterministic metrics** computed purely from ground truth:
   - Parsing: NID, NID-S, TEDS, TEDS-S, MHS, MHS-S, ARD, BLEU, METEOR
   - RAG: Relevant Passage Retrieved (binary hit@k), latency (ms)
3. **LLM-as-judge metrics** via RAGAS: Faithfulness, Context Precision, Context Recall, Answer Relevancy — all computed against gold reference answers, not live traffic.
4. **Reproducible configuration**: `eval_config.yaml` pins judge model (`gpt-4o`), temperature (`0`), embedding model (`all-MiniLM-L6-v2`), and metric thresholds.

### Key design decisions

- **Iterator pattern** for dataset loading — constant memory regardless of dataset size.
- **Incremental CSV writing** — each query result is flushed to disk immediately, enabling crash recovery and real-time progress visibility.
- **Schema contracts** — every parser/RAG output is validated against JSON Schema before metric computation, ensuring data quality in offline runs.

---

## 4. A/B Routing

A/B routing refers to the ability to route evaluation queries to different system variants and compare their results.

### What was built

The project supports **parser and RAG system selection** via CLI flags:

```bash
# Compare different parsers
uv run eval-parsing --dataset omnidocbench --parser stub
uv run eval-parsing --dataset omnidocbench --parser fast
uv run eval-parsing --dataset omnidocbench --parser docling

# Compare different RAG configurations
uv run eval-rag --slice full --rag stub-local --top-k 5
uv run eval-rag --slice full --rag stub-local --top-k 10
uv run eval-rag --slice full --rag stub-local --top-k 20
```

Each run produces an independently timestamped CSV+JSON result pair in `results/`, enabling post-hoc comparison across system variants.

### Implementation

- `get_parser()` in `run_parsing_eval.py:58` dispatches to `stub`, `fast` (pypdf), or `docling` parsers based on the `--parser` flag.
- `get_rag()` in `run_rag_eval.py:161` dispatches to `stub-local` (ChromaDB) or falls back with a warning for unknown names.
- The adapter pattern makes it straightforward to add new system variants without modifying the framework — just implement a new query/parse function and register it.

### How to compare

Run evaluations with different `--parser` or `--rag` flags, then compare the JSON summaries:

```bash
# Parser A/B comparison
uv run eval-parsing --dataset dp_bench --parser fast   # → results/dp_bench_fast_results_*.json
uv run eval-parsing --dataset dp_bench --parser docling # → results/dp_bench_docling_results_*.json

# Compare averages across the two JSON files
```

---

## 5. Shadow Evaluation

Shadow evaluation refers to running a system in parallel without affecting the primary pipeline — evaluating a new system variant against the same inputs as the production variant.

### What was built

While the project does not implement a live shadow-mode server, its architecture fully supports **offline shadow evaluation**:

1. **Same dataset, different systems**: Run the same dataset slice through multiple parsers/RAG systems. The framework produces comparable CSV outputs with identical column schemas.
2. **Reference implementation (stubs)**: The `stubs/` directory provides complete reference implementations (ChromaDB RAG, pypdf parser, docling parser) that serve as baselines. Real systems can be evaluated alongside these stubs.
3. **Deterministic reproducibility**: Temperature=0, pinned models, fixed dataset slices ensure runs are reproducible.
4. **Phoenix trace capture**: When enabled, Phoenix captures per-query traces including retrieval, generation, and evaluation spans. These traces can be compared across runs.

### Shadow RAG pipeline

The ChromaDB stub (`src/eval_harness/stubs/rag/chromadb_query.py`) implements a complete shadow RAG pipeline:

```
question → embed → ChromaDB query → retrieve top-k → generate answer (Claude/GPT) →
  extract citations → validate schema → return
```

This runs independently of any production RAG system, allowing teams to compare their production system's outputs against this reference implementation using the same evaluation metrics.

---

## 6. Collect Metrics

The project implements two categories of metrics: **deterministic** (computed from ground truth) and **LLM-judge** (computed via RAGAS with an LLM).

### Parsing Metrics (all deterministic)

Located in `src/eval_harness/metrics/parsing/`:

| Metric | File | What it measures |
|--------|------|------------------|
| **NID** | `nid.py` | Normalized Indel Distance — text similarity using fuzzy matching (rapidfuzz). Returns NID (full) and NID-S (tables stripped). |
| **TEDS** | `table_teds.py` | Tree Edit Distance Similarity for tables. Parses HTML table trees, uses APTED algorithm. Returns TEDS (content+structure) and TEDS-S (structure only). |
| **MHS** | `mhs.py` | Markdown Hierarchical Similarity — heading hierarchy preservation via tree edit distance. Returns MHS (with text) and MHS-S (structure only). |
| **ARD** | `reading_order.py` | Average Rank Distance — element ordering quality. Computes average positional displacement between predicted and gold sequences. |
| **BLEU** | `text_similarity.py` | N-gram overlap via sacrebleu, normalized to [0, 1]. |
| **METEOR** | `text_similarity.py` | Harmonic mean of precision/recall with stemming via NLTK. |
| **Layout mAP** | `layout_map.py` | COCO-style mean Average Precision for bounding box detection using torchmetrics. |
| **Text Fidelity** | `text_fidelity.py` | Character-level F1 score on character frequency maps. |
| **Structure Recall** | `structure_recall.py` | Fraction of gold element types detected by parser. |

### RAG Metrics

**Deterministic** (computed inline in `run_rag_eval.py`):

| Metric | What it measures |
|--------|------------------|
| **Relevant Passage Retrieved** | Binary: was the gold passage ID found in retrieved chunks? |
| **Latency** | Per-stage timing: retrieval_ms, generation_ms, total_ms |

**LLM-Judge via RAGAS** (`src/eval_harness/adapters/ragas_adapter.py` and `src/eval_harness/metrics/ragas_config.py`):

| Metric | What it measures | How |
|--------|------------------|-----|
| **Faithfulness** | Are generated claims supported by retrieved context? | LLM decomposes answer into claims, checks each against context |
| **Context Precision** | Signal-to-noise in retrieved chunk ranking | Weighted precision: are relevant chunks ranked higher? |
| **Context Recall** | Coverage of gold answer in retrieved context | LLM decomposes gold answer into claims, checks retrieval coverage |
| **Answer Relevancy** | Directness of response to question | Generates questions from answer, measures cosine similarity to original |

### LLM backend configuration

- **Judge model**: Configured in `eval_config.yaml` → `datasets.legal_rag_bench.ragas.judge_model` (default: `gpt-4o`).
- **LLM backend**: Created via `get_llm_backend()` in `ragas_config.py` using `AsyncOpenAI` + `ragas.llms.llm_factory()`.
- **Embeddings**: Shared between RAG retrieval and RAGAS AnswerRelevancy. Configured as `huggingface` (local) or `openai` (API).

---

## 7. Run Evaluation Framework

The evaluation framework is the top-level machinery that ties datasets, adapters, metrics, and results together.

### CLI entry points

Registered in `pyproject.toml` as console scripts:

```toml
[project.scripts]
eval-parsing = "eval_harness.runners.run_parsing_eval:main"
eval-rag = "eval_harness.runners.run_rag_eval:main"
```

### Framework startup sequence

#### Parsing evaluation (`eval-parsing`)

1. Parse CLI args (`--dataset`, `--parser`, `--config`, `--output-dir`, `--limit`).
2. Load configuration from `eval_config.yaml` with environment variable expansion.
3. Load dataset via iterator (OmniDocBench or DP-Bench).
4. Create parser adapter (stub/fast/docling).
5. Open incremental CSV file with timestamped name.
6. Loop over dataset items:
   - Parse document → validate schema → convert to markdown → compute all metrics → write row.
7. Calculate averages across valid (non-error) rows.
8. Write JSON summary with metadata, averages, and counts.

#### RAG evaluation (`eval-rag`)

1. Parse CLI args (`--slice`, `--rag`, `--config`, `--output-dir`, `--force-reingest`, `--top-k`, `--enable-phoenix`, `--phoenix-endpoint`).
2. Load configuration.
3. Optionally initialize Phoenix adapter for observability.
4. Create shared embedder (`HuggingFaceEmbedder` with `all-MiniLM-L6-v2`).
5. Initialize RAGAS evaluator with shared embedder.
6. Create RAG adapter (stub-local ChromaDB or custom).
7. Open incremental CSV file.
8. Optionally wrap entire run in Phoenix `eval_run_span`.
9. Loop over dataset queries:
   - Optionally wrap in Phoenix `rag_query_span`.
   - Query RAG → validate schema → check passage retrieval → compute RAGAS metrics → determine verdict → write row.
10. Export Phoenix traces.
11. Calculate averages.
12. Write JSON summary.

### Error handling

- Parse/query errors produce error rows (all metric scores = 0.0, verdict = `ERROR`) but do not halt the run.
- Schema validation failures raise `SchemaValidationError` with field path and details.
- LLM judge failures (RAGAS) catch exceptions and default scores to 0.0 with stderr logging.

---

## 8. Metrics Retrieval

Metrics retrieval refers to how computed metrics are aggregated, accessed, and reported after evaluation runs.

### What was built

#### Per-query CSV results

Each evaluation run produces a CSV file with one row per query/document:

**Parsing CSV columns**: `query_id`, `error`, `nid`, `nid_s`, `teds`, `teds_s`, `mhs`, `mhs_s`, `ard`, `bleu`, `meteor`

**RAG CSV columns**: `query_id`, `question`, `gold_answer`, `generated_answer`, `relevant_passage_retrieved`, `faithfulness_score`, `context_precision_score`, `context_recall_score`, `answer_relevancy_score`, `judge_verdict`, `total_ms`, `error`

#### Aggregated JSON summaries

Each CSV is accompanied by a JSON summary file with averaged metrics:

```json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "timestamp": "20260520_223534",
  "csv_file": "legal_rag_bench_nano_results_20260520_223534.csv",
  "metrics_avg": {
    "relevant_passage_retrieved": 0.6,
    "faithfulness_score": 0.85,
    "context_precision_score": 0.42,
    "context_recall_score": 0.54,
    "answer_relevancy_score": 0.67,
    "total_ms": 1523.5
  },
  "total_processed": 10,
  "errors": 0,
  "top_k": 5
}
```

#### Phoenix trace export

When Phoenix is enabled, traces are exported via OTLP gRPC to the Phoenix server, or buffered to Parquet files when Phoenix is unreachable:

```python
# From phoenix_adapter.py:export_traces()
{
    "trace_count": 10,
    "mode": "phoenix",  # or "parquet"
    "path": "/tmp/phoenix_traces/evaluations_20260520_223534.parquet"
}
```

#### HTML summary reports

`src/eval_harness/reporting/html_summary.py` generates HTML reports with pass rates, score distributions, and per-question breakdowns.

---

## 9. Citation Check

Citation checking verifies that claims in generated answers are properly traced back to retrieved source chunks.

### What was built

#### Citation extraction

`src/eval_harness/stubs/rag/citations.py:15` implements `extract_citations()`:

1. Identifies chunk_id references in answer text using the pattern `[doc1_chunk_00000]`.
2. For each reference, expands to the full sentence containing the citation.
3. Records `claim_span` (character offsets `[start, end)`) and `chunk_ids` (list of referenced chunks).
4. Deduplicates citations with the same span.

#### Citation validation

`src/eval_harness/stubs/rag/schema_conformance.py:106` implements `_validate_citations()`:

1. Collects all valid `chunk_id`s from `retrieved_chunks`.
2. Verifies every `chunk_id` referenced in `answer.citations` exists in the retrieved set.
3. Raises `SchemaValidationError` for invalid references.

#### Citation schema

The `rag_query_output.schema.json` contract requires:

```json
{
  "answer": {
    "citations": [
      {
        "claim_span": [0, 85],        // character offsets in answer text
        "chunk_ids": ["doc1_chunk_00000"]  // referenced chunks
      }
    ]
  }
}
```

#### Citation in the RAG pipeline

The ChromaDB stub pipeline (`chromadb_query.py:186`) integrates citation extraction:

```python
citations = extract_citations(answer_result["text"], retrieved_chunks)
```

The generator's system prompt instructs the LLM to cite sources:

> "When answering, you MUST cite your sources using the chunk_ids in square brackets like [chunk_id]."

#### Citation evaluation (planned)

The `eval_questions.schema.json` declares a `citation_spans_support_claims` LLM-judge evaluation question that checks whether each cited chunk actually contains the information used in the claim. This is defined in the schema but the current RAG pipeline uses RAGAS metrics rather than the explicit citation evaluation.

---

## 10. Store Results

Result storage refers to how evaluation outputs are persisted for analysis, comparison, and regression testing.

### What was built

#### Incremental CSV files

Results are written to `results/` with timestamped filenames:

```
results/omnidocbench_fast_results_20260520_223534.csv
results/legal_rag_bench_nano_results_20260520_223534.csv
```

**Key design**: CSV files are opened in append mode (`"a"`) and flushed after every row (`csv_file.flush()`), ensuring:
- Progress visibility during long runs
- Crash recovery (partial results are preserved)
- Constant memory (no accumulation)

#### JSON summaries

Each CSV is paired with a JSON summary containing averaged metrics, counts, and configuration metadata. These serve as the primary artifacts for comparison and regression checking.

#### Phoenix trace storage

When Phoenix is enabled:
- Traces are sent to Phoenix server via OTLP gRPC for UI visualization.
- When Phoenix is unreachable, traces are buffered to Parquet files at `/tmp/phoenix_traces/`.
- Parquet files can be uploaded to S3 via `PhoenixAdapter.upload_parquet_to_s3()`.

#### CSV writer utility

`src/eval_harness/reporting/csv_writer.py` provides a reusable `write_results()` function that creates DataFrames with canonical column ordering and writes to CSV.

#### HTML report generation

`src/eval_harness/reporting/html_summary.py` generates static HTML reports with:
- Overall pass rate and total evaluations
- Mean/min/max score statistics
- Per-question breakdown table

---

## 11. Comparison

Comparison refers to the ability to compare evaluation results across runs, system variants, and against baselines.

### What was built

#### Regression checking

`src/eval_harness/reporting/regression_check.py` implements `check_regression()`:

1. Loads current and baseline JSON results.
2. For each `blocker`-severity metric, compares current score against baseline.
3. Flags any metric where the relative decrease exceeds a threshold (default 5%).
4. Raises `RuntimeError` with details if regressions are detected.

```python
# Usage
check_regression(
    current_results=Path("results/latest_summary.json"),
    baseline_path=Path("results/baseline_summary.json"),
    threshold=0.05,  # 5% relative decrease
)
```

#### Multi-variant comparison via A/B runs

Since each evaluation run produces an independent CSV+JSON pair, comparing system variants is straightforward:

1. Run evaluation with variant A (`--parser fast`).
2. Run evaluation with variant B (`--parser docling`).
3. Compare the `metrics_avg` fields from the two JSON summaries.

The JSON summaries include all metadata needed for fair comparison (dataset, parser, timestamp, top_k, slice).

#### RAGAS diagnostic matrix

The `docs/legal-rag-bench-guide.md` provides a diagnostic matrix for interpreting RAGAS metric combinations:

| Faithfulness | Context Precision | Context Recall | Answer Relevancy | Diagnosis |
|---|---|---|---|---|
| High | Low | Low | Low | Poor retrieval — fix retrieval depth or embedding quality |
| High | Low | High | Low | Noisy retrieval — fix ranking/reranking |
| High | High | High | Low | Generator avoids hallucination but is unhelpful — improve generation prompt |
| Low | High | High | High | Hallucination — fix generation grounding |

#### Phoenix trace comparison

When Phoenix is enabled, traces from different runs can be compared in the Phoenix UI, grouped by `session_id` (which includes run name and timestamp).

#### Example results

The `examples/example_results/` directory contains pre-computed summaries for reference:

- `rag_legalbench_nano_summary.json`
- `parsing_omnidocbench_docling_summary.json`
- `parsing_dp_bench_fast_summary.json`

---

## Architecture Summary

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
│  - fast (pypdf)  │              │  - custom (OpenSearch) │
│  - docling       │              │                        │
└────────┬────────┘              └───────────┬───────────┘
         │                                   │
┌────────▼───────────────────────────────────▼──────────────────┐
│                      Metrics Layer                             │
│  Parsing: NID, TEDS, MHS, ARD, BLEU, METEOR, TextF1, mAP     │
│  RAG:      RAGAS (Faithfulness, ContextPrecision/Recall,      │
│            AnswerRelevancy) + deterministic (hit@k, latency)   │
└────────┬───────────────────────────────────┬──────────────────┘
         │                                   │
┌────────▼────────┐              ┌───────────▼───────────┐
│  CSV + JSON      │              │  Phoenix Observability │
│  (incremental)   │              │  (OTLP spans/traces)   │
└────────┬────────┘              └───────────────────────┘
         │
┌────────▼────────┐
│  Regression      │
│  Check + HTML    │
│  Reports         │
└─────────────────┘
```

### Key design patterns

1. **Adapter pattern**: Decouples user systems from the framework. Users implement a function with a defined signature; the adapter wraps it with schema validation.
2. **Schema contracts**: JSON Schema (Draft 2020-12) defines all data interfaces (`parser_output`, `rag_query_output`, `eval_questions`). Runtime validation via `jsonschema`.
3. **Iterator pattern**: Datasets are loaded lazily — constant memory regardless of size.
4. **Incremental writes**: Results flushed to CSV per-query — crash-safe, progress-visible.
5. **Shared embedder**: A single embedder instance is shared between RAG retrieval and RAGAS AnswerRelevancy to avoid duplicate model loads.
6. **Phoenix context managers**: OpenTelemetry span hierarchy managed via Python context managers for proper parent-child relationships.
