# eval-harness

Evaluation framework for document parsing and RAG systems. Supports deterministic metrics, LLM-as-judge evaluation (DeepEval), reproducible baseline comparisons using public benchmarks (OmniDocBench, DP-Bench, Legal RAG Bench), and synthetic span generation for replay evaluation.

## Quick Start

```bash
# Install dependencies
uv sync

# (Optional) Install Phoenix for observability
uv sync --all-extras

# (Optional) Install replay service dependencies
uv sync --extra replay

# Set API keys for DeepEval evaluation
export OPENAI_API_KEY=sk-...
export HF_TOKEN=...  # For Legal RAG Bench (optional if ~/.huggingface/token exists)

# Prepare Legal RAG Bench corpus
uv run python scripts/prepare_legal_rag_bench_corpus.py

# Run parsing evaluation
uv run eval-parsing --dataset dp_bench --parser fast

# Run RAG evaluation (DeepEval LLM-judge metrics)
# Pico slice = 2 questions (fastest for testing)
uv run eval-rag --slice pico --rag stub-local

# Run RAG evaluation with Phoenix Native mode
uv run eval-rag --slice pico --rag stub-local --enable-phoenix --mode native

# Generate baseline spans for replay evaluation (stores metrics in Phoenix)
uv run generate-spans --limit=10

# Verify Phoenix connectivity before running evaluations
uv run eval-harness-check phoenix
uv run eval-harness-check config  # Display resolved configuration

# Run replay evaluation comparing chunking strategies
uv run eval-replay --candidate=stub-chunks-512-overlap-150 --baseline=stub-local --limit=10

# Run with production baseline (no re-evaluation)
uv run eval-replay --candidate=stub-chunks-512-overlap-150 --production-baseline --output results/replay.json

# View results (creates CSV + 2 JSON files)
cat results/replay*.csv
cat results/replay_summary.json
cat results/replay_details.json
```

## First-Time Setup

### 1. Install Dependencies

```bash
# Core dependencies
uv sync

# Optional: Phoenix observability
uv sync --all-extras

# Optional: Replay service
uv sync --extra replay
```

### 2. Set API Keys

For DeepEval evaluation (LLM-as-a-judge metrics):

```bash
export OPENAI_API_KEY=sk-...
# Or add to .env file:
echo "OPENAI_API_KEY=sk-..." > .env
```

For Legal RAG Bench dataset (HuggingFace):

```bash
export HF_TOKEN=hf_...
# Or create ~/.huggingface/token
```

### 3. Prepare Legal RAG Bench Corpus

```bash
# Downloads corpus from HuggingFace and exports to text files for ChromaDB
uv run python scripts/prepare_legal_rag_bench_corpus.py
```

This downloads the isaacus/legal-rag-bench corpus (4,876 passages) and prepares it for the stub RAG implementation.

### 4. Configure (Optional)

Edit `eval_config.yaml` for dataset paths and model settings:

```yaml
datasets:
  legal_rag_bench:
    path: data/rag/legal_rag_bench/corpus_files
    cache_path: data/rag/legal_rag_bench
    k_values: [5, 10, 20]
    ragas:
      judge_model: gpt-4o-mini
      judge_model_provider: openai

generator:
  phoenix_endpoint: ${PHOENIX_ENDPOINT:-http://localhost:6006}
  project_name: case-assistant-synthetic
  default_limit: 100
```

## Parsing Evaluation

Evaluate document parsers on layout-aware benchmarks.

### Datasets

| Dataset | Description | Documents |
|---------|-------------|-----------|
| `omnidocbench` | Multi-modal, English-only (filtered from 1,651 → 593 pages) | 593 pages |
| `dp_bench` | Digital PDF benchmark | 1,052 docs |

### Parsers

| Parser | Description |
|--------|-------------|
| `stub` | Stub implementation for testing |
| `fast` | pypdf - fast digital PDF parsing |
| `docling` | Full OCR pipeline with layout analysis |

### Commands

```bash
uv run eval-parsing --dataset omnidocbench --parser fast
uv run eval-parsing --dataset dp_bench --parser fast
uv run eval-parsing --dataset omnidocbench --parser stub --limit 10
uv run eval-parsing --dataset dp_bench --parser fast --output-dir ./my_results
```

### Metrics

- **NID/NID-S** - Normalized Indel Distance (text similarity, with/without tables)
- **TEDS/TEDS-S** - Tree Edit Distance Similarity (table structure)
- **MHS/MHS-S** - Markdown Hierarchical Similarity (heading structure)
- **ARD** - Average Rank Distance (reading order)
- **BLEU** - Token-level n-gram overlap
- **METEOR** - Harmonic mean of precision/recall with stemming

## RAG Evaluation

Evaluate RAG systems on legal reasoning questions using DeepEval LLM-judge metrics.

### Dataset: Legal RAG Bench

**Source**: isaacus/legal-rag-bench (HuggingFace)

- **100 questions** from Victorian Criminal Charge Book
- **4,876 passages** in corpus
- **Domain**: Criminal law, jury procedures, evidence law
- **Reasoning intensity**: Multi-step legal reasoning required

### Slices

| Slice | Questions | Use Case |
|-------|-----------|----------|
| `pico` | 2 | Fastest testing, quick iteration |
| `nano` | 10 | Quick testing, development |
| `full` | 100 | Complete evaluation |

### Commands

```bash
# Pico slice (2 questions, fastest)
uv run eval-rag --slice pico --rag stub-local

# Nano slice (10 questions)
uv run eval-rag --slice nano --rag stub-local

# Full evaluation with custom retrieval depth
uv run eval-rag --slice full --rag stub-local --top-k 10

# Force re-ingestion of corpus into ChromaDB
uv run eval-rag --slice nano --rag stub-local --force-reingest
```

**Note:** `--rag` is required. The `stub-local` option uses a ChromaDB-based reference implementation for demonstration. To evaluate your own RAG system, implement a custom query function and integrate it via the `RagAdapter`.

### DeepEval Metrics

| Metric | Component | Description |
|--------|-----------|-------------|
| **Faithfulness** | Generator | Factual consistency of generated answer vs retrieved context |
| **Context Precision** | Retriever | Signal-to-noise ratio in retrieved chunks (ranking quality) |
| **Context Recall** | Retriever | Coverage of relevant information in retrieved chunks |
| **Answer Relevancy** | End-to-end | Directness of response to original question |

All metrics use LLM-as-a-judge (gpt-4o-mini) for evaluation. See [docs/legal-rag-bench-guide.md](docs/legal-rag-bench-guide.md) for detailed explanations and examples.

### Additional Metrics

- **Relevant Passage Retrieved** - Binary: was the gold passage retrieved?
- **Latency** - Total query time (ms)

## Phoenix Integration

eval-harness supports two modes of Phoenix integration for RAG evaluation:

### Spans Mode (Default)

Creates nested spans showing each stage of RAG evaluation in Phoenix UI.

```bash
# Start Phoenix server
python -m phoenix.server.main serve

# Run with spans mode (default when --enable-phoenix is set)
uv run eval-rag --slice pico --rag stub-local --enable-phoenix
```

**Span Hierarchy:**
```
rag_evaluation (CHAIN)
├── retriever.query (RETRIEVER)
│   └── embed.query (EMBEDDING)
└── generator.generate (LLM)
```

### Native Mode (Recommended)

Uses Phoenix's Experiment API for structured evaluation with built-in visualizations.

```bash
# Start Phoenix server
python -m phoenix.server.main serve

# Run with Native mode
uv run eval-rag --slice pico --rag stub-local --enable-phoenix --mode native
```

**Native Mode Benefits:**
- **Experiments UI**: Dedicated view for comparing runs side-by-side
- **Richer metadata**: Evaluator explanations stored per-example
- **Failed-only verdicts**: Only stores detailed verdicts when metric fails (~80% storage savings)
- **Better traces**: Suppresses DeepEval's internal OpenAI traces to reduce noise

**Evaluator Return Format:**
```python
{
    "score": 0.85,  # Float: 0-1
    "label": "faithful",  # Str: human-readable label
    "explanation": "Good answer...",  # Str: LLM judge reasoning
    "metadata": {
        "threshold": 0.5,
        "success": True,
        "model": "gpt-4o-mini",
        "evaluation_cost": 0.001,
        "verdicts": [...],  # Only when success=False
    },
}
```

### Graceful Degradation

When `--enable-phoenix` is **NOT** set:
- Evaluation runs normally without Phoenix dependency
- Results still saved to CSV with all metrics
- No tracing overhead

When `--enable-phoenix` IS set but Phoenix is unavailable:
- Spans mode: Falls back to Parquet buffer
- Native mode: Raises clear error message

## Replay Evaluation

**Purpose**: Compare RAG system configurations using cached baseline scores and statistical testing.

### Why Replay Evaluation Matters

**Problem**: RAG systems have many tuning knobs (chunk size, overlap, embedding model, top-k). Testing each configuration requires:
1. Running queries against your vector store
2. Computing LLM-as-judge metrics ($$$)
3. Comparing results to determine if the change helped

**Replay Solution**: Generate baseline spans **once**, store all metrics in Phoenix, then compare any number of candidates against that baseline without re-running the expensive LLM judge.

**Benefits**:
- **Cost**: Reuse stored metrics (no repeated GPT-4 API calls)
- **Speed**: Baseline scores loaded from Phoenix (instant vs minutes)
- **Consistency**: Same baseline for all comparisons
- **Statistical rigor**: Paired Wilcoxon test + effect sizes

### Available Backends

| Backend | Port | Chunk Size | RAM Usage | Speed | Notes |
|---------|------|------------|-----------|-------|-------|
| **Zvec** | 8082 | 512 | Low (in-process) | Fast | Primary backend. SQLite-like vector DB. |

**ChromaDB removed**: Too RAM-heavy for evaluation environment. Zvec recommended for all tests.

---

## Step 1: Generate Baseline Spans

**Goal**: Create pre-evaluated question set with stored metrics in Phoenix.

```bash
# Terminal 1: Start Phoenix (required for span storage)
uv run python -m phoenix.server.main serve

# Terminal 2: Generate baseline spans (run once per baseline configuration)
# CPU-only mode enforced automatically (CUDA_VISIBLE_DEVICES="")
uv run generate-spans --limit=3
```

**What happens**:
1. Loads questions from Legal RAG Bench (or your configured dataset)
2. Runs stub RAG pipeline: retrieval → generation
3. Evaluates each question with DeepEval metrics (GPT-4o LLM judge)
4. Stores everything in Phoenix as span attributes:
   - `input.value`: Question text
   - `output.value`: Generated answer
   - `rag_faithfulness`: Factual consistency score (0-1)
   - `rag_context_precision`: Retrieval signal-to-noise (0-1)
   - `rag_context_recall`: Relevant info coverage (0-1)
   - `rag_answer_relevancy`: Response directness (0-1)
   - `rag_latency_total_ms`: Query time in milliseconds

**What to check**:
- Phoenix UI shows spans at `http://localhost:6006`
- Project name: `case-assistant-synthetic`
- Each span has `rag_faithfulness` attribute with value 0-1

**No baseline spans?**
```bash
# Troubleshooting
uv run eval-harness-check phoenix  # Verify connectivity
uv run eval-harness-check config    # Verify dataset paths
```

---

## Step 2: Run Candidate Evaluation

**Goal**: Compare a new RAG configuration against the cached baseline.

### 2a. Start Candidate Service (Terminal 2)

```bash
# Start Zvec-backed RAG service (CPU-only mode enforced)
uv run python -m eval_harness.stubs.service \
    --config configs/stubs/chunking-512.yaml \
    --port 8082
```

**`--export-spans false` is set in config**: Candidate does not create replay spans in Phoenix.

Verify service is healthy:
```bash
curl http://localhost:8082/health
# Should return: {"status": "healthy"}
```

### 2b. Run Evaluation (Terminal 3)

```bash
uv run eval-replay --candidate-spec configs/candidates/stub-zvec-8082.yaml \
    --production-baseline \
    --limit 3 \
    --output results/comparison.json
```

**What happens**:
1. Queries Phoenix for baseline spans (loads stored scores)
2. Runs candidate service via HTTP:
   - `POST http://localhost:8082/query` (Zvec backend)
   - Payload: `{"question": "...", "top_k": 5}`
   - Candidate returns `retrieved_contexts`, `response`, `timings_ms`
3. Evaluates candidate responses with DeepEval (GPT-4o)
4. Compares candidate vs baseline using paired statistical tests

**What to check**:
- Output files created with timestamps:
  - `results/comparison_YYYYMMDD_HHMMSS_summary.json` - Averages and statistical tests
  - `results/comparison_YYYYMMDD_HHMMSS_details.json` - Per-question scores and contexts
  - `results/comparison_YYYYMMDD_HHMMSS.csv` - Metrics summary (spreadsheet-friendly)
  - `results/comparison_YYYYMMDD_HHMMSS_chunks.csv` - Retrieved contexts per question
- Console shows statistical comparison table with p-values and effect sizes

### 2c. Stop Service

```bash
# Kill the service after evaluation to free RAM
# Ctrl+C in the service terminal, or:
pkill -f "eval_harness.stubs.service"
```

---

## Understanding Results

### Summary JSON (`{output}_summary.json`)

High-level overview with averages and statistical tests:

```json
{
  "candidate": "stub-zvec-8082",
  "baseline": "stub-local",
  "num_questions": 10,
  "averages": {
    "candidate": {
      "faithfulness": 0.85,
      "context_precision": 0.72,
      "latency_total_ms": 1250
    },
    "baseline": {
      "faithfulness": 0.78,
      "context_precision": 0.65,
      "latency_total_ms": 1800
    }
  },
  "statistical_tests": {
    "faithfulness": {
      "p_value": 0.031,
      "effect_size": 0.45,
      "winner": "candidate",
      "pass_fail": true
    }
  }
}
```

**What to look for**:
- **p_value < 0.05**: Statistically significant difference
- **effect_size**: Magnitude of difference
  - `< 0.15`: Negligible
  - `0.15-0.33`: Small
  - `0.33-0.47`: Medium
  - `> 0.47`: Large
- **winner**: Which configuration performed better
- **pass_fail**: Combined metric (p-value + direction)

### Details JSON (`{output}_details.json`)

Per-question breakdown with retrieved chunks:

```json
{
  "questions": [
    {
      "question": "What is the burden of proof...",
      "baseline": {
        "scores": {"faithfulness": 0.9, "context_precision": 0.8},
        "retrieved_contexts": ["...", "..."],
        "response_text": "The prosecution bears..."
      },
      "candidate": {
        "scores": {"faithfulness": 0.85, "context_precision": 0.75},
        "retrieved_contexts": ["...", "...", "..."],
        "response_text": "In criminal cases..."
      }
    }
  ]
}
```

**What to check**:
- Retrieved contexts differ between baseline/candidate
- Response quality correlates with metric scores
- Latency differences align with expectations

### CSV (`{output}.csv`)

Spreadsheet format for regression tracking:

```csv
metric,candidate_avg,baseline_avg,p_value,effect_size,winner
faithfulness,0.85,0.78,0.031,0.45,candidate
context_precision,0.72,0.65,0.120,0.22,candidate
```

### Chunks CSV (`{output}_chunks.csv`)

Retrieved contexts per question for investigating retrieval differences:

```csv
question_id,question,source,chunk_index,chunk_text,score
q_0000,What is criminal law?,baseline,0,Criminal law is...,0.95
q_0000,What is criminal law?,baseline,1,The burden of proof...,0.87
q_0000,What is criminal law?,candidate,0,Criminal law defines...,0.92
```

**What to check**:
- Different chunk sizes affect retrieval (candidate returns more/fewer chunks)
- Chunk content overlap between baseline/candidate
- Score distributions (higher scores = better semantic match)

---

## Statistical Testing Explained

### Wilcoxon Signed-Rank Test

**Why**: Paired test—same questions evaluated on both configurations.

**Null hypothesis**: Both configurations perform identically.

**p-value interpretation**:
- `< 0.05`: Reject null—significant difference
- `≥ 0.05`: No conclusion—difference could be noise

**Example**:
```
p_value = 0.031  → 3.1% chance difference is from noise
p_value = 0.450  → 45% chance difference is from noise
```

### Cliff's Delta (Effect Size)

**Why**: p-values don't tell you "how much" better.

**Interpretation**:
| Range | Meaning |
|-------|---------|
| 0.0 - 0.147 | Negligible |
| 0.147 - 0.33 | Small |
| 0.33 - 0.474 | Medium |
| > 0.474 | Large |

**Example**:
```
effect_size = 0.15  → Small but meaningful improvement
effect_size = 0.50  → Large improvement (worth deploying)
```

### Decision Framework

| p-value | Effect Size | Action |
|----------|-------------|--------|
| ≥ 0.05 | Any | No conclusion—need more data |
| < 0.05 | < 0.15 | Significant but tiny—probably not worth it |
| < 0.05 | 0.15 - 0.33 | Small win—consider if low risk |
| < 0.05 | > 0.33 | Clear win—deploy |

---

## HTTP-Based Evaluation (Recommended)

**Why**: Run candidate as HTTP service, test without code changes.

### 1. Start Candidate Service

```bash
# Terminal 1: Start stub service with chunk_size=512, overlap=150
uv run python -m eval_harness.stubs.service \
    --config configs/stubs/chunking-512.yaml \
    --port 8082
```

**`--export-spans false`**: Critical for evaluation mode—prevents candidate from creating replay spans in Phoenix (keeps span collection clean).

### 2. Create Candidate Spec

`configs/candidates/my-experiment.yaml`:
```yaml
name: my-experiment
description: Testing chunk_size=256, overlap=25

candidate:
  service_url: http://localhost:8082/query
  service_version: "1.0.0"
  contract_version: "1.0"
  timeout_seconds: 30
  max_retries: 2
```

### 3. Run Evaluation

```bash
uv run eval-replay \
    --candidate-spec configs/candidates/my-experiment.yaml \
    --production-baseline \
    --output results/my_experiment.json
```

**Benefits**:
- Candidate can be in any language (Python, Go, Rust)
- Test deployed services (not just local code)
- Swap candidates without redeploying eval-harness

---

## Installation

```bash
# Core replay dependencies
uv sync --extra replay

# Full observability (optional)
uv sync --all-extras
```

---

## Documentation

- [docs/replay-service.md](docs/replay-service.md) - Architecture and implementation details
- [docs/http-contract.md](docs/http-contract.md) - HTTP service contract specification
- [docs/evaluation-system-walkthrough.md](docs/evaluation-system-walkthrough.md) - Complete evaluation system guide

## Phoenix Observability (Optional)

eval-harness integrates with [Arize Phoenix](https://docs.arize.com/phoenix) for RAG pipeline tracing and visualization.

### Installation

```bash
uv sync --all-extras  # Install phoenix dependencies
```

### Usage

```bash
# Start Phoenix server (runs on http://localhost:6006)
python -m phoenix.server.main serve

# Run evaluation with Phoenix spans mode (default)
uv run eval-rag --slice pico --rag stub-local --enable-phoenix

# Run evaluation with Phoenix Native mode (recommended)
uv run eval-rag --slice pico --rag stub-local --enable-phoenix --mode native

# Generate synthetic spans (auto-traces to Phoenix)
uv run generate-spans --limit=10

# Run replay evaluation (creates replay_* spans)
uv run eval-replay --candidate=stub-chunks-512-overlap-150 --baseline=stub-local

# View traces in browser
open http://localhost:6006
```

### Configuration

Edit `eval_config.yaml` for Phoenix settings:

```yaml
phoenix:
  enabled: true
  endpoint: http://localhost:6006
  export_path: /tmp/phoenix_traces
```

## Results

Results written to `results/` with timestamp:

```
results/legal_rag_bench_pico_results_20260526_223534.csv
results/legal_rag_bench_nano_results_20260526_224102.csv
results/legal_rag_bench_full_results_20260526_225000.csv
```

CSV format: one row per query, all metrics as columns. Files append incrementally for real-time progress visibility.

### Example: RAG Evaluation Output

```csv
query_id,question,gold_answer,generated_answer,relevant_passage_retrieved,faithfulness_score,context_precision_score,context_recall_score,answer_relevancy_score,judge_verdict,total_ms
q_0,"Bob and Ted...","No. While the bench book...","I don't have enough...",True,1.0,0.25,0.333,0.0,NEEDS_REVIEW,1523
q_1,"What is the burden...","The prosecution bears...","The prosecution must...",False,0.85,0.60,0.75,0.92,PASS,1845
```

## Project Structure

```
.
├── pyproject.toml          # Package configuration
├── uv.lock                 # Dependency lock file
├── eval_config.yaml        # Dataset paths and model settings
├── README.md
│
├── contracts/              # JSON Schema contracts
├── docs/                   # Documentation
│   ├── legal-rag-bench-guide.md
│   ├── evaluation-system-walkthrough.md
│   └── replay-service.md
│
├── scripts/                # Dataset utilities
│   ├── download_datasets.py
│   └── prepare_legal_rag_bench_corpus.py
│
├── src/eval_harness/
│   ├── __init__.py
│   ├── config.py           # Config loader
│   ├── adapters/           # User integrates their systems here
│   │   ├── parser_adapter.py
│   │   ├── rag_adapter.py
│   │   └── deepeval_adapter.py  # DeepEval LLM-judge wrapper
│   ├── datasets/           # Benchmark loaders
│   │   ├── legal_rag_bench.py  # isaacus/legal-rag-bench (HF)
│   │   ├── omnidocbench.py
│   │   └── dp_bench.py
│   ├── metrics/            # ALL evaluation metrics
│   │   ├── parsing/        # NID, TEDS, MHS, BLEU, METEOR
│   │   └── deepeval_config.py  # DeepEval LLM/embedding backends
│   ├── replay/             # Replay evaluation module
│   │   ├── phoenix_client.py
│   │   ├── corpus.py
│   │   ├── tasks.py
│   │   └── comparison.py
│   ├── runners/            # CLI entry points
│   │   ├── run_parsing_eval.py
│   │   ├── run_rag_eval.py
│   │   ├── generate_spans.py
│   │   └── run_replay_eval.py
│   ├── stubs/              # REFERENCE IMPLEMENTATIONS (demo only)
│   │   ├── parsing/
│   │   │   ├── stub_parser.py
│   │   │   ├── digital_pdf_parser.py
│   │   │   └── docling_parser.py
│   │   ├── rag/            # ChromaDB stub for demonstration
│   │   │   ├── chromadb_client.py
│   │   │   ├── chromadb_config.py
│   │   │   ├── chromadb_query.py
│   │   │   ├── chunker.py
│   │   │   ├── embedder.py
│   │   │   ├── generator.py
│   │   │   └── ingestion.py
│   │   └── span_generator/ # Synthetic span generation
│   │       ├── tracer.py
│   │       ├── span_schema.py
│   │       ├── loader.py
│   │       ├── runner.py
│   │       └── config.py
│   └── observability/      # Phoenix integration
│       ├── config.py
│       └── phoenix_adapter.py
│
├── tests/                  # Unit and integration tests
│
├── data/                   # Benchmark datasets (gitignored)
│   ├── parsing/
│   │   ├── omnidocbench_english/
│   │   └── dp_bench/
│   └── rag/
│       └── legal_rag_bench/
│           ├── corpus_files/  # 4,876 passages as .txt files
│           └── cache/         # HuggingFace cache
│
└── results/                # CSV outputs (gitignored)
```

**Note:** `stubs/` contain reference implementations for demo/testing only. Users integrate their own parsers and RAG systems via `adapters/`.

## Dependencies

**Parsing:**
- `pypdf` - Fast digital PDF parsing
- `docling` - OCR and layout analysis
- `sacrebleu` - BLEU score
- `nltk` - METEOR score

**RAG:**
- `deepeval` - LLM-as-a-judge metrics
- `sentence-transformers` - Semantic embeddings
- `chromadb` - Vector store (stub implementation)
- `openai` - LLM judge (set `OPENAI_API_KEY` in `.env`)
- `datasets` - HuggingFace dataset loader

**Observability (optional):**
- `arize-phoenix>=4.0.0` - RAG pipeline tracing and visualization
- `openinference-instrumentation-openai` - OpenAI instrumentation for DeepEval internal traces

**Replay (optional):**
- `arize-phoenix>=4.0.0` - Phoenix SDK for span export/query
- `openinference-semantic-conventions>=0.1.0` - OpenInference attribute constants

## Using Your Own RAG System

eval-harness is designed to evaluate **your** RAG system, not provide one. The ChromaDB stub in `stubs/rag/` is for demonstration only.

### Integration Pattern

1. Implement a query function with signature:
   ```python
   def query(question: str, corpus_dir: Path) -> dict:
       # Your RAG logic here
       return {
           "answer": {"text": "..."},
           "retrieved_chunks": [{"doc_id": "...", "score": 0.85, "text": "..."}],
           "timings_ms": {"retrieval": 50, "generation": 500, "total": 550},
       }
   ```

2. Wrap with `RagAdapter`:
   ```python
   from eval_harness.adapters.rag_adapter import RagAdapter

   adapter = RagAdapter(query_callable=query)
   ```

3. Run evaluation via Python script or extend CLI

### OpenSearch Integration

For teams with vectors stored on OpenSearch (AWS or self-hosted):

- **Guide**: [docs/guides/opensearch-integration.md](docs/guides/opensearch-integration.md)
- **Example**: [examples/opensearch_rag_adapter.py](examples/opensearch_rag_adapter.py)

Other vector stores (Pinecone, pgvector, Weaviate) follow the same adapter pattern.

## Documentation

- [Legal RAG Bench: Comprehensive Guide](docs/legal-rag-bench-guide.md) - Dataset structure, DeepEval metrics explained, score interpretation
- [Replay Service Documentation](docs/replay-service.md) - Synthetic span generation and replay evaluation
- [Evaluation System Walkthrough](docs/evaluation-system-walkthrough.md) - Complete guide to the evaluation system
- [Design Documents](docs/design/) - Architecture, data flow, and schema design
- [OpenSearch Integration Guide](docs/guides/opensearch-integration.md) - Complete walkthrough for OpenSearch users
- [Parser Output Schema Explained](docs/guides/parser-output-schema-explained.md) - Why the universal schema exists
- [Schema Alignment Guide](docs/guides/schema-alignment-guide.md) - OmniDocBench → eval-harness mapping
- [Custom Dataset Guide](docs/guides/custom-dataset-guide.md) - Integrate your own data
- [Contracts README](contracts/README.md) - Schema documentation
