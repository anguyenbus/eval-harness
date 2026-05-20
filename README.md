# eval-harness

Evaluation framework for document parsing and RAG systems. Supports deterministic metrics, LLM-as-judge evaluation, and reproducible baseline comparisons using public benchmarks (OmniDocBench, DP-Bench, LegalBench-RAG).

## Quick Start

```bash
# Install dependencies
uv sync

# Download datasets
uv run python scripts/download_datasets.py --datasets all

# Set API key for RAG evaluation
export OPENAI_API_KEY=sk-...

# Run evaluations
uv run eval-parsing --dataset dp_bench --parser fast
uv run eval-rag --dataset legalbench_rag --slice nano

# View results
cat results/*.csv
```

## First-Time Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Download Datasets

**Parsing datasets (HuggingFace):**
```bash
uv run python scripts/download_datasets.py --datasets all

# Or specific dataset
uv run python scripts/download_datasets.py --datasets dp_bench
uv run python scripts/download_datasets.py --datasets omnidocbench
```

**RAG dataset (manual Dropbox download):**
```bash
uv run python scripts/download_datasets.py --datasets legalbench_rag
```

Follow the printed instructions:
1. Visit https://github.com/zeroentropy-cc/legalbenchrag
2. Download via Dropbox link in README
3. Extract to `data/rag/legalbench_rag/`

### 3. Configure Dataset Paths

Edit `eval_config.yaml` to match your downloaded dataset locations:

```yaml
datasets:
  omnidocbench:
    path: data/parsing/omnidocbench_english
  dp_bench:
    path: data/parsing/dp_bench
  legalbench_rag:
    path: data/rag/legalbench_rag
```

### 4. Set API Keys

For RAG evaluation, set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
# Or add to .env file:
echo "OPENAI_API_KEY=sk-..." > .env
```

### 5. Run Evaluations

See commands in [Parsing Evaluation](#parsing-evaluation) and [RAG Evaluation](#rag-evaluation) sections below.

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

Evaluate retrieval-augmented generation systems on legal Q&A.

### Dataset

| Slice | Queries | Description |
|-------|---------|-------------|
| `nano` | 48 | Quick test (12 queries × 4 corpora) |
| `mini` | 776 | Standard evaluation |
| `full` | 6,889 | Complete benchmark |

### Commands

```bash
# Nano slice (default top-k=5)
uv run eval-rag --dataset legalbench_rag --slice nano

# Mini slice with more retrieved chunks
uv run eval-rag --dataset legalbench_rag --slice mini --top-k 10

# Force re-ingestion of corpus into ChromaDB
uv run eval-rag --dataset legalbench_rag --slice mini --force-reingest

# Full evaluation
uv run eval-rag --dataset legalbench_rag --slice full
```

**Note:** The stub option (ChromaDB-based reference implementation) is used by default. To evaluate your own RAG system, implement a custom query function and integrate it via the `RagAdapter`.

### Metrics

- **Recall@k** - Gold span overlap with retrieved chunks
- **Precision@k** - Relevant chunks retrieved / k
- **F1 Score** - Token-level answer quality
- **Exact Match** - Exact string match
- **Answer Supported** - LLM judgment if answer cites retrieved context
- **Citation Precision** - Valid citations / total citations
- **Latency** - Retrieval, generation, total time (ms)

## Results

Results written to `results/` with timestamp:

```
results/dp_bench_fast_results_20260518_223805.csv
results/legalbench_rag_nano_results_20260518_224102.csv
```

CSV format: one row per document/query, all metrics as columns. Files append incrementally for real-time progress visibility.

### Example: Parsing Evaluation Output

**DP-Bench (digital PDFs):**
```csv
query_id,nid,nid_s,teds,teds_s,mhs,mhs_s,ard,bleu,meteor
dp_bench_001,0.852,0.871,0.742,0.768,0.910,0.925,0.125,0.623,0.541
dp_bench_002,0.891,0.905,0.801,0.822,0.945,0.951,0.089,0.701,0.612
```

**OmniDocBench (multi-modal, English-only):**
```csv
query_id,nid,nid_s,teds,teds_s,mhs,mhs_s,ard,bleu,meteor
omnidocbench_0,0.793,0.793,0.0,0.0,0.0,0.0,0.0,0.235,0.503
omnidocbench_1,0.852,0.852,0.0,0.0,0.0,0.0,0.666,0.396,0.622
omnidocbench_2,0.956,0.956,0.0,0.0,0.0,0.0,0.247,0.683,0.808
```

### Example: RAG Evaluation Output

```csv
query_id,recall_at_k,precision_at_k,f1_score,answer_supported,citation_precision,total_ms
legalbench_0,0.85,0.80,0.72,True,0.90,1523
legalbench_1,0.62,0.58,0.55,True,0.85,1845
legalbench_2,0.91,0.88,0.79,True,0.95,1398
```

## Dataset Acquisition

### Parsing Datasets (HuggingFace)

```bash
uv run python scripts/download_datasets.py --datasets all
uv run python scripts/download_datasets.py --datasets dp_bench
uv run python scripts/download_datasets.py --datasets omnidocbench
```

**Note**: OmniDocBench is automatically filtered to English-only during download (1,651 → 593 pages). Filter keeps: academic_literature, research_report, exam_paper, colorful_textbook, book, PPT2PDF.

Downloads to `data/parsing/`. Requires `huggingface_hub`.

### LegalBench-RAG (Manual Download)

```bash
uv run python scripts/download_datasets.py --datasets legalbench_rag
```

Follow the printed instructions:
1. Visit https://github.com/zeroentropy-cc/legalbenchrag
2. Download via Dropbox link in README
3. Extract to `data/rag/legalbench_rag/`

Expected structure:
```
data/rag/legalbench_rag/
├── corpus/
│   ├── contractnli/
│   ├── cuad/
│   ├── maud/
│   └── privacyqa/
└── queries/
    ├── legalbench_rag_test.json
    └── legalbench_rag_mini.json
```

## Configuration

Edit `eval_config.yaml` for dataset paths and model settings:

```yaml
datasets:
  omnidocbench:
    path: /path/to/omnidocbench
  dp_bench:
    path: /path/to/dp_bench
  legalbench_rag:
    path: /path/to/legalbench_rag
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
├── eval_questions/         # Declarative evaluation criteria
│
├── scripts/                # Dataset download utilities
│   └── download_datasets.py
│
├── src/eval_harness/
│   ├── __init__.py
│   ├── config.py           # Config loader
│   ├── adapters/           # User integrates their systems here
│   │   ├── parser_adapter.py
│   │   └── rag_adapter.py
│   ├── datasets/           # Benchmark loaders
│   │   ├── legalbench_rag.py
│   │   ├── omnidocbench.py
│   │   └── dp_bench.py
│   ├── metrics/            # ALL evaluation metrics
│   │   └── parsing/        # NID, TEDS, MHS, BLEU, METEOR
│   │       ├── nid.py
│   │       ├── teds.py
│   │       ├── mhs.py
│   │       ├── reading_order.py
│   │       ├── text_similarity.py
│   │       └── markdown_converter.py
│   ├── runners/            # CLI entry points
│   │   ├── run_parsing_eval.py
│   │   └── run_rag_eval.py
│   └── stubs/              # REFERENCE IMPLEMENTATIONS (demo only)
│       ├── parsing/
│       │   ├── stub_parser.py
│       │   ├── digital_pdf_parser.py
│       │   └── docling_parser.py
│       └── rag/            # ChromaDB stub for demonstration
│           ├── chromadb_client.py
│           ├── chromadb_config.py
│           ├── chromadb_query.py
│           ├── chunker.py
│           ├── citations.py
│           ├── embedder.py
│           ├── exceptions.py
│           ├── generator.py
│           ├── ingestion.py
│           ├── retriever.py
│           ├── schema_conformance.py
│           └── tracing.py
│
├── tests/                  # Unit and integration tests
│   └── stubs/              # Tests for reference implementations
│       └── rag/
│
├── data/                   # Benchmark datasets (gitignored)
│   ├── parsing/
│   │   ├── omnidocbench_english/
│   │   └── dp_bench/
│   └── rag/
│       └── legalbench_rag/
│
└── results/                # CSV outputs (gitignored)
```

**Note:** `stubs/` contain reference implementations for demo/testing only. Users integrate their own parsers and RAG systems via `adapters/`.

## Dependencies

- `pypdf` - Fast digital PDF parsing
- `docling` - OCR and layout analysis (optional)
- `sacrebleu` - BLEU score
- `nltk` - METEOR score
- `sentence-transformers` - Semantic embedding (all-MiniLM-L6-v2)
- `chromadb` - Vector store
- `openai` - LLM generation (set `OPENAI_API_KEY` in `.env`)

## Performance

- **Parsing**: ~250ms/doc (DP-Bench, fast parser, with all metrics)
- **RAG**: ~2-3s/query (nano slice, ChromaDB + gpt-4o)

Optimizations:
- Model caching (embedder, generator loaded once)
- Pre-embedding batch ingestion
- CSV append for real-time progress
- sacrebleu direct import (100x faster than HF wrapper)

## Using Your Own RAG System

eval-harness is designed to evaluate **your** RAG system, not provide one. The ChromaDB stub in `stubs/rag/` is for demonstration only.

### Integration Pattern

1. Implement a query function with signature:
   ```python
   def query(question: str, corpus_dir: Path) -> dict:
       # Your RAG logic here
       return {
           "answer": {"text": "...", "answer_supported": True, "citations": [...]},
           "retrieved_chunks": [{"chunk_id": "...", "score": 0.85, "char_span": [0, 100]}],
           "timings_ms": {"retrieval": 50, "generation": 500, "total": 550}
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

- **Guide**: [docs/guides/opensearch-integration.md](docs/guides/opensearch-integration.md) - Complete integration walkthrough
- **Example**: [examples/opensearch_rag_adapter.py](examples/opensearch_rag_adapter.py) - Working code template

Other vector stores (Pinecone, pgvector, Weaviate) follow the same adapter pattern.

### OmniDocBench Integration

For teams already using OmniDocBench format:

- **Guide**: [docs/guides/schema-alignment-guide.md](docs/guides/schema-alignment-guide.md) - Minimal integration effort (~40 min)
- **Example**: [examples/omnidocbench_adapter.py](examples/omnidocbench_adapter.py) - Adapter template

No data migration required — just add an adapter wrapper.

## Documentation

- [Design Documents](docs/design/) - Architecture, data flow, and schema design
- [OpenSearch Integration Guide](docs/guides/opensearch-integration.md) - Complete walkthrough for OpenSearch users
- [Parser Output Schema Explained](docs/guides/parser-output-schema-explained.md) - Why the universal schema exists
- [Schema Alignment Guide](docs/guides/schema-alignment-guide.md) - OmniDocBench → eval-harness mapping
- [FAQ: OmniDocBench Users](docs/guides/faq-omnidocbench-users.md) - Common questions
- [Custom Dataset Guide](docs/guides/custom-dataset-guide.md) - Integrate your own data
- [Contracts README](contracts/README.md) - Schema documentation
