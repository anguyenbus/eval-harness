# eval-harness

Evaluation framework for document parsing and RAG systems. Supports deterministic metrics, LLM-as-judge evaluation (RAGAS), and reproducible baseline comparisons using public benchmarks (OmniDocBench, DP-Bench, Legal RAG Bench).

## Quick Start

```bash
# Install dependencies
uv sync

# Set API keys for RAGAS evaluation
export OPENAI_API_KEY=sk-...
export HF_TOKEN=...  # For Legal RAG Bench (optional if ~/.huggingface/token exists)

# Prepare Legal RAG Bench corpus
uv run python scripts/prepare_legal_rag_bench_corpus.py

# Run parsing evaluation
uv run eval-parsing --dataset dp_bench --parser fast

# Run RAG evaluation (RAGAS LLM-judge metrics)
uv run eval-rag --slice nano

# View results
cat results/*.csv
```

## First-Time Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Set API Keys

For RAGAS evaluation (LLM-as-a-judge metrics):

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
      judge_model: gpt-4o
      judge_model_provider: openai
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

Evaluate RAG systems on legal reasoning questions using RAGAS LLM-judge metrics.

### Dataset: Legal RAG Bench

**Source**: isaacus/legal-rag-bench (HuggingFace)

- **100 questions** from Victorian Criminal Charge Book
- **4,876 passages** in corpus
- **Domain**: Criminal law, jury procedures, evidence law
- **Reasoning intensity**: Multi-step legal reasoning required

### Slices

| Slice | Questions | Use Case |
|-------|-----------|----------|
| `nano` | 10 | Quick testing, development |
| `full` | 100 | Complete evaluation |

### Commands

```bash
# Nano slice (10 questions, default top-k=5)
uv run eval-rag --slice nano

# Full evaluation with custom retrieval depth
uv run eval-rag --slice full --top-k 10

# Force re-ingestion of corpus into ChromaDB
uv run eval-rag --slice nano --force-reingest
```

**Note:** The stub option (ChromaDB-based reference implementation) is used by default. To evaluate your own RAG system, implement a custom query function and integrate it via the `RagAdapter`.

### RAGAS Metrics

| Metric | Component | Description |
|--------|-----------|-------------|
| **Faithfulness** | Generator | Factual consistency of generated answer vs retrieved context |
| **Context Precision** | Retriever | Signal-to-noise ratio in retrieved chunks (ranking quality) |
| **Context Recall** | Retriever | Coverage of relevant information in retrieved chunks |
| **Answer Relevancy** | End-to-end | Directness of response to original question |

All metrics use LLM-as-a-judge (gpt-4o) for evaluation. See [docs/legal-rag-bench-guide.md](docs/legal-rag-bench-guide.md) for detailed explanations and examples.

### Additional Metrics

- **Relevant Passage Retrieved** - Binary: was the gold passage retrieved?
- **Latency** - Total query time (ms)

## Results

Results written to `results/` with timestamp:

```
results/legal_rag_bench_nano_results_20260520_223534.csv
results/legal_rag_bench_full_results_20260520_224102.csv
```

CSV format: one row per query, all metrics as columns. Files append incrementally for real-time progress visibility.

### Example: RAG Evaluation Output

```csv
query_id,question,gold_answer,generated_answer,relevant_passage_retrieved,faithfulness_score,context_precision_score,context_recall_score,answer_relevancy_score,judge_verdict,total_ms
q_0,"Bob and Ted...","No. While the bench book...","I don't have enough...",True,1.0,0.25,0.333,0.0,NEEDS_REVIEW,1523
q_1,"What is the burden...","The prosecution bears...","The prosecution must...",False,0.85,0.60,0.75,0.92,PASS,1845
```

## Configuration

Edit `eval_config.yaml` for dataset paths and model settings:

```yaml
datasets:
  legal_rag_bench:
    path: data/rag/legal_rag_bench/corpus_files
    cache_path: data/rag/legal_rag_bench
    k_values: [5, 10, 20]
    ragas:
      judge_model: gpt-4o
      judge_model_provider: openai
      temperature: 0

  omnidocbench:
    path: /path/to/omnidocbench

  dp_bench:
    path: /path/to/dp_bench
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
│   └── legal-rag-bench-guide.md
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
│   │   └── ragas_adapter.py  # RAGAS LLM-judge wrapper
│   ├── datasets/           # Benchmark loaders
│   │   ├── legal_rag_bench.py  # isaacus/legal-rag-bench (HF)
│   │   ├── omnidocbench.py
│   │   └── dp_bench.py
│   ├── metrics/            # ALL evaluation metrics
│   │   ├── parsing/        # NID, TEDS, MHS, BLEU, METEOR
│   │   └── ragas_config.py  # RAGAS LLM/embedding backends
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
│           ├── embedder.py
│           ├── generator.py
│           └── ingestion.py
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
- `ragas>=0.2` - LLM-as-a-judge metrics
- `sentence-transformers` - Semantic embeddings
- `chromadb` - Vector store (stub implementation)
- `openai` - LLM judge (set `OPENAI_API_KEY` in `.env`)
- `datasets` - HuggingFace dataset loader

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

- **Guide**: [docs/guides/opensearch-integration.md](docs/guides/opensearch-integration.md)
- **Example**: [examples/opensearch_rag_adapter.py](examples/opensearch_rag_adapter.py)

Other vector stores (Pinecone, pgvector, Weaviate) follow the same adapter pattern.

## Documentation

- [Legal RAG Bench: Comprehensive Guide](docs/legal-rag-bench-guide.md) - Dataset structure, RAGAS metrics explained, score interpretation
- [Design Documents](docs/design/) - Architecture, data flow, and schema design
- [OpenSearch Integration Guide](docs/guides/opensearch-integration.md) - Complete walkthrough for OpenSearch users
- [Parser Output Schema Explained](docs/guides/parser-output-schema-explained.md) - Why the universal schema exists
- [Schema Alignment Guide](docs/guides/schema-alignment-guide.md) - OmniDocBench → eval-harness mapping
- [Custom Dataset Guide](docs/guides/custom-dataset-guide.md) - Integrate your own data
- [Contracts README](contracts/README.md) - Schema documentation
