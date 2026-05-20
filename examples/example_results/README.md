# Example Evaluation Results

Sample output files from eval-harness runs.

## Files

### Parsing Evaluation

| File | Dataset | Parser | Description |
|------|---------|--------|-------------|
| `parsing_dp_bench_fast.csv` | DP-Bench | pypdf (fast) | Digital PDF parsing results |
| `parsing_dp_bench_fast_summary.json` | - | - | Metrics summary |
| `parsing_omnidocbench_docling.csv` | OmniDocBench | docling | Multi-modal parsing results |
| `parsing_omnidocbench_docling_summary.json` | - | - | Metrics summary |

### RAG Evaluation

| File | Dataset | Slice | Description |
|------|---------|-------|-------------|
| `rag_legalbench_nano.csv` | LegalBench-RAG | nano (48 queries) | RAG evaluation results |
| `rag_legalbench_nano_summary.json` | - | - | Metrics summary |

## CSV Format

### Parsing CSV Columns

```
query_id, error, nid, nid_s, teds, teds_s, mhs, mhs_s, ard, bleu, meteor
```

- `query_id`: Document identifier
- `error`: Error message if parsing failed
- `nid`, `nid_s`: Normalized Indel Distance (all / sparse)
- `teds`, `teds_s`: Tree Edit Distance Similarity (all / sparse)
- `mhs`, `mhs_s`: Markdown Hierarchical Similarity (all / sparse)
- `ard`: Average Rank Distance (reading order)
- `bleu`: BLEU score (n-gram overlap)
- `meteor`: METEOR score (harmonic mean with stemming)

### RAG CSV Columns

```
query_id, question, gold_answer, generated_answer, answer_supported,
recall_at_k, precision_at_k, num_relevant_chunks, num_citations,
citation_precision, f1_score, exact_match, retrieval_ms, generation_ms,
total_ms, top_k, retrieved_chunk_ids, retrieved_scores, error
```

## JSON Summary Format

```json
{
  "dataset": "dataset_name",
  "parser" / "slice": "parser_or_slice_name",
  "timestamp": "YYYYMMDD_HHMMSS",
  "csv_file": "results_file_name.csv",
  "metrics_avg": {
    "metric_name": 0.85
  },
  "total_processed": 100,
  "errors": 0
}
```
