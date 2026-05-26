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

### RAG Evaluation - Legal RAG Bench

| File | Dataset | Slice | Description |
|------|---------|-------|-------------|
| `rag_legal_rag_bench_nano.csv` | Legal RAG Bench | nano (10 queries) | RAG evaluation results with DeepEval metrics |
| `rag_legal_rag_bench_nano_summary.json` | - | - | Metrics summary |
| `rag_legal_rag_bench_nano_details.json` | - | - | Detailed LLM judge reasoning for each query |

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

### Legal RAG Bench CSV Columns

```
query_id, question, gold_answer, generated_answer, relevant_passage_retrieved,
faithfulness_score, context_precision_score, context_recall_score,
answer_relevancy_score, judge_verdict, total_ms, error, framework_version,
metric_computation_time_ms, llm_judge_model
```

- `query_id`: Question identifier (1-N)
- `question`: The legal question
- `gold_answer`: Reference answer from Legal RAG Bench dataset
- `generated_answer`: RAG system's generated answer
- `relevant_passage_retrieved`: Boolean - was gold standard passage found in retrieval?
- `faithfulness_score`: 0.0-1.0 - factual consistency with retrieved context
- `context_precision_score`: 0.0-1.0 - relevant chunks ranked higher than irrelevant?
- `context_recall_score`: 0.0-1.0 - did retrieval find all info needed for gold answer?
- `answer_relevancy_score`: 0.0-1.0 - does response directly address the question?
- `judge_verdict`: PASS/FAIL - overall LLM judgment
- `total_ms`: End-to-end latency per query
- `error`: Error message if any
- `framework_version`: DeepEval version (e.g., "4.0.3")
- `metric_computation_time_ms`: Time spent computing all DeepEval metrics
- `llm_judge_model`: Model used for LLM-as-judge (e.g., "gpt-4o")

## JSON Summary Format

```json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "timestamp": "20260521_210254",
  "csv_file": "legal_rag_bench_nano_results_20260521_210254.csv",
  "metrics_avg": {
    "relevant_passage_retrieved": 0.2,
    "faithfulness_score": 1.0,
    "context_precision_score": 0.2367,
    "context_recall_score": 0.4,
    "answer_relevancy_score": 0.8024,
    "total_ms": 3289.7862
  },
  "total_processed": 10,
  "errors": 0,
  "top_k": 5,
  "evaluation_framework": "deepeval",
  "framework_version": "4.0.3",
  "judge_model": "gpt-4o",
  "phoenix": {
    "enabled": true,
    "trace_count": 10,
    "endpoint": "http://localhost:6006"
  }
}
```

## JSON Details Format

The `rag_legal_rag_bench_nano_details.json` file contains detailed LLM judge reasoning for each metric:

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
      "query_id": 1,
      "question": "...",
      "reasoning": {
        "faithfulness": {
          "reason": "Explanation of faithfulness score",
          "claims": [...],
          "truths": [...]
        },
        "context_precision": {
          "reason": "Explanation of precision score",
          "verdicts": [...]
        },
        "context_recall": {
          "reason": "Explanation of recall score",
          "verdicts": [...]
        },
        "answer_relevancy": {
          "reason": "Explanation of relevancy score"
        }
      }
    }
  ]
}
```

## Metric Interpretation

### Faithfulness
- **1.0**: All claims supported by context (no hallucination)
- **0.5-1.0**: Most claims supported (minor hallucination)
- **<0.5**: Significant hallucination

### Context Precision
- **>0.7**: Relevant chunks ranked at top (good retrieval)
- **0.3-0.7**: Mixed relevance (re-ranking needed)
- **<0.3**: Poor ranking (retriever needs improvement)

### Context Recall
- **>0.7**: All key information found (excellent coverage)
- **0.3-0.7**: Most information found (acceptable)
- **<0.3**: Critical information missing (retrieval insufficient)

### Answer Relevancy
- **>0.8**: Direct, complete answer (excellent)
- **0.5-0.8**: Partially addresses question (acceptable)
- **<0.5**: Evasive or irrelevant (poor generation)

### Relevant Passage Retrieved
- **1.0 (True)**: Gold standard passage found in retrieval
- **0.0 (False)**: Gold passage not found

## Example Score Analysis

From the nano results:
- **Faithfulness: 1.0** - Perfect, no hallucinations across all queries
- **Context Precision: 0.24** - Poor, only 24% of retrieved chunks relevant on average
- **Context Recall: 0.40** - Fair, 40% of gold answer information found
- **Answer Relevancy: 0.80** - Good, answers mostly address questions
- **Passage Retrieval: 20%** - Only 2/10 queries found the gold passage

This pattern suggests: **Good generator, poor retriever**. The system answers truthfully based on what it retrieves, but retrieval misses critical information.
