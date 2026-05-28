# Example Evaluation Results

Sample output files from eval-harness runs, organized by evaluation type.

## Structure

```
example_results/
├── eval_parser/          # Parsing/DX extraction benchmarks
├── eval_rag/             # RAG evaluation (Legal RAG Bench)
├── eval_replay/          # Replay/caching evaluation (TODO)
└── README.md             # This file
```

## Parser Evaluation (`eval_parser/`)

Legal RAG Bench nano slice (10 queries) RAG evaluation with DeepEval metrics.

| File | Dataset | Parser | Description |
|------|---------|--------|-------------|
| `parsing_dp_bench_fast_summary.json` | DP-Bench | pypdf (fast) | Digital PDF parsing metrics summary |
| `parsing_omnidocbench_docling_summary.json` | OmniDocBench | docling | Multi-modal parsing metrics summary |

## RAG Evaluation (`eval_rag/`)

Legal RAG Bench nano slice (2 queries) with DeepEval metrics.

| File | Description |
|------|-------------|
| `rag_legal_rag_bench_nano_summary.json` | Metrics summary with experiment metadata |
| `rag_legal_rag_bench_nano_results.csv` | Per-query results with cost tracking |
| `rag_legal_rag_bench_nano_results.parquet` | Same data in Parquet format |

## Replay Evaluation (`eval_replay/`)

Placeholder for replay/caching evaluation examples.

## CSV Format

### Legal RAG Bench (v2)

```csv
experiment_id,query_id,question,gold_answer,generated_answer,
relevant_passage_retrieved,faithfulness_score,faithfulness_label,faithfulness_verdicts,
context_precision_score,context_precision_label,context_precision_verdicts,
context_recall_score,context_recall_label,context_recall_verdicts,
answer_relevancy_score,answer_relevancy_label,answer_relevancy_verdicts,
total_ms,error,app_cost_usd,judge_cost_usd,total_cost_usd,
judge_faithfulness_cost_usd,judge_context_precision_cost_usd,
judge_context_recall_cost_usd,judge_answer_relevancy_cost_usd
```

**Key columns:**
- `relevant_passage_retrieved`: Boolean (1/0) - was gold passage in retrieval?
- `faithfulness_score`: 0.0-1.0 - factual consistency with context
- `context_precision_score`: 0.0-1.0 - relevant chunks ranked higher?
- `context_recall_score`: 0.0-1.0 - all gold info found?
- `answer_relevancy_score`: 0.0-1.0 - response addresses question?
- `*_label`: Interpretive label (faithful/imprecise, relevant/irrelevant, etc.)
- `*_verdicts`: JSON array of per-chunk LLM verdicts
- `app_cost_usd`: Application/LLM generation cost
- `judge_cost_usd`: Total LLM-as-judge cost
- `total_cost_usd`: Combined cost
- `judge_*_cost_usd`: Per-metric judge cost breakdown

## JSON Summary Format

```json
{
  "experiment_name": "",
  "experiment_id": "RXhwZXJpbWVudDo5",
  "dataset_id": "RGF0YXNldDox",
  "metrics_avg": {
    "faithfulness_score": 1.0,
    "context_precision_score": 0.0,
    "context_recall_score": 0.5833,
    "answer_relevancy_score": 0.9
  },
  "total_processed": 2,
  "errors": 0
}
```

## Metric Interpretation

| Metric | Good (>0.7) | Fair (0.3-0.7) | Poor (<0.3) |
|--------|-------------|----------------|-------------|
| **Faithfulness** | No hallucination | Minor hallucination | Significant hallucination |
| **Context Precision** | Relevant chunks top-ranked | Mixed relevance | Poor ranking |
| **Context Recall** | All info found | Most info found | Critical info missing |
| **Answer Relevancy** | Direct, complete | Partially addresses | Evasive/irrelevant |
| **Passage Retrieved** | Gold passage found | - | Gold passage missed |

## Pattern Analysis

**Good generator, poor retriever:**
- High faithfulness + answer relevancy
- Low context precision + recall
- Low passage retrieval rate

**Poor generator, good retriever:**
- Low faithfulness
- High context precision + recall
- Hallucinations despite good context
