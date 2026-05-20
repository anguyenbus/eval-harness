# Legal RAG Bench: Comprehensive Guide

## Table of Contents

1. [Overview](#overview)
2. [Dataset Structure](#dataset-structure)
3. [RAGAS Metrics Explained](#ragas-metrics-explained)
4. [Example Analysis](#example-analysis)
5. [Interpreting Scores](#interpreting-scores)
6. [Usage](#usage)

---

## Overview

**Legal RAG Bench** is a benchmark dataset designed for evaluating Retrieval-Augmented Generation (RAG) systems on legal reasoning tasks. It focuses on questions from the Victorian Criminal Charge Book, requiring deep understanding of legal procedures, jury instructions, and court protocols.

### Key Characteristics

| Attribute | Value |
|-----------|-------|
| **Source** | Victorian Criminal Charge Book |
| **Total Questions** | 100 (reasoning-intensive legal questions) |
| **Corpus Size** | 4,876 passages |
| **Domain** | Criminal law, jury procedures, evidence law |
| **Dataset Provider** | `isaacus/legal-rag-bench` (HuggingFace) |
| **License** | Academic/research use |

### Why Legal RAG Bench?

Legal RAG evaluation presents unique challenges:
- **Reasoning intensity**: Questions require multi-step legal reasoning
- **Precision matters**: Legal accuracy is critical; hallucinations unacceptable
- **Context dependence**: Answers must be grounded in specific legal passages
- **Expertise gap**: General LLMs lack specialized legal knowledge

---

## Dataset Structure

### Two Splits

Legal RAG Bench consists of two separate subsets:

#### 1. Corpus Split (Documents)
- **Purpose**: Source documents for retrieval
- **Size**: 4,876 passages
- **Fields**:
  - `id`: Unique passage identifier (e.g., "1.2-c2-s2")
  - `title`: Section heading
  - `text`: Full passage content
  - `footnotes`: Additional legal references

#### 2. QA Split (Questions)
- **Purpose**: Evaluation questions with reference answers
- **Size**: 100 questions
- **Fields**:
  - `id`: Unique question identifier
  - `question`: The legal question to answer
  - `answer`: Reference answer (ground truth)
  - `relevant_passage_id`: ID of the corpus passage containing the answer

### Slices for Evaluation

| Slice | Questions | Use Case |
|-------|-----------|----------|
| `nano` | 10 | Quick testing, development |
| `full` | 100 | Complete evaluation |

---

## RAGAS Metrics Explained

RAGAS (Retrieval Augmented Generation Assessment) uses LLM-as-a-judge to evaluate RAG systems across four dimensions:

### 1. Faithfulness

**Component Evaluated**: Generator (LLM)

**Definition**: Measures factual consistency of the generated response against retrieved context.

**How it works**:
1. Breaks the generated answer into individual claims
2. Uses an LLM judge to verify each claim against retrieved context
3. Calculates ratio of supported claims to total claims

**Formula**:
```
Faithfulness = (Claims supported by context) / (Total claims)
```

**Score range**: 0.0 to 1.0

**What it catches**: Hallucinations, invented facts, information not in context

---

### 2. Context Precision

**Component Evaluated**: Retriever

**Definition**: Evaluates retriever's ability to rank relevant chunks higher than irrelevant ones.

**How it works**:
1. Uses LLM to judge each retrieved chunk's relevance to the question
2. Calculates precision@k for each position
3. Computes mean precision weighted by relevance

**Formula**:
```
Context Precision@K = Σ(Precision@k × v_k) / (Total relevant items)

where Precision@k = (relevant items at rank k) / k
```

**Score range**: 0.0 to 1.0

**What it catches**: Poor ranking, noise in retrieval, irrelevant chunks at top positions

**Key insight**: Order matters! An irrelevant chunk at position 1 hurts more than at position 5.

---

### 3. Context Recall

**Component Evaluated**: Retriever

**Definition**: Measures how many relevant documents were successfully retrieved.

**How it works**:
1. Breaks the reference (gold) answer into claims
2. Uses LLM to check if each claim can be attributed to retrieved context
3. Calculates ratio of supported claims to total claims

**Formula**:
```
Context Recall = (Claims in reference supported by context) / (Total claims in reference)
```

**Score range**: 0.0 to 1.0

**What it catches**: Missing information, incomplete retrieval

**Requires**: Ground truth reference answer

---

### 4. Answer Relevancy

**Component Evaluated**: End-to-end (Retriever + Generator)

**Definition**: Measures how relevant the response is to the original question.

**How it works**:
1. Generates N artificial questions FROM the answer
2. Computes cosine similarity between generated questions and original question
3. Returns mean similarity score

**Formula**:
```
Answer Relevancy = (1/N) × Σ(cosine_similarity(generated_question_embedding, original_question_embedding))
```

**Score range**: 0.0 to 1.0

**What it catches**: Evasive answers, "I don't know" responses, incomplete answers, responses that don't address the question

**Note**: Uses embeddings model (sentence-transformers) for similarity calculation

---

## Example Analysis

Let's analyze a real example from Legal RAG Bench evaluation:

### The Question

> **Query 1**: "Bob and Ted are close friends. Ted is on trial for drug offences, and Bob has been selected as a juror in Ted's case. Is the judge required to excuse Bob from serving on the jury?"

### The Reference (Gold) Answer

> "No. While the bench book instructs judges to inform members of the jury panel that they can excuse themselves if they know the accused, this is not mandatory. Instead, the court may excuse a potential juror if they are satisfied that the person will not be able to consider the case impartially."

### The Generated Answer

> "I don't have enough information to answer this question. The provided context does not specifically address whether a close personal relationship, such as friendship, is a valid reason for excusing a juror. The context mentions that a court may excuse a potential juror if it is satisfied that the person will not be able to consider the case impartially or is unable to serve for any other reasons [1.2-c2-s2_chunk_00000]. However, it does not explicitly state whether a friendship with the accused would automatically disqualify someone from serving as a juror."

### The Scores

| Metric | Score | Analysis |
|--------|-------|----------|
| Faithfulness | 1.0 | Perfect - all claims supported |
| Context Precision | 0.25 | Poor - only 1/5 chunks relevant |
| Context Recall | 0.333 | Poor - only 1/3 gold claims found |
| Answer Relevancy | 0.0 | Failed - answer doesn't address question |
| Relevant Passage Retrieved | True | Correct passage found |

---

## Detailed Score Breakdown

### Faithfulness = 1.0 (Perfect)

The generated answer makes specific claims:
1. *"context does not specifically address whether friendship is a valid reason"* ✓
2. *"context mentions court may excuse if person can't consider impartially"* ✓
3. *"does not explicitly state whether friendship would disqualify"* ✓

All 3 claims are supported by the retrieved context. No hallucination.

```
Faithfulness = 3/3 = 1.0
```

**Why this matters**: The LLM is being honest about context limitations rather than inventing information.

---

### Context Precision = 0.25 (Poor)

With `top_k=5`, the retriever returned 5 chunks:
- Chunk 1: [1.2-c2-s2] Relevant ✓ (discusses juror impartiality)
- Chunk 2: [X.X-X-X] Not relevant ✗
- Chunk 3: [X.X-X-X] Not relevant ✗
- Chunk 4: [X.X-X-X] Not relevant ✗
- Chunk 5: [X.X-X-X] Not relevant ✗

```
Context Precision = 1/5 = 0.20
```

(The actual score is 0.25, suggesting the LLM judge found partial relevance in some chunks)

**Why this matters**: 80% of retrieved context is noise, wasting tokens and potentially confusing the generator.

---

### Context Recall = 0.333 (Poor)

The gold answer contains 3 key claims:
1. *"judges inform jurors they can excuse themselves if they know the accused"* ✗ Not in context
2. *"this is not mandatory"* ✗ Not in context
3. *"court may excuse if person can't consider impartially"* ✓ In context

```
Context Recall = 1/3 ≈ 0.333
```

**Why this matters**: Critical legal information is missing from retrieval, preventing complete answering.

---

### Answer Relevancy = 0.0 (Failed)

The answer "I don't have enough information" generates questions like:
- "What information is missing?"
- "Why is the context insufficient?"

These have low semantic similarity to the original: *"Is the judge required to excuse Bob?"*

```
cosine_similarity("What information is missing?", "Is judge required to excuse Bob?") ≈ 0.2
```

Average across N generated questions ≈ 0.0

**Why this matters**: While honest, the answer is unhelpful. A better system would either:
1. Retrieve better context (addressing precision/recall)
2. Synthesize an answer from partial information
3. Ask a clarifying question

---

## Interpreting Scores

### Score Ranges

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 0.9 - 1.0 | Excellent | No action needed |
| 0.7 - 0.9 | Good | Minor optimization |
| 0.5 - 0.7 | Fair | Investigate failure cases |
| 0.3 - 0.5 | Poor | Significant issues |
| 0.0 - 0.3 | Failed | System redesign needed |

### Metric Combinations

| Faithfulness | Context Precision | Context Recall | Diagnosis |
|--------------|-------------------|----------------|-----------|
| High | High | High | Optimal system |
| High | Low | High | Noisy retriever - filter results |
| High | High | Low | Insufficient retrieval - increase k |
| Low | High | High | Generator hallucinating - prompt/model issue |
| Low | Low | Low | Complete failure - redesign system |

### Common Patterns

#### Pattern 1: High Faithfulness, Low Answer Relevancy
```
Faithfulness: 1.0
Answer Relevancy: 0.1
```
**Cause**: Generator produces factually correct but unhelpful answers (e.g., "I don't know")
**Fix**: Improve retrieval (recall) or adjust generation prompt

#### Pattern 2: Low Context Precision, High Faithfulness
```
Context Precision: 0.2
Faithfulness: 0.9
```
**Cause**: Generator successfully ignores irrelevant context
**Fix**: Improve retriever ranking or use re-ranking

#### Pattern 3: High Context Precision, Low Context Recall
```
Context Precision: 0.9
Context Recall: 0.3
```
**Cause**: Retriever finds relevant chunks but misses key information
**Fix**: Increase retrieval depth (k) or improve embedding model

---

## Usage

### Installation

```bash
uv sync
```

### Running Evaluation

```bash
# Quick test (10 questions)
uv run eval-rag --slice nano

# Full evaluation (100 questions)
uv run eval-rag --slice full

# Custom retrieval depth
uv run eval-rag --slice full --top-k 10
```

### Configuration

Edit `eval_config.yaml`:

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
```

### Output Format

**CSV** (`legal_rag_bench_full_results_TIMESTAMP.csv`):
```csv
query_id,question,gold_answer,generated_answer,relevant_passage_retrieved,faithfulness_score,context_precision_score,context_recall_score,answer_relevancy_score,judge_verdict,total_ms,error
```

**JSON** (`legal_rag_bench_full_results_TIMESTAMP.json`):
```json
{
  "dataset": "legal_rag_bench",
  "slice": "full",
  "timestamp": "20260520_223534",
  "metrics_avg": {
    "faithfulness_score": 0.85,
    "context_precision_score": 0.42,
    "context_recall_score": 0.38,
    "answer_relevancy_score": 0.67
  },
  "total_processed": 100,
  "errors": 0
}
```

---

## References

- **Dataset**: [isaacus/legal-rag-bench](https://huggingface.co/datasets/isaacus/legal-rag-bench)
- **RAGAS Docs**: [https://docs.ragas.io](https://docs.ragas.io)
- **Paper**: Legal RAG Bench: A Dataset for Evaluating Legal RAG Systems

---

*Generated: 2025-05-20*
