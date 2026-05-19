# OpenSearch RAG Integration Guide

Complete guide for evaluating RAG systems backed by OpenSearch vector indexes using eval-harness.

## Overview

eval-harness is an evaluation framework — it does not provide RAG implementations. You bring your own RAG system (OpenSearch, Pinecone, pgvector, etc.) and integrate via the `RagAdapter` interface.

This guide shows how to integrate an OpenSearch-backed RAG system for evaluation on benchmarks like LegalBench-RAG.

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   eval-harness  │      │  Your Adapter    │      │   OpenSearch    │
│   (framework)   │◄────►│   (your code)    │◄────►│   (your index)  │
│                 │      │                  │      │                 │
│ - datasets      │      │ - query()        │      │ - vectors       │
│ - metrics       │      │ - auth           │      │ - metadata      │
│ - runners       │      │ - embeddings     │      │ - k-NN search   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

**Key point:** eval-harness never touches your OpenSearch instance directly. It only calls your adapter's `query()` function.

## Prerequisites

- OpenSearch cluster (AWS OpenSearch Service or self-hosted)
- Existing vector index with embeddings
- Python 3.13+
- Dependencies: `pip install -r requirements.txt` (eval-harness), `opensearch-py` (your code)

## Step-by-Step Integration

### Step 1: Install eval-harness

```bash
git clone <your-repo>
cd eval-harness
uv sync
```

### Step 2: Implement Your Query Function

Create `my_opensearch_rag.py`:

```python
"""OpenSearch-backed RAG query function for eval-harness evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3


def create_opensearch_client(
    endpoint: str,
    region: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> OpenSearch:
    """Create OpenSearch client with appropriate authentication.

    Args:
        endpoint: OpenSearch cluster endpoint (e.g., https://search-xxx.us-east-1.es.amazonaws.com)
        region: AWS region (for IAM auth)
        username: Basic auth username
        password: Basic auth password

    Returns:
        Configured OpenSearch client
    """
    if region:
        # AWS IAM authentication
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            "es",  # OpenSearch Service
            session_token=credentials.token,
        )
        return OpenSearch(
            hosts=[endpoint],
            http_auth=auth,
            connection_class=RequestsHttpConnection,
            use_ssl=True,
            verify_certs=True,
            timeout=30,
        )
    elif username and password:
        # HTTP Basic authentication
        return OpenSearch(
            hosts=[endpoint],
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=True,
            timeout=30,
        )
    else:
        # No authentication (development only)
        return OpenSearch(
            hosts=[endpoint],
            use_ssl=False,
            verify_certs=False,
            timeout=30,
        )


def query_opensearch(
    question: str,
    corpus_dir: Path,  # Ignored for OpenSearch (index is pre-built)
    endpoint: str,
    index_name: str,
    region: str | None = None,
    username: str | None = None,
    password: str | None = None,
    vector_field: str = "embedding",
    text_field: str = "text",
    doc_id_field: str = "doc_id",
    char_span_field: str | None = None,
    top_k: int = 5,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Query OpenSearch index for RAG evaluation.

    Args:
        question: User question to answer
        corpus_dir: Unused (required by adapter interface)
        endpoint: OpenSearch endpoint URL
        index_name: Index name to query
        region: AWS region for IAM auth
        username: Basic auth username
        password: Basic auth password
        vector_field: Field name containing embeddings
        text_field: Field name containing text content
        doc_id_field: Field name containing document ID
        char_span_field: Field name containing character span [start, end)
        top_k: Number of chunks to retrieve
        embedding_model: Embedding model identifier (if using API-based embedder)

    Returns:
        Dict conforming to rag_query_output schema:
        {
            "answer": {"text": str, "answer_supported": bool, "citations": [...]},
            "retrieved_chunks": [{"chunk_id": str, "score": float, "char_span": [int, int]}, ...],
            "timings_ms": {"retrieval": int, "generation": int, "total": int}
        }
    """
    import time

    # 1. Create client
    client = create_opensearch_client(endpoint, region, username, password)

    # 2. Embed question (YOUR embedding logic here)
    # Example with sentence-transformers:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embedder.encode(question).tolist()

    # Or use your existing embedder:
    # query_vector = your_embedder.embed(question)

    # 3. Search OpenSearch
    start_retrieval = time.time()

    response = client.search(
        index=index_name,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    vector_field: {
                        "vector": query_vector,
                        "k": top_k,
                    }
                }
            },
        },
    )

    retrieval_ms = int((time.time() - start_retrieval) * 1000)

    # 4. Parse results into eval-harness format
    retrieved_chunks = []
    for hit in response["hits"]["hits"]:
        chunk = {
            "chunk_id": hit["_id"],
            "score": hit["_score"],
        }

        # Add char_span if available in your index
        if char_span_field and char_span_field in hit["_source"]:
            chunk["char_span"] = hit["_source"][char_span_field]

        retrieved_chunks.append(chunk)

    # 5. Generate answer (YOUR generation logic here)
    start_generation = time.time()

    # Example: simple template (replace with your LLM call)
    context = "\n".join([hit["_source"][text_field] for hit in response["hits"]["hits"]])
    generated_answer = f"Based on the retrieved context: {context[:200]}..."

    # Or use your LLM:
    # generated_answer = your_llm.generate(question, retrieved_chunks)

    generation_ms = int((time.time() - start_generation) * 1000)

    return {
        "answer": {
            "text": generated_answer,
            "answer_supported": True,  # YOUR: LLM-as-judge or heuristic
            "citations": [{"chunk_ids": [c["chunk_id"] for c in retrieved_chunks]}],
        },
        "retrieved_chunks": retrieved_chunks,
        "timings_ms": {
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "total": retrieval_ms + generation_ms,
        },
    }
```

### Step 3: Wrap with RagAdapter

Create `my_eval.py`:

```python
"""Run eval-harness with OpenSearch-backed RAG."""

from pathlib import Path

from eval_harness.adapters.rag_adapter import RagAdapter
from eval_harness.config import load_config
from eval_harness.datasets import load_legalbench_rag
from eval_harness.runners.run_rag_eval import _calculate_recall_at_k, _token_f1

from my_opensearch_rag import query_opensearch


def main() -> None:
    # Load config
    config = load_config("eval_config.yaml")

    # Create adapter with your OpenSearch configuration
    def opensearch_wrapper(question: str, corpus_dir: Path) -> dict:
        return query_opensearch(
            question=question,
            corpus_dir=corpus_dir,
            endpoint="https://search-xxx.us-east-1.es.amazonaws.com",
            index_name="legal_chunks",
            region="us-east-1",  # Or use username/password
            top_k=5,
        )

    adapter = RagAdapter(query_callable=opensearch_wrapper)

    # Load dataset
    corpus_dir = Path(config["datasets"]["legalbench_rag"]["path"])
    dataset = load_legalbench_rag(corpus_dir, slice="nano")

    # Run evaluation
    results = []
    for query_id, query_text, gold_spans, gold_answer in dataset:
        output = adapter.query(query_text, corpus_dir)

        # Calculate metrics
        recall_metrics = _calculate_recall_at_k(gold_spans, output["retrieved_chunks"])
        f1_metrics = _token_f1(gold_answer, output["answer"]["text"])

        results.append({
            "query_id": query_id,
            "recall_at_k": recall_metrics["recall_at_k"],
            "f1_score": f1_metrics["f1"],
            # ... other metrics
        })

    # Print summary
    avg_recall = sum(r["recall_at_k"] for r in results) / len(results)
    avg_f1 = sum(r["f1_score"] for r in results) / len(results)
    print(f"Recall@5: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
```

### Step 4: Run Evaluation

```bash
python my_eval.py
```

Or use the CLI if you've registered your adapter:

```bash
# (Advanced) Add your adapter to run_rag_eval.py --rag option
eval-rag --dataset legalbench_rag --slice nano --rag opensearch
```

## Configuration Patterns

### AWS OpenSearch Service (IAM Auth)

```python
def query_opensearch(question: str, corpus_dir: Path) -> dict:
    return query_opensearch_impl(
        question=question,
        endpoint="https://search-xxx.us-east-1.es.amazonaws.com",
        index_name="my-index",
        region="us-east-1",  # Triggers IAM auth
        # Credentials from boto3 Session automatically
    )
```

### Self-hosted OpenSearch (Basic Auth)

```python
def query_opensearch(question: str, corpus_dir: Path) -> dict:
    return query_opensearch_impl(
        question=question,
        endpoint="https://opensearch.example.com",
        index_name="my-index",
        username="admin",
        password="secret",
    )
```

### Development (No Auth)

```python
def query_opensearch(question: str, corpus_dir: Path) -> dict:
    return query_opensearch_impl(
        question=question,
        endpoint="http://localhost:9200",
        index_name="my-index",
        # No auth params
    )
```

## Schema Mapping

Your OpenSearch documents should contain:

| Field | Required | Purpose | Eval-harness Mapping |
|-------|----------|---------|---------------------|
| `_id` | Yes | Unique chunk identifier | `chunk_id` |
| `_score` | Auto | k-NN similarity score | `score` |
| `<text_field>` | Yes | Text content | Used for LLM context |
| `<char_span_field>` | No | Character span in source doc | `char_span` (for recall@k) |
| `<doc_id_field>` | No | Source document ID | Tracking |

**Configurable field names** — you specify which field to use for text, char_span, etc.

## Common Scenarios

### Scenario 1: Different Embedding Models

Your OpenSearch index uses a specific embedding model. Match it:

```python
# If your index uses OpenAI embeddings:
from openai import OpenAI
client = OpenAI()
query_vector = client.embeddings.create(
    input=question,
    model="text-embedding-3-small"
).data[0].embedding

# If your index uses Cohere:
import cohere
co = cohere.Client("api-key")
query_vector = co.embed(question, model="embed-english-v3.0").embeddings[0]

# If your index uses AWS Bedrock:
import boto3
bedrock = boto3.client("bedrock-runtime")
# ... Bedrock embedding call
```

**Critical:** Use the same embedding model that indexed your documents, or recall@k will be meaningless.

### Scenario 2: Complex Document Structure

If your chunks have metadata (doc_id, page_num, section):

```python
retrieved_chunks = []
for hit in response["hits"]["hits"]:
    chunk = {
        "chunk_id": hit["_id"],
        "score": hit["_score"],
    }

    # Preserve metadata for your LLM context building
    metadata = {
        "doc_id": hit["_source"].get("doc_id"),
        "page_num": hit["_source"].get("page_num"),
        "section": hit["_source"].get("section"),
    }
    chunk["metadata"] = metadata

    # Add char_span for eval-harness metrics
    if "char_span" in hit["_source"]:
        chunk["char_span"] = hit["_source"]["char_span"]

    retrieved_chunks.append(chunk)
```

### Scenario 3: AWS Bedrock Anthropic Integration

When using AWS OpenSearch Service, you'll likely also use AWS Bedrock for LLM generation. This is the most common pattern for AWS-based RAG systems.

**Why Bedrock for Anthropic:**
- Existing AWS credential chain (no separate API keys)
- VPC endpoint privacy (no internet gateway needed)
- Centralized governance and cost management
- Native AWS integration

#### Implementation

```python
import json
import boto3
from typing import Any

# Bedrock runtime client (uses same credentials as OpenSearch)
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")


def generate_answer_bedrock(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    max_tokens: int = 1024,
) -> str:
    """Generate answer using AWS Bedrock Anthropic Claude.

    Args:
        question: User question
        retrieved_chunks: Chunks retrieved from OpenSearch
        model_id: Bedrock model ID
        max_tokens: Maximum tokens in response

    Returns:
        Generated answer text
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        text = chunk.get("text", "")
        doc_id = chunk.get("doc_id", "unknown")
        context_parts.append(f"[Document {doc_id}, Part {i}] {text}")

    context = "\n\n".join(context_parts)

    # Claude via Bedrock uses Anthropic message format
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": f"""You are a legal assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        }]
    })

    response = bedrock.invoke_model(
        modelId=model_id,
        body=body,
    )

    response_body = json.loads(response["body"].read().decode())
    return response_body["content"][0]["text"]
```

#### Available Bedrock Anthropic Models

| Model ID | Use Case | Context Window |
|----------|----------|----------------|
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | Balanced quality/speed | 200K tokens |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | Fast, cost-effective | 200K tokens |
| `anthropic.claude-3-opus-20240229-v1:0` | Highest quality | 200K tokens |

#### Cross-Region Considerations

**OpenSearch and Bedrock regions must match** or you'll incur data transfer costs:

```python
# Correct: Same region
opensearch_endpoint = "https://search-xxx.us-east-1.es.amazonaws.com"
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Wrong: Different regions (data transfer + latency)
# opensearch_endpoint = "https://search-xxx.us-east-1.es.amazonaws.com"
# bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
```

#### VPC Endpoints (Production)

For production AWS deployments, use VPC endpoints to keep traffic private:

```python
import boto3

# Bedrock runtime VPC endpoint
# https://docs.aws.amazon.com/bedrock/latest/userguide/vpc-endpoints.html
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    # Uses VPC endpoint if configured in your VPC
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
)
```

#### Prompt Template for Legal RAG

Legal Q&A benefits from structured prompts:

```python
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "system": """You are a legal research assistant. Your role is to answer questions based on provided legal documents.

Guidelines:
- Answer ONLY using the provided context
- If the answer is not in the context, say "I cannot answer from the provided documents"
- Cite specific documents when possible
- Be precise and avoid speculation""",
    "messages": [{
        "role": "user",
        "content": f"""<documents>
{context}
</documents>

<question>
{question}
</question>

Provide a concise answer based on the documents above:"""
    }]
})
```

#### Streaming Responses (Optional)

For real-time feedback, use streaming:

```python
def generate_answer_stream(question: str, retrieved_chunks: list) -> str:
    """Generate answer with streaming (real-time token generation)."""
    context = "\n\n".join([c.get("text", "") for c in retrieved_chunks])

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        }]
    })

    response = bedrock.invoke_model_with_response_stream(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=body,
    )

    full_answer = ""
    for event in response["body"]:
        chunk = event.get("chunk")
        if chunk:
            delta = json.loads(chunk.get("bytes").decode())
            if delta["type"] == "content_block_delta":
                text = delta["delta"].get("text", "")
                full_answer += text
                # print(text, end="", flush=True)  # Optional: real-time output

    return full_answer
```

#### Cost Estimation

Bedrock pricing varies by model. Approximate costs for LegalBench-RAG:

| Model | Input per 1K tokens | Output per 1K tokens | Est. cost per 100 queries |
|-------|-------------------|---------------------|---------------------------|
| Haiku | $0.00025 | $0.00125 | ~$0.15 |
| Sonnet | $0.00125 | $0.005 | ~$0.60 |
| Opus | $0.0075 | $0.015 | ~$2.00 |

*Nano slice (48 queries) × average 2K tokens context + 200 tokens answer*

#### Error Handling

```python
def generate_answer_with_fallback(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[str, bool, str | None]:
    """Generate answer with fallback on errors.

    Returns:
        (answer_text, success, error_message)
    """
    try:
        answer = generate_answer_bedrock(question, retrieved_chunks)
        return answer, True, None

    except bedrock.exceptions.ThrottlingException:
        # Rate limit - implement retry or fallback
        return "", False, "Bedrock rate limit exceeded"

    except bedrock.exceptions.AccessDeniedException:
        # IAM permission issue
        return "", False, "Bedrock access denied. Check model permissions."

    except bedrock.exceptions.ValidationException as e:
        # Invalid request
        return "", False, f"Bedrock validation error: {e}"

    except Exception as e:
        return "", False, f"Unexpected error: {e}"
```

#### IAM Permissions Required

Your IAM role/user needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
      ]
    }
  ]
}
```

For development, you can use wildcard (`"*"`) for resource, but production should specify exact model IDs.

## Metrics Explained

eval-harness calculates:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Recall@k** | Gold span overlaps any retrieved chunk | Did we retrieve relevant evidence? |
| **Precision@k** | Relevant chunks / k | How precise was retrieval? |
| **F1 Score** | Token overlap between gold and generated answer | Answer quality |
| **Exact Match** | Exact string match | Strict correctness |
| **Answer Supported** | LLM judgment or heuristic | Does answer cite evidence? |
| **Citation Precision** | Valid citations / total citations | Citation accuracy |

## Troubleshooting

### Low Recall@k

- Check embedding model matches index
- Verify `top_k` is sufficient (try 10, 20)
- Inspect query vector vs document vectors
- Check OpenSearch k-NN configuration

### Connection Errors

```python
# Debug connection
client = create_opensearch_client(endpoint, region)
print(client.info())  # Should return cluster info
```

### Timeout Errors

```python
# Increase timeout
return OpenSearch(
    hosts=[endpoint],
    timeout=60,  # Increase from 30
    # ...
)
```

### Auth Errors (AWS)

```python
# Verify credentials
import boto3
session = boto3.Session()
print(session.get_credentials().access_key)  # Should not be None
```

## Advanced Topics

### Batch Query Optimization

Reduce OpenSearch round-trips:

```python
# Query multiple questions at once
def batch_query_opensearch(questions: list[str]) -> list[dict]:
    # Use msearch or bulk queries
    # Return list of results
    pass
```

### Hybrid Search (Vector + Keyword)

```python
response = client.search(
    index=index_name,
    body={
        "query": {
            "hybrid": {
                "queries": [
                    {"knn": {vector_field: {"vector": query_vector, "k": top_k}}},
                    {"match": {text_field: question}}
                ]
            }
        }
    },
)
```

### Filter by Metadata

```python
response = client.search(
    index=index_name,
    body={
        "query": {
            "bool": {
                "must": [
                    {"knn": {vector_field: {"vector": query_vector, "k": top_k}}}
                ],
                "filter": [
                    {"term": {"doc_type": "contract"}},
                    {"range": {"date": {"gte": "2023-01-01"}}}
                ]
            }
        }
    },
)
```

## Next Steps

1. Implement `query_opensearch()` with your credentials and index details
2. Wrap with `RagAdapter` in `my_eval.py`
3. Run on `nano` slice first (48 queries) to validate
4. Scale to `mini` or `full` slices once validated

## Resources

- [OpenSearch Python Client](https://opensearch.org/docs/latest/clients/python/)
- [AWS OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/)
- [eval-harness README](../../README.md)
- [LegalBench-RAG Paper](https://arxiv.org/abs/2308.07924)
