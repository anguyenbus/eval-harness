"""
OpenSearch RAG Adapter Example for eval-harness.

This example shows how to integrate your OpenSearch-backed RAG system
with eval-harness for evaluation on LegalBench-RAG benchmark.

Usage:
    1. Configure your OpenSearch connection (OPENSEARCH_ENDPOINT, etc.)
    2. Run: python examples/opensearch_rag_adapter.py

Requirements:
    - pip install opensearch-py requests-aws4auth boto3
    - OpenSearch cluster with existing vector index
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


# =============================================================================
# CONFIGURATION - Update these for your environment
# =============================================================================

OPENSEARCH_ENDPOINT = os.getenv(
    "OPENSEARCH_ENDPOINT",
    "https://search-your-domain.us-east-1.es.amazonaws.com",
)

OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "legal_chunks")

# AWS Region (for IAM auth) or set username/password for basic auth
OPENSEARCH_REGION = os.getenv("OPENSEARCH_REGION", "us-east-1")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

# Field names in your OpenSearch documents
VECTOR_FIELD = os.getenv("VECTOR_FIELD", "embedding")
TEXT_FIELD = os.getenv("TEXT_FIELD", "content")
DOC_ID_FIELD = os.getenv("DOC_ID_FIELD", "document_id")
CHAR_SPAN_FIELD = os.getenv("CHAR_SPAN_FIELD", "char_span")

# Embedding model - must match what was used to build your index
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# LLM backend for answer generation
# - "template": Simple concatenation (default, for testing)
# - "bedrock": AWS Bedrock Anthropic Claude (requires AWS credentials)
# - "openai": OpenAI GPT models (requires OPENAI_API_KEY)
LLM_BACKEND = os.getenv("LLM_BACKEND", "template")

# Bedrock model ID (when LLM_BACKEND=bedrock)
# Options: anthropic.claude-3-5-sonnet-20241022-v2:0 (default)
#          anthropic.claude-3-5-haiku-20241022-v1:0 (fast, cheap)
#          anthropic.claude-3-opus-20240229-v1:0 (highest quality)
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
)


# =============================================================================
# OPENSEARCH CLIENT
# =============================================================================


def create_opensearch_client(
    endpoint: str = OPENSEARCH_ENDPOINT,
    region: str | None = OPENSEARCH_REGION,
    username: str | None = OPENSEARCH_USERNAME,
    password: str | None = OPENSEARCH_PASSWORD,
) -> OpenSearch:
    """Create OpenSearch client with appropriate authentication.

    Supports:
    - AWS IAM authentication (boto3 credentials)
    - HTTP Basic authentication
    - No authentication (development)

    Args:
        endpoint: OpenSearch cluster endpoint URL
        region: AWS region for IAM auth
        username: Basic auth username
        password: Basic auth password

    Returns:
        Configured OpenSearch client
    """
    if region:
        # AWS IAM authentication
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise ValueError(
                "AWS credentials not found. Run 'aws configure' "
                "or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )

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
            max_retries=3,
            retry_on_timeout=True,
        )

    elif username and password:
        # HTTP Basic authentication
        return OpenSearch(
            hosts=[endpoint],
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=True,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )

    else:
        # No authentication (development only!)
        return OpenSearch(
            hosts=[endpoint],
            use_ssl=False,
            verify_certs=False,
            timeout=30,
        )


# =============================================================================
# EMBEDDING
# =============================================================================


class Embedder:
    """Embedding model wrapper.

    Supports multiple backends. Use the same model that indexed your documents.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            if self.model_name.startswith("text-embedding"):
                # OpenAI
                from openai import OpenAI as OpenAIClient

                self._model = OpenAIClient()
                self._backend = "openai"

            elif self.model_name.startswith("cohere"):
                # Cohere
                import cohere

                self._model = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
                self._backend = "cohere"

            else:
                # Sentence Transformers (local)
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._backend = "sentence_transformers"

        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if hasattr(self, "_backend"):
            if self._backend == "openai":
                response = self._model.embeddings.create(
                    input=text,
                    model=self.model_name,
                )
                return response.data[0].embedding

            elif self._backend == "cohere":
                response = self._model.embed(text, model=self.model_name)
                return response.embeddings[0]

        # Default: sentence_transformers
        return self.model.encode(text).tolist()


# Global embedder instance (cached)
_embedder = None


def get_embedder() -> Embedder:
    """Get or create global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


# =============================================================================
# LLM GENERATION
# =============================================================================


class BedrockLLM:
    """AWS Bedrock LLM wrapper for answer generation.

    Supports Anthropic Claude models via Bedrock.
    Uses the same AWS credentials as OpenSearch IAM auth.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = OPENSEARCH_REGION,
        max_tokens: int = 1024,
    ):
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy-load Bedrock client."""
        if self._client is None:
            import json

            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
            self._json = json
        return self._client

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        """Generate answer using Bedrock Anthropic Claude.

        Args:
            question: User question
            retrieved_chunks: Retrieved chunks with 'text' field

        Returns:
            Generated answer text
        """
        # Build context
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "")
            doc_id = chunk.get("doc_id", f"doc_{i}")
            context_parts.append(f"[{doc_id}] {text}")

        context = "\n\n".join(context_parts)

        # Bedrock request for Claude
        body = self._json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "messages": [{
                "role": "user",
                "content": f"""You are a legal assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Provide a concise answer:"""
            }]
        })

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
            )

            response_body = self._json.loads(response["body"].read().decode())
            return response_body["content"][0]["text"]

        except Exception as e:
            # Fallback to template on error
            return f"Error generating answer: {e}. Context preview: {context[:200]}..."


# Global LLM instance (cached)
_llm = None
_llm_backend = os.getenv("LLM_BACKEND", "template")  # template, bedrock, openai


def get_llm() -> BedrockLLM | None:
    """Get or create global LLM instance.

    Returns None if using template backend (default).
    """
    global _llm
    if _llm is None and _llm_backend == "bedrock":
        _llm = BedrockLLM(model_id=BEDROCK_MODEL_ID)
    return _llm


# =============================================================================
# QUERY FUNCTION
# =============================================================================


def query_opensearch(
    question: str,
    corpus_dir: Path,  # Required by adapter interface, unused for OpenSearch
    endpoint: str = OPENSEARCH_ENDPOINT,
    index_name: str = OPENSEARCH_INDEX,
    region: str | None = OPENSEARCH_REGION,
    username: str | None = OPENSEARCH_USERNAME,
    password: str | None = OPENSEARCH_PASSWORD,
    vector_field: str = VECTOR_FIELD,
    text_field: str = TEXT_FIELD,
    doc_id_field: str = DOC_ID_FIELD,
    char_span_field: str | None = CHAR_SPAN_FIELD,
    top_k: int = 5,
) -> dict[str, Any]:
    """Query OpenSearch index for RAG evaluation.

    This is the main integration point between your OpenSearch RAG system
    and eval-harness.

    Args:
        question: User question to answer
        corpus_dir: Required by adapter interface (unused here)
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

    Returns:
        Dict conforming to eval-harness rag_query_output schema:
        {
            "answer": {"text": str, "answer_supported": bool, "citations": [...]},
            "retrieved_chunks": [{"chunk_id": str, "score": float, "char_span": [int, int]}, ...],
            "timings_ms": {"retrieval": int, "generation": int, "total": int}
        }
    """
    # 1. Create client
    client = create_opensearch_client(endpoint, region, username, password)

    # 2. Embed question
    embedder = get_embedder()
    query_vector = embedder.embed(question)

    # 3. Search OpenSearch
    start_retrieval = time.time()

    try:
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
    except Exception as e:
        return {
            "answer": {"text": "", "answer_supported": False, "citations": []},
            "retrieved_chunks": [],
            "timings_ms": {"retrieval": 0, "generation": 0, "total": 0},
            "error": str(e),
        }

    retrieval_ms = int((time.time() - start_retrieval) * 1000)

    # 4. Parse results into eval-harness format
    retrieved_chunks = []
    for hit in response["hits"]["hits"]:
        chunk = {
            "chunk_id": hit["_id"],
            "score": float(hit["_score"]),
        }

        # Add char_span if available (for recall@k calculation)
        source = hit.get("_source", {})
        if char_span_field and char_span_field in source:
            span = source[char_span_field]
            if isinstance(span, list) and len(span) == 2:
                chunk["char_span"] = [int(span[0]), int(span[1])]

        # Store metadata for LLM context building
        if doc_id_field in source:
            chunk["doc_id"] = source[doc_id_field]
        chunk["text"] = source.get(text_field, "")

        retrieved_chunks.append(chunk)

    # 5. Generate answer
    start_generation = time.time()

    llm = get_llm()

    if llm is not None:
        # Use configured LLM (Bedrock, OpenAI, etc.)
        generated_answer = llm.generate(question, retrieved_chunks)
    else:
        # Template fallback (default for testing)
        context_texts = [c.get("text", "") for c in retrieved_chunks]
        context = "\n\n".join(context_texts)
        generated_answer = f"Based on {len(retrieved_chunks)} retrieved chunks: {context[:500]}..."

    generation_ms = int((time.time() - start_generation) * 1000)

    # 6. Build citations (all retrieved chunks)
    citations = [
        {"chunk_ids": [c["chunk_id"] for c in retrieved_chunks]}
    ]

    return {
        "answer": {
            "text": generated_answer,
            "answer_supported": True,  # TODO: LLM-as-judge or heuristic
            "citations": citations,
        },
        "retrieved_chunks": retrieved_chunks,
        "timings_ms": {
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "total": retrieval_ms + generation_ms,
        },
    }


# =============================================================================
# EVAL-HARNESS INTEGRATION
# =============================================================================


def main() -> None:
    """Run eval-harness with OpenSearch RAG adapter."""
    from eval_harness.adapters.rag_adapter import RagAdapter
    from eval_harness.config import load_config
    from eval_harness.datasets import load_legalbench_rag
    from eval_harness.runners.run_rag_eval import _calculate_recall_at_k, _token_f1

    print("Loading config...")
    config = load_config("eval_config.yaml")

    # Create adapter with OpenSearch configuration
    print(f"Connecting to OpenSearch: {OPENSEARCH_ENDPOINT}")
    print(f"Index: {OPENSEARCH_INDEX}")

    def opensearch_wrapper(question: str, corpus_dir: Path) -> dict:
        return query_opensearch(
            question=question,
            corpus_dir=corpus_dir,
            top_k=5,
        )

    adapter = RagAdapter(query_callable=opensearch_wrapper)

    # Load dataset
    corpus_dir = Path(config["datasets"]["legalbench_rag"]["path"])
    print(f"Loading dataset from: {corpus_dir}")

    dataset = load_legalbench_rag(corpus_dir, slice="nano")

    # Run evaluation
    print(f"\nEvaluating {len(list(dataset))} queries...\n")

    results = []
    for idx, (query_id, query_text, gold_spans, gold_answer) in enumerate(dataset, 1):
        print(f"[{idx}] {query_text[:50]}...")

        output = adapter.query(query_text, corpus_dir)

        # Check for errors
        if "error" in output:
            print(f"  ERROR: {output['error']}")
            continue

        # Calculate metrics
        recall_metrics = _calculate_recall_at_k(
            gold_spans, output["retrieved_chunks"]
        )
        f1_metrics = _token_f1(gold_answer, output["answer"]["text"])

        result = {
            "query_id": query_id,
            "recall_at_k": recall_metrics["recall_at_k"],
            "precision_at_k": recall_metrics["precision_at_k"],
            "f1_score": f1_metrics["f1"],
            "retrieval_ms": output["timings_ms"]["retrieval"],
            "generation_ms": output["timings_ms"]["generation"],
        }
        results.append(result)

        print(f"  Recall@5: {result['recall_at_k']:.4f}, F1: {result['f1_score']:.4f}")

    # Print summary
    if results:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        avg_recall = sum(r["recall_at_k"] for r in results) / len(results)
        avg_precision = sum(r["precision_at_k"] for r in results) / len(results)
        avg_f1 = sum(r["f1_score"] for r in results) / len(results)
        avg_retrieval = sum(r["retrieval_ms"] for r in results) / len(results)
        avg_generation = sum(r["generation_ms"] for r in results) / len(results)

        print(f"Avg Recall@5:   {avg_recall:.4f}")
        print(f"Avg Precision@5: {avg_precision:.4f}")
        print(f"Avg F1 Score:   {avg_f1:.4f}")
        print(f"Avg Retrieval:  {avg_retrieval:.0f}ms")
        print(f"Avg Generation: {avg_generation:.0f}ms")
        print(f"Avg Total:      {avg_retrieval + avg_generation:.0f}ms")


if __name__ == "__main__":
    main()
