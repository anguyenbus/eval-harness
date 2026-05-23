"""
Span schema constants for OpenInference compliance.

This module provides a single source of truth for OpenInference attribute
names and vendor-prefixed attributes used in synthetic span generation.
"""

from openinference.semconv.trace import SpanAttributes

# Re-export OpenInference SpanAttributes for convenience
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
METADATA = SpanAttributes.METADATA
SESSION_ID = SpanAttributes.SESSION_ID

# EMBEDDING span attributes
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
# EMBEDDING_TEXT is a nested attribute, defined as constant string
EMBEDDING_TEXT = "embedding.embeddings.0.embedding.text"
EMBEDDING_VECTOR_DIM = "embedding.embeddings.0.embedding.vector_dim"

# RETRIEVER span attributes
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
# Retrieval document attribute keys (used as formatted strings)
RETRIEVAL_DOCUMENT_ID = "retrieval.documents.{i}.document.id"
RETRIEVAL_DOCUMENT_CONTENT = "retrieval.documents.{i}.document.content"
RETRIEVAL_DOCUMENT_SCORE = "retrieval.documents.{i}.document.score"

# LLM span attributes
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
# LLM message attribute keys
LLM_MESSAGE_ROLE = "role"
LLM_MESSAGE_CONTENT = "content"

# Vendor-prefixed queryable attributes (for Phoenix filtering)
EVAL_HARNESS_PREFIX = "eval_harness"
SYNTHETIC_MARKER = f"{EVAL_HARNESS_PREFIX}.synthetic"
SOURCE_DATASET = f"{EVAL_HARNESS_PREFIX}.source_dataset"
GENERATOR_VERSION = f"{EVAL_HARNESS_PREFIX}.generator_version"
CASE_ID = f"{EVAL_HARNESS_PREFIX}.case_id"
TENANT_ID_HASHED = f"{EVAL_HARNESS_PREFIX}.tenant_id_hashed"

# Vendor attribute values
GENERATOR_VERSION_VALUE = "0.1.0"
SOURCE_DATASET_VALUE = "legal-rag-bench"
SYNTHETIC_MARKER_VALUE = "true"

# Metadata JSON String attribute keys
# These are encoded as a JSON string in the canonical 'metadata' attribute
METADATA_KEY_SOURCE_QUESTION_ID = "source_question_id"
METADATA_KEY_EXPECTED_PASSAGE_ID = "expected_passage_id"
METADATA_KEY_EXPECTED_ANSWER = "expected_answer"
METADATA_KEY_SYNTHETIC_RUN_ID = "synthetic_run_id"
METADATA_KEY_STUB_PIPELINE_VERSION = "stub_pipeline_version"
METADATA_KEY_TENANT_ID_HASHED = "tenant_id_hashed"

__all__ = [
    # OpenInference re-exports
    "INPUT_VALUE",
    "OUTPUT_VALUE",
    "SPAN_KIND",
    "METADATA",
    "SESSION_ID",
    # EMBEDDING attributes
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_EMBEDDINGS",
    "EMBEDDING_TEXT",
    "EMBEDDING_VECTOR_DIM",
    # RETRIEVER attributes
    "RETRIEVAL_DOCUMENTS",
    "RETRIEVAL_DOCUMENT_ID",
    "RETRIEVAL_DOCUMENT_CONTENT",
    "RETRIEVAL_DOCUMENT_SCORE",
    # LLM attributes
    "LLM_MODEL_NAME",
    "LLM_INPUT_MESSAGES",
    "LLM_OUTPUT_MESSAGES",
    "LLM_TOKEN_COUNT_TOTAL",
    "LLM_MESSAGE_ROLE",
    "LLM_MESSAGE_CONTENT",
    # Vendor-prefixed queryable attributes
    "SYNTHETIC_MARKER",
    "SOURCE_DATASET",
    "GENERATOR_VERSION",
    "CASE_ID",
    "TENANT_ID_HASHED",
    # Vendor attribute values
    "GENERATOR_VERSION_VALUE",
    "SOURCE_DATASET_VALUE",
    "SYNTHETIC_MARKER_VALUE",
    # Metadata JSON keys
    "METADATA_KEY_SOURCE_QUESTION_ID",
    "METADATA_KEY_EXPECTED_PASSAGE_ID",
    "METADATA_KEY_EXPECTED_ANSWER",
    "METADATA_KEY_SYNTHETIC_RUN_ID",
    "METADATA_KEY_STUB_PIPELINE_VERSION",
    "METADATA_KEY_TENANT_ID_HASHED",
]
