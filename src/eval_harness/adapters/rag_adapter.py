"""
RAG adapter with schema validation.

The adapter pattern allows swapping between stub and real RAG systems while
ensuring all output conforms to the rag_query_output schema.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval_harness.adapters.schema_validator import (
    validate as schema_validate,
)

# Type alias for query callable
QueryCallable = Callable[[str, Path], dict[str, Any]]


class RagAdapter:
    """
    Adapter for RAG systems with schema validation.

    The adapter wraps a query function and validates its output against
    the rag_query_output schema before returning. This ensures that any RAG
    system (stub or real) produces conformant output.

    Attributes:
        query_callable: Function with signature (question: str, corpus_dir: Path) -> dict.

    Example:
        >>> adapter = RagAdapter()
        >>> output = adapter.query("What is this?", Path("corpus"))
        >>> # output is guaranteed to validate against rag_query_output schema

    """

    def __init__(self, query_callable: QueryCallable | None = None) -> None:
        """
        Initialize RAG adapter.

        Args:
            query_callable: Optional query function. If None, uses stub RAG.
                Must have signature: (question: str, corpus_dir: Path) -> dict.

        """
        if query_callable is None:
            # Import stub as default
            from eval_harness.stubs.stub_ingestion import query as stub_query

            self._query = stub_query
        else:
            self._query = query_callable

    def query(self, question: str, corpus_dir: Path) -> dict[str, Any]:
        """
        Query a RAG system and return validated output.

        Invokes the underlying query callable, validates the output against
        rag_query_output.schema.json, and returns the validated dict.

        Args:
            question: The question text to query.
            corpus_dir: Path to document corpus directory.

        Returns:
            Validated RAG query output dictionary conforming to schema.

        Raises:
            SchemaValidationError: If RAG output fails schema validation.

        """
        # Invoke RAG system
        output = self._query(question, corpus_dir)

        # Validate against schema
        schema_path = Path("contracts/rag_query_output.schema.json")
        schema_validate(output, schema_path)

        return output
