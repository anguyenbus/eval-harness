"""
Corpus dataset builder for Phoenix EvaluationDataset conversion.

This module provides functionality to convert Phoenix spans into
Phoenix-compatible EvaluationDataset format for replay evaluation.
"""

from __future__ import annotations

from typing import Any

from beartype import beartype


@beartype
class CorpusBuilder:
    """
    Builds Phoenix EvaluationDataset from spans.

    Extracts retrieval documents, LLM inputs/outputs, and ground truth
    from Phoenix spans for evaluation.

    Example:
        >>> builder = CorpusBuilder()
        >>> dataset = builder.build_dataset(spans)

    """

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize corpus builder."""
        pass

    @beartype
    def build_dataset(self, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Build EvaluationDataset from Phoenix spans.

        Args:
            spans: List of span dictionaries from Phoenix.

        Returns:
            List of dataset entries compatible with Phoenix experiments.

        """
        dataset = []

        for span in spans:
            entry = self._extract_entry(span)
            if entry:
                dataset.append(entry)

        return dataset

    @beartype
    def _extract_entry(self, span: dict[str, Any]) -> dict[str, Any] | None:
        """
        Extract dataset entry from a single span.

        Args:
            span: Span dictionary.

        Returns:
            Dataset entry dictionary or None if extraction fails.

        """
        # Extract from RETRIEVER span
        if self._is_retriever_span(span):
            return self._extract_retrieval_entry(span)

        # Extract from LLM span
        if self._is_llm_span(span):
            return self._extract_llm_entry(span)

        return None

    @beartype
    def _is_retriever_span(self, span: dict[str, Any]) -> bool:
        """Check if span is a RETRIEVER span."""
        span_kind = span.get("span_kind", "")
        return span_kind.upper() == "RETRIEVER"

    @beartype
    def _is_llm_span(self, span: dict[str, Any]) -> bool:
        """Check if span is an LLM span."""
        span_kind = span.get("span_kind", "")
        return span_kind.upper() == "LLM"

    @beartype
    def _extract_retrieval_entry(self, span: dict[str, Any]) -> dict[str, Any]:
        """Extract retrieval entry from RETRIEVER span."""
        # Extract documents from span attributes
        documents = []
        for key in span.keys():
            if key.startswith("retrieval.documents."):
                # Parse document attributes
                parts = key.split(".")
                if len(parts) >= 4:
                    doc_idx = parts[2]
                    attr_name = parts[3]

                    # Ensure documents list is long enough
                    while len(documents) <= int(doc_idx):
                        documents.append({})

                    documents[int(doc_idx)][attr_name] = span[key]

        return {
            "retrieval_documents": documents,
            "input": span.get("input.value", ""),
        }

    @beartype
    def _extract_llm_entry(self, span: dict[str, Any]) -> dict[str, Any]:
        """Extract LLM entry from LLM span."""
        # Extract input and output messages
        input_messages = []
        output_messages = []

        for key in span.keys():
            if key.startswith("llm.input_messages."):
                # Parse input message
                parts = key.split(".")
                if len(parts) >= 4:
                    msg_idx = int(parts[2])
                    attr_name = parts[3]

                    while len(input_messages) <= msg_idx:
                        input_messages.append({})

                    input_messages[msg_idx][attr_name] = span[key]

            elif key.startswith("llm.output_messages."):
                # Parse output message
                parts = key.split(".")
                if len(parts) >= 4:
                    msg_idx = int(parts[2])
                    attr_name = parts[3]

                    while len(output_messages) <= msg_idx:
                        output_messages.append({})

                    output_messages[msg_idx][attr_name] = span[key]

        return {
            "llm_input_messages": input_messages,
            "llm_output_messages": output_messages,
        }
