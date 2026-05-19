# Contract Schemas

This directory contains JSON Schema contracts that define the input/output interfaces for the evaluation harness.

## parser_output.schema.json

Defines the structured output produced by document parsers. One JSON document per source file. Contains:
- `schema_version`: Semver of this schema (currently "1.0.0")
- `parser_version`: Semver of the parser that produced this output
- `source`: Document provenance (doc_id, filename, mime_type, sha256, etc.)
- `pages`: Per-page metadata (width, height, rotation)
- `elements`: Ordered list of structural elements with type discriminator
- `warnings`: Non-fatal parser warnings (e.g., low OCR confidence)

Elements have 24 possible types including: heading, paragraph, list, table, figure, caption, footnote, header, footer, page_number, code_block, equation. Each element includes:
- `element_id`: Stable unique identifier within document
- `type`: Element type discriminator
- `page_index`: Which page this element appears on
- `char_span`: [start, end) character offsets in full document text
- `text`: Plain text content
- `content`: Type-specific structured content
- Optional: `bbox`, `level`, `parent_id`, `confidence`

## rag_query_output.schema.json

Defines the structured output produced by RAG query systems for a single question. Enables citation evaluation by linking every claim in the answer to retrieved chunks, and every retrieved chunk to parser elements via element_id and char_span. Contains:
- `schema_version`: Semver of this schema (currently "1.0.0")
- `system_version`: Version pins for all pipeline components
- `query`: The question being asked (query_id, text, metadata)
- `answer`: Generated answer with citations (text, answer_supported, citations[])
- `retrieved_chunks`: All chunks surfaced by retriever, in rank order

Each retrieved chunk includes:
- `chunk_id`: Unique within this response
- `rank`: Retrieval rank (0 = highest score)
- `score`: Final retrieval score after reranking
- `doc_id`: Joins back to parser_output.source.doc_id
- `element_ids`: Links to parser_output.elements[] for structure-aware eval
- `char_span`: [start, end) character offsets in source document
- `text`: Verbatim text passed to generator

## eval_questions.schema.json

Defines the schema for declaring evaluation questions. Two kinds of evaluators:
- `deterministic`: Code-based metrics computed against ground truth (e.g., precision_at_k, char_level_f1, table_teds)
- `llm_judge`: LLM-as-judge prompts with binary Yes/No output

Each question includes:
- `question_id`: Stable identifier (used as Phoenix annotation_name)
- `pillar`: Top-level grouping (parsing, retrieval, faithfulness, citation)
- `criterion`: Short human-readable name (e.g., "span_overlap", "no_hallucinated_facts")
- `evaluator`: Either deterministic metric config or LLM judge config
- `required_content`: Fields needed for evaluation (e.g., retrieved_chunks, gold_spans)
- `scope_guardrail`: What this evaluator should NOT penalize (prevents eval drift)
- Optional: `applies_to` filters, `severity` levels for regression checks

This schema enables the eval harness to skip questions when required fields are missing, apply scope guardrails consistently, and aggregate results by pillar for reporting.
