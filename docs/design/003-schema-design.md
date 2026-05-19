# Schema Design

**Status:** Proposed
**Author:** Eval-Harness Team
**Date:** 2025-01-19

## 1. Overview

Eval-Harness uses JSON Schema contracts to define data interfaces. This ensures:

- **Explicit documentation** — schema is the specification
- **Runtime validation** — catch errors early
- **Version management** — breaking changes require explicit version bump
- **Tooling support** — generate validators and documentation

## 2. Schema Files

| Schema | Purpose | Used By |
|--------|---------|---------|
| `parser_output.schema.json` | Document parser output | Parsing evaluation |
| `rag_query_output.schema.json` | RAG system response | RAG evaluation |
| `eval_questions.schema.json` | Evaluation criteria | Metrics configuration |

## 3. Parser Output Schema

### 3.1 Purpose

Universal schema for document parsing output. Normalizes different parser formats into a common structure for metric calculation.

### 3.2 Required Fields

| Field | Type | Purpose |
|-------|------|---------|
| `schema_version` | string | Schema version for compatibility checking |
| `parser_version` | string | Parser version for regression tracking |
| `source` | object | Source document metadata |
| `pages` | array | Page-level metadata (dimensions, DPI) |
| `elements` | array | Document elements in reading order |

### 3.3 Source Metadata

Identifies where the data came from:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `doc_id` | string | Yes | Unique document identifier |
| `filename` | string | Yes | Original filename |
| `mime_type` | string | Yes | Document MIME type (usually `application/pdf`) |
| `sha256` | string | No | Hash for cache invalidation |

### 3.4 Element Structure

Each element represents a content unit (paragraph, table, heading, etc.):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `element_id` | string | Yes | Unique identifier within document |
| `type` | string | Yes | Element type (see table below) |
| `text` | string | Yes | Extracted text content |
| `page_index` | integer | Yes | Zero-based page number |
| `char_span` | array | Yes | Character offsets [start, end] |
| `bbox` | object | No | Bounding box coordinates |
| `content` | object | Yes | Type-specific data |

### 3.5 Element Types

| Type | Description | Has bbox? | Content structure |
|------|-------------|-----------|-------------------|
| `paragraph` | Body text | Optional | `{"kind": "text"}` |
| `heading` | Heading or title | Optional | `{"kind": "text", "level": 1-6}` |
| `table` | Table data | Recommended | `{"kind": "table", "rows": N, "cols": M}` |
| `list` | List items | Optional | `{"kind": "list", "items": [...]}` |
| `figure` | Image or figure | Recommended | `{"kind": "figure", "uri": "..."}` |
| `equation` | Math formula | Optional | `{"kind": "equation", "latex": "..."}` |
| `page_break` | Page separator | No | `{"kind": "page_break"}` |

### 3.6 Design Rationale

**Why `char_span`?**
- Enables RAG citation tracking
- Allows text-level metrics (NID, BLEU)
- Supports span overlap detection for recall calculation

**Why discriminated `content` field?**
- Type-specific data lives with element (table rows, figure URIs)
- Extensible without breaking changes
- Clear validation per content type

**Why `element_id` required?**
- Stable references for citations
- Debugging individual elements
- Tracking elements through pipeline

## 4. RAG Query Output Schema

### 4.1 Purpose

Universal schema for RAG system responses. Normalizes different RAG formats for metric calculation.

### 4.2 Required Fields

| Field | Type | Purpose |
|-------|------|---------|
| `answer` | object | Generated answer with metadata |
| `retrieved_chunks` | array | Chunks retrieved from corpus |
| `timings_ms` | object | Performance breakdown |

### 4.3 Answer Structure

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Generated answer text |
| `answer_supported` | boolean | Yes | LLM judgment: is answer grounded in retrieved context? |
| `citations` | array | No | References from answer to chunks |

Each citation contains:
- `chunk_ids`: array of chunk IDs referenced

### 4.4 Retrieved Chunk Structure

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `chunk_id` | string | Yes | Unique chunk identifier |
| `score` | number | Yes | Retrieval relevance score |
| `char_span` | array | No | Character offsets for recall calculation |

### 4.5 Timing Structure

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `retrieval` | integer | No | Retrieval latency in milliseconds |
| `generation` | integer | No | Generation latency in milliseconds |
| `total` | integer | Yes | End-to-end latency in milliseconds |

### 4.6 Design Rationale

**Why separate `retrieved_chunks` from `citations`?**
- Retrieved: what system found
- Citations: what answer claims to use
- Enables citation quality metrics (precision: valid citations / total)

**Why `answer_supported` boolean?**
- LLM-as-judge results (via Bedrock Claude)
- Enables analysis of grounding quality
- Simple flag for aggregation

**Why `timings_ms` required?**
- Performance is a first-class metric
- Breakdown reveals bottlenecks (retrieval vs generation)
- Enables cost tracking (token counts derived from timing)

## 5. Telemetry Fields

### 5.1 Tracing Context

Each output includes optional telemetry fields for Phoenix:

| Field | Type | Purpose |
|-------|------|---------|
| `trace_id` | string | Links to Phoenix trace |
| `job_id` | string | Links to evaluation job |
| `timestamp_ms` | integer | Processing timestamp |

### 5.2 LLM Usage (RAG only)

For RAG evaluation, track Bedrock usage:

| Field | Type | Purpose |
|-------|------|---------|
| `model_id` | string | Bedrock model identifier |
| `input_tokens` | integer | Prompt token count |
| `output_tokens` | integer | Completion token count |
| `cost_usd` | number | Estimated cost |

## 6. Validation Strategy

### 6.1 Validation Points

Validation happens in the adapter before returning to framework:

```
Raw Output → Adapter → Validate → Return Validated Output
                      ↓
                 Error on failure
```

### 6.2 Error Handling

On validation failure:
- Clear error message pointing to specific issue
- Field path in error (e.g., `elements[0].char_span`)
- Expected vs actual values
- Trace ID for debugging in Phoenix

### 6.3 Version Management

**`schema_version` format:** `major.minor.patch`

- **Major**: Breaking changes (removed fields, type changes)
- **Minor**: Additions (new optional fields, new enum values)
- **Patch**: Bug fixes (documentation, constraint corrections)

**Compatibility check:** Framework warns on major version mismatch, errors on incompatible structure.

## 7. Minimal Examples

### 7.1 Minimal Parser Output

Contains one paragraph element, one page. All required fields present, no optional extras.

### 7.2 Minimal RAG Output

Contains answer text, retrieved chunk list, total timing. Sufficient for basic evaluation.

## 8. Extension Points

### 8.1 Adding New Element Types

1. Add type name to schema enum
2. Define `content.kind` structure
3. Bump minor version
4. Update metrics to handle new type if needed

### 8.2 Adding New Telemetry

1. Add optional field to schema
2. Update Phoenix span attributes
3. No version bump if optional

## 9. Related Documents

- [001-Architecture-Overview](001-architecture-overview.md)
- [002-Data-Flow-Detailed](002-data-flow-detailed.md)
- [005-Adapter-Implementation](005-adapter-implementation.md)
