# Schema Design

**Status:** Proposed
**Author:** Eval-Harness Team
**Date:** 2025-01-19

## 1. Overview

Eval-Harness uses JSON Schema contracts to define data interfaces. This ensures:
- **Explicit documentation** — schema IS the specification
- **Runtime validation** — catch errors early
- **Version management** — breaking changes require explicit version bump
- **Tooling support** — generate validators, documentation

## 2. Schema Files

| Schema | Purpose | Used By |
|--------|---------|---------|
| `parser_output.schema.json` | Document parser output | Parsing evaluation |
| `rag_query_output.schema.json` | RAG system response | RAG evaluation |
| `eval_questions.schema.json` | Evaluation criteria | Metrics configuration |

## 3. Parser Output Schema

### 3.1 Purpose

Universal schema for document parsing output. Normalizes different parser formats into a common structure for metric calculation.

### 3.2 Structure

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ParserOutput",
  "type": "object",
  "required": [
    "schema_version",
    "parser_version",
    "source",
    "pages",
    "elements"
  ],
  "properties": {
    "schema_version": {
      "type": "string",
      "description": "Schema version for compatibility checking"
    },
    "parser_version": {
      "type": "string",
      "description": "Parser version for regression tracking"
    },
    "source": {
      "type": "object",
      "description": "Source document metadata"
    },
    "pages": {
      "type": "array",
      "description": "Page-level metadata"
    },
    "elements": {
      "type": "array",
      "description": "Document elements in reading order"
    }
  }
}
```

### 3.3 Key Fields

**Source Metadata:**
```json
"source": {
  "doc_id": "unique_identifier",
  "filename": "document.pdf",
  "mime_type": "application/pdf",
  "sha256": "optional_hash_for_cache_invalidation"
}
```

**Element (Minimal):**
```json
{
  "element_id": "elem_0",
  "type": "paragraph",
  "text": "Content here",
  "page_index": 0,
  "char_span": [0, 50],
  "content": {"kind": "text"}
}
```

**Element (Complete):**
```json
{
  "element_id": "elem_0",
  "type": "table",
  "text": "Table representation",
  "page_index": 0,
  "char_span": [100, 500],
  "bbox": {"x0": 50, "y0": 100, "x1": 500, "y1": 300},
  "content": {
    "kind": "table",
    "rows": 3,
    "cols": 2,
    "cells": [...]
  }
}
```

### 3.4 Element Types

| Type | Description | Has bbox? | Content kind |
|------|-------------|-----------|--------------|
| `paragraph` | Body text | Optional | `text` |
| `heading` | Heading/Title | Optional | `text` + `level` |
| `table` | Table data | Recommended | `table` |
| `list` | List items | Optional | `list` |
| `figure` | Image/Figure | Recommended | `figure` |
| `equation` | Math formula | Optional | `equation` |
| `page_break` | Page separator | No | `page_break` |

### 3.5 Design Rationale

**Why `char_span`?**
- Enables RAG citation tracking
- Allows text-level metrics
- Supports span overlap detection

**Why discriminated `content`?**
- Type-specific data (table rows, figure URIs)
- Extensible without breaking changes
- Clear validation per content type

**Why `element_id` required?**
- Stable references for citations
- Debugging individual elements
- Tracking elements through pipeline

## 4. RAG Query Output Schema

### 4.1 Purpose

Universal schema for RAG system responses. Normalizes different RAG formats for metric calculation.

### 4.2 Structure

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RagQueryOutput",
  "type": "object",
  "required": ["answer", "retrieved_chunks", "timings_ms"],
  "properties": {
    "answer": {
      "type": "object",
      "required": ["text", "answer_supported"],
      "properties": {
        "text": {"type": "string"},
        "answer_supported": {"type": "boolean"},
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["chunk_ids"],
            "properties": {
              "chunk_ids": {
                "type": "array",
                "items": {"type": "string"}
              }
            }
          }
        }
      }
    },
    "retrieved_chunks": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["chunk_id", "score"],
        "properties": {
          "chunk_id": {"type": "string"},
          "score": {"type": "number"},
          "char_span": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2
          }
        }
      }
    },
    "timings_ms": {
      "type": "object",
      "required": ["total"],
      "properties": {
        "retrieval": {"type": "integer"},
        "generation": {"type": "integer"},
        "total": {"type": "integer"}
      }
    }
  }
}
```

### 4.3 Key Fields

**Answer:**
```json
{
  "text": "Generated answer text",
  "answer_supported": true,
  "citations": [
    {"chunk_ids": ["doc1_chunk5", "doc2_chunk3"]}
  ]
}
```

**Retrieved Chunk:**
```json
{
  "chunk_id": "unique_identifier",
  "score": 0.85,
  "char_span": [100, 500]  // Optional: for recall calculation
}
```

### 4.4 Design Rationale

**Why separate `retrieved_chunks` from `citations`?**
- Retrieved: what system found
- Citations: what answer claims to use
- Enables citation quality metrics

**Why `answer_supported` boolean?**
- LLM-as-judge results
- Heuristic checks
- Simple flag for analysis

**Why `timings_ms` required?**
- Performance is a first-class metric
- Retrieval vs generation breakdown
- Latency profiling

## 5. Validation Strategy

### 5.1 Validation Points

```
┌────────────┐    validate    ┌────────────┐
│ Raw Output │ ─────────────▶ │ Adapter    │
└────────────┘               └──────┬─────┘
                                   │
                            Validated
                            Output
```

**Location:** In adapter before returning to framework.

**Benefits:**
- Fail fast on bad data
- Clear error messages
- Framework isolation from user code bugs

### 5.2 Error Handling

```python
try:
    validate(output, schema_path)
except SchemaValidationError as e:
    # Clear error message pointing to issue
    raise ValueError(f"Invalid output: {e.message}")
```

### 5.3 Version Management

**`schema_version` field:**
- `major.minor.patch` format
- Major: breaking changes
- Minor: additions (backward compatible)
- Patch: bug fixes

**Compatibility check:**
```python
if output["schema_version"] != REQUIRED_VERSION:
    # Warn or error based on major version difference
    ...
```

## 6. Schema Examples

### 6.1 Minimal Parser Output

```json
{
  "schema_version": "1.0.0",
  "parser_version": "1.0.0",
  "source": {
    "doc_id": "doc001",
    "filename": "doc001.pdf",
    "mime_type": "application/pdf"
  },
  "pages": [
    {"page_index": 0, "width": 612, "height": 792}
  ],
  "elements": [
    {
      "element_id": "e0",
      "type": "paragraph",
      "text": "Hello world",
      "page_index": 0,
      "char_span": [0, 11],
      "content": {"kind": "text"}
    }
  ]
}
```

### 6.2 Complete Parser Output

```json
{
  "schema_version": "1.0.0",
  "parser_version": "2.1.0",
  "source": {
    "doc_id": "annual_report_2024",
    "filename": "report.pdf",
    "mime_type": "application/pdf",
    "sha256": "abc123..."
  },
  "pages": [
    {
      "page_index": 0,
      "width": 612,
      "height": 792,
      "dpi": 200
    }
  ],
  "elements": [
    {
      "element_id": "h1",
      "type": "heading",
      "text": "Annual Report 2024",
      "page_index": 0,
      "char_span": [0, 17],
      "bbox": {"x0": 100, "y0": 50, "x1": 500, "y1": 80},
      "content": {
        "kind": "text",
        "level": 1
      }
    },
    {
      "element_id": "t1",
      "type": "table",
      "text": "Table 1: Revenue",
      "page_index": 0,
      "char_span": [100, 500],
      "bbox": {"x0": 50, "y0": 100, "x1": 550, "y1": 300},
      "content": {
        "kind": "table",
        "rows": 5,
        "cols": 3,
        "header_rows": 1,
        "cells": [
          {"row": 0, "col": 0, "text": "Year"},
          {"row": 0, "col": 1, "text": "Revenue"}
        ]
      }
    }
  ]
}
```

### 6.3 RAG Query Output

```json
{
  "answer": {
    "text": "The agreement expires on December 31, 2025.",
    "answer_supported": true,
    "citations": [
      {"chunk_ids": ["contract_p5_chunk12"]}
    ]
  },
  "retrieved_chunks": [
    {
      "chunk_id": "contract_p5_chunk12",
      "score": 0.89,
      "char_span": [1200, 1350]
    },
    {
      "chunk_id": "contract_p3_chunk8",
      "score": 0.72,
      "char_span": [800, 950]
    }
  ],
  "timings_ms": {
    "retrieval": 45,
    "generation": 520,
    "total": 565
  }
}
```

## 7. Extension Points

### 7.1 Adding New Element Types

1. Add to `type` enum in schema
2. Document new `content.kind` structure
3. Bump minor version

**Example:**
```json
{
  "element_id": "f1",
  "type": "footnote",
  "text": "See reference...",
  "page_index": 0,
  "char_span": [500, 550],
  "content": {
    "kind": "footnote",
    "refers_to": "elem_5"
  }
}
```

### 7.2 Adding New Content Types

1. Add new content kind structure
2. Extend validation if needed
3. Update metrics to handle new type

## 8. Related Documents

- [001-Architecture-Overview](001-architecture-overview.md)
- [002-Data-Flow-Detailed](002-data-flow-detailed.md)
- [005-Adapter-Implementation](005-adapter-implementation.md)
