# Parser Output Schema: Why It Exists and How It Works

## Overview

This guide explains the rationale behind eval-harness's `parser_output.schema.json`, how the adapter pattern works, and the trade-offs involved.

## Table of Contents

1. [The Problem: Benchmark Fragmentation](#the-problem-benchmark-fragmentation)
2. [The Solution: Universal Schema](#the-solution-universal-schema)
3. [How Adapters Work](#how-adapters-work)
4. [How Metrics Consume the Schema](#how-metrics-consume-the-schema)
5. [Why This Approach Is Good](#why-this-approach-is-good)
6. [Pros and Cons](#pros-and-cons)
7. [Registering Your Adapter](#registering-your-adapter)
8. [Code Examples](#code-examples)

---

## The Problem: Benchmark Fragmentation

### Landscape of Document Parsing Benchmarks

Each benchmark has its own JSON schema:

**OmniDocBench (verified from actual data):**
```json
[
  {
    "layout_dets": [
      {
        "category_type": "text_block",
        "poly": [268.9, 319.9, 322.9, 319.9, 322.9, 351.0, 268.9, 351.0],
        "ignore": false,
        "order": 2,
        "anno_id": "box_id_0",
        "text": "that"
      }
    ],
    "page_info": {
      "page_no": 0,
      "height": 2339,
      "width": 1653,
      "image_path": "page-xxx.png",
      "page_attribute": {
        "language": "english",
        "data_source": "book",
        "layout": "single_column"
      }
    }
  }
]
```

**DP-Bench (verified from actual data):**
```json
{
  "01030000000001.pdf": {
    "elements": [
      {
        "coordinates": [
          {"x": 170.9, "y": 102.3},
          {"x": 208.5, "y": 102.3},
          {"x": 208.5, "y": 120.6},
          {"x": 170.9, "y": 120.6}
        ],
        "category": "Header",
        "id": 0,
        "page": 1,
        "content": {
          "text": "314",
          "html": "",
          "markdown": ""
        }
      }
    ]
  }
}
```

### Problems This Fragmentation Causes

1. **No common evaluation:** Each benchmark needs its own metrics implementation
2. **No cross-benchmark comparison:** Cannot compare parser performance across benchmarks
3. **Vendor lock-in:** Switching benchmarks requires rewriting evaluation code
4. **Inconsistent metrics:** Different names for same concept (precision vs recall)
5. **Maintenance burden:** Each benchmark change requires updates to evaluation code

---

## The Solution: Universal Schema

### Design Philosophy

eval-harness uses a **universal schema** that is NOT tied to any specific benchmark:

1. **Benchmark-agnostic:** Works with OmniDocBench, DP-Bench, or your proprietary data
2. **Evaluation-focused:** Optimized for computing metrics, not for representing raw parser output
3. **Minimal adapter cost:** ~50 lines of code to convert any format to universal schema
4. **Write-once, reuse everywhere:** Same adapter works for all eval-harness metrics

### The Universal Schema

See [`references/parser_output.schema.json`](../../references/parser_output.schema.json) for full specification.

**Key structure:**
```json
{
  "schema_version": "1.0.0",
  "parser_version": "1.0.0",
  "source": {
    "doc_id": "unique_id",
    "filename": "document.pdf",
    "mime_type": "application/pdf",
    "sha256": "..."
  },
  "pages": [
    {"page_index": 0, "width": 612, "height": 792}
  ],
  "elements": [
    {
      "element_id": "elem_001",
      "type": "paragraph",
      "page_index": 0,
      "char_span": [0, 50],
      "text": "Content here",
      "content": {"kind": "text"},
      "bbox": {"x0": 10, "y0": 20, "x1": 100, "y1": 30}
    }
  ]
}
```

### Why This Structure?

| Component | Purpose |
|-----------|---------|
| `schema_version` | Prevent breaking changes; eval-harness rejects incompatible versions |
| `parser_version` | Attribute results to specific parser release for regression tracking |
| `source.sha256` | Detect when re-parsing is needed (cache invalidation) |
| `pages[]` | Per-page metadata for layout metrics |
| `elements[]` | Ordered list — reading order matters for ARD metric |
| `element_id` | Stable identifier for citations and debugging |
| `type` enum | Normalized types across all benchmarks (19 standard types) |
| `char_span` | Character offsets within document; used for RAG citation tracking |
| `bbox` (optional) | Layout metrics if available; not required for text-based metrics |
| `content` (discriminated) | Type-specific structured data (table cells, figure URIs, etc.) |

---

## How Adapters Work

### The Adapter Pattern

An adapter is a pure function that converts from any format to eval-harness format:

```
┌─────────────────────┐
│ Your Parser Output │  (OmniDocBench, DP-Bench, custom, etc.)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Adapter Function  │  (~50 lines of conversion code)
│   - map_types()     │
│   - extract_bbox()  │
│   - sort_by_order() │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  parser_output      │  (universal schema)
│  schema conformant  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   All Metrics Work  │  (NID, TEDS, MHS, ARD, BLEU, etc.)
└─────────────────────┘
```

### OmniDocBench → ParserOutput Mapping

| OmniDocBench Field | ParserOutput Field | Conversion |
|--------------------|--------------------|------------|
| `layout_dets[]` | `elements[]` | Direct array mapping |
| `anno_id` | `element_id` | Direct copy |
| `text` | `text` | Direct copy |
| `category_type` | `type` | Type mapping function (e.g., "text_block" → "paragraph") |
| `poly[8]` | `bbox{x0,y0,x1,y1}` | min/max extraction from 8-point polygon |
| `order` | *(array position)* | Sort by `order` field first |
| `page_info.page_no` | `elements[].page_index` | Direct copy |
| `page_info.width/height` | `pages[]` | Object to array conversion |

### DP-Bench → ParserOutput Mapping

| DP-Bench Field | ParserOutput Field | Conversion |
|----------------|--------------------|------------|
| Filename key | `source.doc_id` | Extract from JSON key |
| `elements[]` | `elements[]` | Direct array mapping |
| `id` | `element_id` | Convert int to string |
| `category` | `type` | Type mapping (e.g., "Header" → "header", "Paragraph" → "paragraph") |
| `coordinates[]` | `bbox{x0,y0,x1,y1}` | Extract min/max from 4 points |
| `page` | `elements[].page_index` | Convert to 0-indexed |
| `content.text` | `text` | Direct copy |

### Type Mapping Example

```python
def map_category_type(category_type: str) -> str:
    """Map benchmark-specific types to eval-harness universal types."""
    mapping = {
        # OmniDocBench → eval-harness
        "text_block": "paragraph",
        "title": "heading",
        "section_header": "heading",
        "equation_isolated": "equation",

        # DP-Bench → eval-harness
        "Header": "header",
        "Paragraph": "paragraph",
        "Table": "table",
    }
    return mapping.get(category_type, "paragraph")
```

### Bbox Extraction Examples

**From OmniDocBench 8-point polygon:**
```python
def extract_bbox_from_poly(poly: list[float] | None) -> dict[str, float] | None:
    """Extract bounding box from 8-point polygon."""
    if not poly or len(poly) < 4:
        return None

    if len(poly) == 8:
        xs = poly[0::2]  # [x0, x1, x2, x3]
        ys = poly[1::2]  # [y0, y1, y2, y3]
        return {
            "x0": min(xs),
            "y0": min(ys),
            "x1": max(xs),
            "y1": max(ys)
        }
    return None
```

**From DP-Bench 4-point coordinates:**
```python
def extract_bbox_from_coords(coordinates: list[dict]) -> dict[str, float] | None:
    """Extract bounding box from DP-Bench coordinates array."""
    if not coordinates or len(coordinates) < 4:
        return None

    xs = [c["x"] for c in coordinates]
    ys = [c["y"] for c in coordinates]

    return {
        "x0": min(xs),
        "y0": min(ys),
        "x1": max(xs),
        "y1": max(ys)
    }
```

### Why Adapters Are Small

Adapters are small because:
1. **Mechanical transformation:** Field renaming, type mapping, coordinate conversion
2. **No business logic:** No metric computation, no complex algorithms
3. **Lossless conversion:** All information preserved, just reorganized
4. **One-time effort:** Write once, reuse for all metrics

Typical adapter:
- 10 lines: metadata extraction
- 15 lines: element loop with type mapping
- 5 lines: bbox conversion
- 10 lines: return statement
- **Total: ~40-50 lines**

---

## How Metrics Consume the Schema

Understanding how metrics actually use the parser_output schema clarifies design decisions.

| Metric | Schema Fields Used | How It Works |
|--------|-------------------|--------------|
| **NID/NID-S** | `elements[].text` | Converts elements to Markdown, concatenates text, computes fuzzy string match |
| **TEDS** | `elements[]` where `type=="table"` + `content.kind=="table"` | Extracts table structure (rows, cols, cells), computes tree edit distance |
| **MHS** | `elements[]` where `type=="heading"` + `level` | Compares heading hierarchy between gold and prediction |
| **ARD** | `elements[]` array order (implicit) | Measures displacement in reading order using element_id matching |
| **BLEU/METEOR** | `elements[].text` | Token-level n-gram overlap on concatenated text |
| **Precision@k/Recall@k** | `elements[].bbox` (optional) | Spatial overlap between predicted and gold bounding boxes |

### Key Insight

Metrics consume the schema via **Markdown conversion** for text similarity and **direct field access** for structure. The universal schema enables:

1. Single Markdown converter works for all benchmarks
2. Same metric code works for all parsers
3. No benchmark-specific logic in metric implementations

---

## Why This Approach Is Good

### 1. Separation of Concerns

| Concern | Who Owns It |
|---------|-------------|
| **Benchmark format** | Benchmark provider (OmniDocBench, DP-Bench) |
| **Parser output** | Your team's parser |
| **Adapter** | Your team (one-time, ~50 lines) |
| **Metrics** | eval-harness (reusable, standardized) |

No single entity needs to know everything.

### 2. Evaluation Independence

eval-harness doesn't need to know:
- What benchmark you're using
- What parser you're using
- How your data is stored

It only needs: `parser_output` schema → metrics

### 3. Cross-Benchmark Comparison (With Caveats)

Because all data flows through the same schema:

```bash
# Evaluate same parser on different benchmarks
eval-parsing --dataset omnidocbench --parser my_adapter
eval-parsing --dataset dp_bench --parser my_adapter

# Results use same metrics, same scale
```

**Important caveat:** Cross-benchmark comparison indicates relative performance, NOT absolute equality. Different benchmarks have varying difficulty levels, document types, and annotation quality. NID=0.85 on DP-Bench (easier, digital PDFs) ≠ NID=0.70 on OmniDocBench (harder, academic papers with equations).

### 4. Future-Proofing

New benchmark appears?
- Write one adapter (~50 lines)
- All existing metrics work immediately

New metric appears?
- Implement once against `parser_output` schema
- Works for all benchmarks automatically

### 5. RAG Integration

The `char_span` field enables citation tracking:

```json
{
  "element_id": "para_001",
  "text": "The contract terminates on 2025-12-31.",
  "char_span": [1250, 1295]
}
```

**How it's used:** In eval-harness RAG evaluation, chunks track their `char_span` offsets. When a RAG system cites evidence, eval-harness validates the citation spans against the source document.

**Caveat:** Span-based citation requires character-level alignment between parser output and source document. OCR errors and extraction inconsistencies can break alignment. Element-based citation (using `element_id`) is more robust.

---

## Pros and Cons

### Pros

| Pro | Explanation |
|-----|-------------|
| **Universal evaluation** | Same metrics work across all benchmarks and custom datasets |
| **Minimal integration** | ~50 lines, ~40 minutes one-time effort |
| **No data migration** | Original data untouched; adapter is read-only transformation |
| **Cross-benchmark comparison** | Compare parser performance using same metrics, same scale |
| **Future-proof** | New benchmarks → new adapter only; metrics unchanged |
| **Citation-ready** | `element_id` and `char_span` enable RAG citation evaluation |
| **Versioned** | `schema_version` prevents breaking changes |
| **Extensible** | Discriminated `content` union supports new element types |
| **Vendor-agnostic** | Works with Docling, Tesseract, Azure, PaddleOCR, etc. |
| **Debuggable** | `warnings[]` array for parser diagnostics |

### Cons

| Con | Mitigation |
|-----|------------|
| **One-time adapter cost** | ~50 lines per format; templates provided in examples/ |
| **Learning curve** | Need to understand schema structure; this guide + examples provided |
| **Another schema to maintain** | eval-harness maintains it; you just use it |
| **Type mapping required** | Mapping function provided in examples/omnidocbench_adapter.py |
| **char_span computation** | Required for citation eval; straightforward for single-doc, complex for multi-page |
| **Not optimized for storage** | This is evaluation format, not storage format; use native format for storage |
| **JSON verbosity** | Trade-off for clarity and validation; negligible compared to parsing time |
| **Cross-benchmark comparison limited** | Different difficulty levels; use for relative comparison only |

### When eval-harness Schema Is NOT Worth It

| Scenario | Reason |
|----------|--------|
| One-off experiment | Native benchmark evaluation is simpler |
| Single benchmark, no plans to expand | Existing eval may be sufficient |
| Non-Python environment | Adapter requires Python runtime |
| Extremely latency-sensitive | Even minimal adapter overhead matters (though parsing dominates) |

---

## Registering Your Adapter

### Quick Integration

After writing your adapter function, register it with eval-harness:

**Option 1: Add to stubs/ (for testing)**
```bash
# Copy your adapter to stubs directory
cp my_adapter.py src/eval_harness/stubs/my_parser.py

# Edit run_parsing_eval.py to add your parser choice
# See get_parser() function for examples
```

**Option 2: Use as module (for production)**
```bash
# Keep adapter in your project
# Pass module path to --parser argument (requires CLI extension)
eval-parsing --dataset my_dataset --parser myproject.adapters.my_adapter
```

### CLI Integration Template

To add your parser as a CLI option, edit `src/eval_harness/runners/run_parsing_eval.py`:

```python
def get_parser(parser_name: str) -> tuple[ParserAdapter, str]:
    """Get parser adapter by name."""
    # ... existing parsers ...

    elif parser_name == "my_parser":
        from myproject.adapters import my_adapter
        return ParserAdapter(my_adapter.parse), "my_parser"

    else:
        # fallback
        ...
```

Then add to argparse choices:
```python
parser.add_argument(
    "--parser",
    choices=["stub", "fast", "docling", "my_parser"],  # Add yours
    ...
)
```

---

## Code Examples

### Minimal Viable Adapter

The smallest adapter that works (text-only evaluation):

```python
from pathlib import Path

def my_adapter(pdf_path: Path) -> dict:
    """Minimal adapter for text-only evaluation."""

    # Your parser (any format)
    raw_output = my_parser(pdf_path)

    # Convert to eval-harness format
    elements = []
    char_offset = 0

    for item in raw_output["items"]:
        elements.append({
            "element_id": f"elem_{len(elements)}",
            "type": "paragraph",  # Simplified
            "page_index": 0,
            "char_span": [char_offset, char_offset + len(item["text"])],
            "text": item["text"],
            "content": {"kind": "text"}
        })
        char_offset += len(item["text"])

    return {
        "schema_version": "1.0.0",
        "parser_version": "1.0.0",
        "source": {
            "doc_id": pdf_path.stem,
            "filename": pdf_path.name,
            "mime_type": "application/pdf"
        },
        "pages": [{"page_index": 0, "width": 612, "height": 792}],
        "elements": elements
    }
```

This gets you: NID, BLEU, METEOR (text-based metrics).

Add `type` mapping and `bbox` for: TEDS, MHS, ARD (layout metrics).

### With Table Support

```python
def convert_table(my_table: dict) -> dict:
    """Convert your table format to eval-harness TableContent."""
    return {
        "kind": "table",
        "rows": my_table["row_count"],
        "cols": my_table["col_count"],
        "header_rows": my_table.get("header_rows", 1),
        "cells": [
            {
                "row": cell["row"],
                "col": cell["col"],
                "text": cell["content"],
                "row_span": cell.get("row_span", 1),
                "col_span": cell.get("col_span", 1)
            }
            for cell in my_table["cells"]
        ]
    }
```

---

## Summary

### The Core Idea

**Universal schema + adapter pattern = evaluate anything with standard metrics.**

- **Universal schema**: `parser_output.schema.json` — works for all benchmarks
- **Adapter pattern**: ~50 lines converts any format to universal schema
- **Standard metrics**: NID, TEDS, MHS, ARD, BLEU, METEOR work on everything

### Why It Exists

1. **Benchmark fragmentation** → universal schema unifies evaluation
2. **No cross-benchmark comparison** → same schema enables comparison (with caveats)
3. **Vendor lock-in** → adapter pattern decouples parser from evaluation
4. **Maintenance burden** → write metrics once, reuse forever

### The Trade-off

| Cost | Benefit |
|------|---------|
| ~50 lines adapter (one-time) | Universal evaluation forever |
| Learn new schema | All benchmarks work automatically |
| Type mapping function | Cross-benchmark comparison (relative) |
| char_span computation | RAG citation evaluation |

For most teams, the one-time ~40 minute investment is worth the long-term benefits.

---

## Related Documentation

- [Schema Alignment Guide](schema-alignment-guide.md) — OmniDocBench → eval-harness mapping details
- [Custom Dataset Guide](custom-dataset-guide.md) — Create your own dataset loader
- [FAQ: OmniDocBench Users](faq-omnidocbench-users.md) — Common questions from OmniDocBench teams
- [OpenSearch Integration](opensearch-integration.md) — RAG evaluation with vector stores
- [OmniDocBench Adapter Example](../../examples/omnidocbench_adapter.py) — Working adapter code
