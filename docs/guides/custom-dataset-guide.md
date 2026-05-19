# Custom Dataset Integration Guide

## For Teams with Custom Datasets

Your team has proprietary data and wants to use eval-harness for evaluation. This guide shows how.

## Quick Start

```bash
# 1. Install eval-harness
uv sync

# 2. Create dataset loader (~100 lines)
# 3. Create adapter for your parser (~50 lines)
# 4. Run evaluation
eval-parsing --dataset your_dataset --parser your_parser
```

## Step 1: Understand Your Data Format

### Common Formats

**Format A: JSON per document**
```json
{
  "document_id": "doc_001",
  "pages": [
    {
      "page_no": 1,
      "elements": [
        {"type": "heading", "text": "Introduction", "bbox": [...]},
        {"type": "paragraph", "text": "...", "bbox": [...]}
      ]
    }
  ]
}
```

**Format B: Single JSON with all annotations**
```json
{
  "annotations": [
    {"doc_id": "doc_001", "elements": [...]},
    {"doc_id": "doc_002", "elements": [...]}
  ]
}
```

**Format C: Separate files**
```
data/
├── annotations/
│   ├── doc_001.json
│   └── doc_002.json
├── documents/
│   ├── doc_001.pdf
│   └── doc_002.pdf
```

### What eval-harness needs

Ground truth + parser output in compatible format:

1. **Ground truth**: Your annotated data (gold standard)
2. **Parser output**: Your parser's predictions (same format)
3. **Comparison**: eval-harness compares the two

## Step 2: Create Dataset Loader

### Location

Create: `src/eval_harness/datasets/your_dataset.py`

### Template

```python
"""
Your custom dataset loader.

Follows the same pattern as load_omnidocbench() and load_dp_bench().
"""

from pathlib import Path
from typing import Iterator
from collections.abc import Iterator


def load_your_dataset(root: Path) -> Iterator[tuple]:
    """
    Load your custom dataset for evaluation.
    
    Yields:
        (doc_id, pdf_path, ground_truth) tuples where:
        - doc_id: Unique identifier
        - pdf_path: Path to PDF/image file
        - ground_truth: Dict with your annotations
    """
    # YOUR IMPLEMENTATION HERE
    
    # Example for Format A (JSON per doc):
    json_dir = root / "annotations"
    pdf_dir = root / "documents"
    
    for json_file in json_dir.glob("*.json"):
        # Load ground truth
        gt = json.loads(json_file.read_text())
        doc_id = gt["document_id"]
        
        # Find corresponding PDF
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        
        if not pdf_path.exists():
            continue
        
        yield doc_id, pdf_path, gt
```

### Register Dataset

Edit `src/eval_harness/datasets/__init__.py`:

```python
from eval_harness.datasets.your_dataset import load_your_dataset

__all__ = ["load_legalbench_rag", "load_omnidocbench", "load_dp_bench", "load_your_dataset"]
```

## Step 3: Create Parser Adapter

### If Your Parser Outputs Custom Format

```python
def parse_your_parser(pdf_path: Path) -> dict:
    """Your existing parser."""
    # Returns your custom format
    return {
        "elements": [...],
        "pages": [...]
    }


def parse_for_eval(pdf_path: Path) -> dict:
    """Adapter for eval-harness."""
    your_output = parse_your_parser(pdf_path)
    return convert_to_eval_harness(your_output, pdf_path)
```

### Conversion Function

```python
def convert_to_eval_harness(your_output: dict, pdf_path: Path) -> dict:
    """Convert your format to parser_output schema."""
    elements = []
    char_offset = 0
    
    for elem in your_output.get("elements", []):
        elements.append({
            "element_id": f"{pdf_path.stem}_{len(elements)}",
            "type": map_type(elem["type"]),
            "page_index": elem.get("page", 0),
            "char_span": [char_offset, char_offset + len(elem["text"])],
            "text": elem["text"],
            "content": {"kind": "text"}
        })
        char_offset += len(elem["text"])
    
    return {
        "schema_version": "1.0.0",
        "parser_version": "1.0.0",
        "source": {
            "doc_id": pdf_path.stem,
            "filename": pdf_path.name,
            "mime_type": "application/pdf"
        },
        "pages": your_output.get("pages", []),
        "elements": elements
    }


def map_type(your_type: str) -> str:
    """Map your element types to eval-harness types."""
    mapping = {
        "your_heading": "heading",
        "your_body": "paragraph",
        "your_table": "table"
    }
    return mapping.get(your_type, "paragraph")
```

## Step 4: Update CLI (Optional)

If you want CLI support, edit `src/eval_harness/runners/run_parsing_eval.py`:

```python
# Add to dataset choices
parser.add_argument(
    "--dataset",
    choices=["omnidocbench", "dp_bench", "your_dataset"],  # Add yours
    required=True,
)

# Add to load_dataset()
def load_dataset(dataset_name: str, config: dict):
    if dataset_name == "your_dataset":
        from eval_harness.datasets import load_your_dataset
        root = Path(config["datasets"]["your_dataset"]["path"])
        return load_your_dataset(root)
    # ... existing datasets ...
```

## Step 5: Configure Dataset Path

Edit `eval_config.yaml`:

```yaml
datasets:
  your_dataset:
    path: /path/to/your/data
```

## Step 6: Run Evaluation

```bash
# Test with limit
eval-parsing --dataset your_dataset --parser your_parser --limit 10

# Full evaluation
eval-parsing --dataset your_dataset --parser your_parser
```

## Common Scenarios

### Scenario 1: You Have PDF + JSON Annotations

```python
def load_your_dataset(root: Path):
    json_dir = root / "annotations"
    pdf_dir = root / "pdfs"
    
    for json_file in json_dir.glob("*.json"):
        gt = json.loads(json_file.read_text())
        pdf_path = pdf_dir / f"{gt['doc_id']}.pdf"
        
        if pdf_path.exists():
            yield gt['doc_id'], pdf_path, gt
```

### Scenario 2: You Have CSV Annotations

```python
import pandas as pd

def load_your_dataset(root: Path):
    csv_file = root / "annotations.csv"
    pdf_dir = root / "pdfs"
    
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        pdf_path = pdf_dir / row['pdf_filename']
        gt = row.to_dict()
        
        if pdf_path.exists():
            yield row['doc_id'], pdf_path, gt
```

### Scenario 3: You Have Database Storage

```python
import sqlite3

def load_your_dataset(root: Path):
    db_path = root / "annotations.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT doc_id, pdf_path, annotations FROM documents")
    
    for row in cursor.fetchall():
        doc_id, pdf_rel_path, gt_json = row
        pdf_path = root / pdf_rel_path
        
        if pdf_path.exists():
            gt = json.loads(gt_json)
            yield doc_id, pdf_path, gt
```

### Scenario 4: Images Instead of PDFs

```python
def load_your_dataset(root: Path):
    images_dir = root / "images"
    json_dir = root / "annotations"
    
    for json_file in json_dir.glob("*.json"):
        gt = json.loads(json_file.read_text())
        image_name = gt["image_name"]
        image_path = images_dir / image_name
        
        if image_path.exists():
            yield gt["doc_id"], image_path, gt
```

## What Metrics You Get

Based on your data:

| Your Data Has | Metrics Available |
|---------------|-------------------|
| Text content | NID, BLEU, METEOR |
| Headings | MHS |
| Tables | TEDS |
| Reading order | ARD |
| Bounding boxes | Layout accuracy |
| Element types | Type precision/recall |

## Handling Different Data Types

### Text-Only Annotations

```python
# Minimal viable
{
    "elements": [
        {"type": "paragraph", "text": "Content", "page_index": 0}
    ]
}
```

### With Bounding Boxes

```python
{
    "elements": [
        {
            "type": "paragraph",
            "text": "Content",
            "page_index": 0,
            "bbox": {"x0": 10, "y0": 20, "x1": 100, "y1": 30}
        }
    ]
}
```

### With Tables

```python
{
    "elements": [
        {
            "type": "table",
            "page_index": 0,
            "content": {
                "kind": "table",
                "rows": 3,
                "cols": 2,
                "cells": [
                    {"row": 0, "col": 0, "text": "Header"},
                    {"row": 1, "col": 0, "text": "Cell 1"}
                ]
            }
        }
    ]
}
```

## Troubleshooting

### Issue: "Dataset not found"

**Solution:** Check `eval_config.yaml` path is correct.

### Issue: "No documents yielded"

**Solution:** Verify PDF/image files exist at expected paths.

### Issue: "Parser output validation failed"

**Solution:** Ensure parser output has required fields: `elements`, `type`, `text`, `page_index`.

### Issue: "Metrics showing zeros"

**Solution:** 
1. Check ground truth and prediction have same format
2. Verify text content is being extracted correctly
3. Run with `--limit 1` and inspect output

## Checklist

Before running full evaluation:

- [ ] Dataset loader yields (doc_id, file_path, ground_truth)
- [ ] Parser returns parser_output schema format
- [ ] Ground truth and parser output use same element types
- [ ] Test with `--limit 10` first
- [ ] Check results CSV is being populated

## Next Steps

1. Start with subset of data (10-20 documents)
2. Validate metrics look reasonable
3. Scale to full dataset
4. Compare with your existing evaluation (if any)
5. Integrate into CI/CD for regression testing

## Support

- Dataset loader examples: `src/eval_harness/datasets/omnidocbench.py`
- Adapter examples: `examples/omnidocbench_adapter.py`
- Schema reference: `contracts/parser_output.schema.json`
