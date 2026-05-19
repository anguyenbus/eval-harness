"""
OmniDocBench Parser Adapter for eval-harness.

This is a minimal adapter that wraps your existing OmniDocBench-format parser
to work with eval-harness evaluation.

Usage:
    eval-parsing --dataset omnidocbench --parser examples.omnidocbench_adapter
"""

from pathlib import Path
from typing import Any


def parse(pdf_path: Path) -> dict[str, Any]:
    """
    Parse document and return eval-harness parser_output schema.

    This is the main function that eval-harness calls.

    If you already have a parser that outputs OmniDocBench format,
    wrap it here. Otherwise, implement your parsing logic below.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary conforming to parser_output.schema.json
    """
    # OPTION 1: You have OmniDocBench-format parser
    # Uncomment and modify:
    #
    # from my_parser import parse_omnidocbench
    # omni_output = parse_omnidocbench(pdf_path)
    # return omnidocbench_to_parser_harness(omni_output, pdf_path)

    # OPTION 2: Implement parsing directly
    # (Replace this stub with your actual parsing logic)

    doc_id = pdf_path.stem

    # Example: stub implementation
    # In real use, this would call your parser (Tesseract, Docling, etc.)
    elements = [
        {
            "element_id": f"{doc_id}_para_001",
            "type": "paragraph",
            "page_index": 0,
            "char_span": [0, 50],
            "text": "This is example text from your parser.",
            "content": {"kind": "text"}
        }
    ]

    return {
        "schema_version": "1.0.0",
        "parser_version": "1.0.0",
        "parsed_at": "2025-01-01T00:00:00Z",
        "source": {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "mime_type": "application/pdf",
            "sha256": "",  # Optional: compute hash if needed
            "page_count": 1,
            "language": "en"
        },
        "pages": [
            {
                "page_index": 0,
                "width": 612,
                "height": 792,
                "rotation": 0
            }
        ],
        "elements": elements,
        "warnings": []
    }


def omnidocbench_to_parser_harness(
    omni_output: dict[str, Any],
    pdf_path: Path
) -> dict[str, Any]:
    """
    Convert OmniDocBench format to eval-harness parser_output schema.

    Use this if you have an existing OmniDocBench-format parser.

    Args:
        omni_output: Your parser's output in OmniDocBench format
        pdf_path: Original PDF path for metadata

    Returns:
        Dictionary conforming to parser_output.schema.json
    """
    doc_id = pdf_path.stem

    # Extract page info
    page_info = omni_output.get("page_info", {})
    layout_dets = omni_output.get("layout_dets", [])

    # Convert elements
    elements = []
    char_offset = 0

    # Sort by order field (OmniDocBench reading order)
    sorted_dets = sorted(
        layout_dets,
        key=lambda d: d.get("order", float("inf"))
    )

    for det in sorted_dets:
        text = det.get("text", "")
        if not text:
            continue

        # Extract bbox from poly (if available)
        bbox = _extract_bbox(det.get("poly"))

        element = {
            "element_id": det.get("anno_id", f"{doc_id}_{len(elements):03d}"),
            "type": _map_category_type(det.get("category_type", "text_block")),
            "page_index": page_info.get("page_no", 0),
            "char_span": [char_offset, char_offset + len(text)],
            "text": text,
            "content": {"kind": "text"}
        }

        if bbox:
            element["bbox"] = bbox

        elements.append(element)
        char_offset += len(text)

    return {
        "schema_version": "1.0.0",
        "parser_version": "1.0.0",
        "source": {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "mime_type": "application/pdf",
            "page_count": 1
        },
        "pages": [{
            "page_index": page_info.get("page_no", 0),
            "width": page_info.get("width", 612),
            "height": page_info.get("height", 792)
        }],
        "elements": elements,
        "warnings": []
    }


def _extract_bbox(poly: list[float] | None) -> dict[str, float] | None:
    """
    Extract bounding box from OmniDocBench polygon.

    OmniDocBench uses 8-point polygon; eval-harness uses 4-point bbox.
    """
    if not poly or len(poly) < 4:
        return None

    if len(poly) == 8:
        # Polygon: [x1, y1, x2, y2, x3, y3, x4, y4]
        # Bounding box: min x, min y, max x, max y
        xs = poly[0::2]
        ys = poly[1::2]
        return {
            "x0": min(xs),
            "y0": min(ys),
            "x1": max(xs),
            "y1": max(ys)
        }
    elif len(poly) == 4:
        # Already a bbox: [x0, y0, x1, y1]
        return {
            "x0": poly[0],
            "y0": poly[1],
            "x1": poly[2],
            "y1": poly[3]
        }

    return None


def _map_category_type(category_type: str) -> str:
    """
    Map OmniDocBench category_type to eval-harness element type.

    Args:
        category_type: OmniDocBench category (e.g., "text_block", "table")

    Returns:
        eval-harness element type
    """
    mapping = {
        "text_block": "paragraph",
        "title": "heading",
        "section_header": "heading",
        "header": "header",
        "footer": "footer",
        "table": "table",
        "figure": "figure",
        "equation": "equation",
        "equation_isolated": "equation",
        "list": "list",
        "caption": "caption",
        "footnote": "footnote",
        "page_number": "page_number",
        "code": "code_block"
    }

    return mapping.get(category_type, "paragraph")


# For testing: run directly on a PDF
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python omnidocbench_adapter.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    result = parse(pdf_path)

    import json
    print(json.dumps(result, indent=2))
