"""
Fast parser for digitally created PDFs (no OCR needed).

Uses pypdf for direct text extraction from digital PDFs.
Much faster than docling for documents with embedded text.
"""

import hashlib
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def parse(pdf_path: Path) -> dict[str, Any]:
    """
    Parse a digital PDF using pypdf (fast, no OCR).

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Dictionary conforming to parser_output.schema.json.

    """
    doc_id = pdf_path.stem
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    reader = PdfReader(pdf_path)
    page_count = len(reader.pages)

    # Build pages array
    pages = []
    for page_no in range(page_count):
        page = reader.pages[page_no]
        # pypdf uses mediabox size (points)
        mediabox = page.mediabox
        pages.append(
            {
                "page_index": page_no,
                "width": float(mediabox.width),
                "height": float(mediabox.height),
                "rotation": page.rotation if hasattr(page, "rotation") else 0,
            }
        )

    # Build elements array
    elements = []
    char_offset = 0
    element_id = 0

    for page_no in range(page_count):
        page = reader.pages[page_no]
        text = page.extract_text()

        if not text:
            continue

        # Simple heuristic: split by double newlines for paragraphs
        # For MVP, treat entire page text as one paragraph
        element_id += 1
        char_start = char_offset
        char_end = char_offset + len(text)
        char_offset = char_end

        elements.append(
            {
                "element_id": f"{doc_id}_paragraph_{element_id:03d}",
                "type": "paragraph",
                "page_index": page_no,
                "char_span": [char_start, char_end],
                "text": text,
                "content": {"kind": "text"},
                "bbox": {
                    "x0": 72.0,  # Default margin
                    "y0": 72.0,
                    "x1": pages[page_no]["width"] - 72.0,
                    "y1": pages[page_no]["height"] - 72.0,
                },
                "parent_id": None,
            }
        )

    return {
        "schema_version": "1.0.0",
        "parser_version": "pypdf-fast-0.1.0",
        "parsed_at": "2025-01-01T00:00:00Z",
        "source": {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "mime_type": "application/pdf",
            "sha256": sha256,
            "page_count": page_count,
            "language": "en",
        },
        "pages": pages,
        "elements": elements,
        "warnings": [],
    }
