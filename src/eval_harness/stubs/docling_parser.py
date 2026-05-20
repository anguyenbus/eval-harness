"""
Docling parser for document parsing evaluation.

This module uses the Docling library to parse PDF documents and convert
the output to the eval-harness schema format.
"""

# Force CPU usage BEFORE importing docling
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DOCLING_DEVICE"] = "cpu"

import hashlib
from pathlib import Path
from typing import Any

# Try to import docling, provide clear error if not available
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.settings import settings
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DocItemLabel

    DOCLING_AVAILABLE = True

    # Create converter once and reuse (weight loading is expensive)
    _DOCLING_CONVERTER: DocumentConverter | None = None

    def _get_converter() -> DocumentConverter:
        """Get cached DocumentConverter instance."""
        global _DOCLING_CONVERTER
        if _DOCLING_CONVERTER is None:
            _DOCLING_CONVERTER = DocumentConverter()
        return _DOCLING_CONVERTER

except ImportError:
    DOCLING_AVAILABLE = False
    DocumentConverter = None
    InputFormat = None
    settings = None
    DocItemLabel = None

    def _get_converter():
        raise ImportError("docling is not installed")


# Map docling labels to our schema types
# Schema allows: heading, paragraph, list, list_item, table, figure, caption,
#                footnote, header, footer, page_number, code_block, equation
DOCLING_LABEL_MAP = {
    DocItemLabel.TITLE: "heading",
    DocItemLabel.DOCUMENT_INDEX: "heading",
    DocItemLabel.SECTION_HEADER: "heading",
    DocItemLabel.PARAGRAPH: "paragraph",
    DocItemLabel.TABLE: "table",
    DocItemLabel.PICTURE: "figure",
    DocItemLabel.FORMULA: "equation",
    DocItemLabel.TEXT: "paragraph",  # Map generic text to paragraph
    DocItemLabel.LIST_ITEM: "list_item",
    DocItemLabel.CODE: "code_block",  # Map code to code_block
    DocItemLabel.CAPTION: "caption",
    DocItemLabel.PAGE_HEADER: "header",
    DocItemLabel.PAGE_FOOTER: "footer",
    DocItemLabel.FOOTNOTE: "footnote",
    # Skip or map to paragraph: CHECKBOX, REFERENCE
    DocItemLabel.CHECKBOX_UNSELECTED: "paragraph",
    DocItemLabel.CHECKBOX_SELECTED: "paragraph",
    DocItemLabel.REFERENCE: "paragraph",
}


def _get_doc_id(pdf_path: Path) -> str:
    """Generate a stable doc_id from PDF filename."""
    return pdf_path.stem


def _get_sha256(pdf_path: Path) -> str:
    """Generate SHA-256 hash of PDF file."""
    content = pdf_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def _polygon_from_bbox(bbox: Any, page_height: int) -> list[float]:
    """
    Convert docling bbox to polygon format.

    Docling bbox is [x0, y0, x1, y1] with origin at top-left.
    Our polygon format is [x1, y1, x2, y2, x3, y3, x4, y4].

    Args:
        bbox: Docling bounding box with x0, y0, x1, y1
        page_height: Page height for coordinate conversion

    Returns:
        Polygon as [x0, y0, x1, y0, x1, y1, x0, y1]

    """
    if not bbox:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    x0 = float(bbox.l)
    y0 = float(bbox.t)
    x1 = float(bbox.r)
    y1 = float(bbox.b)

    # Return polygon in clockwise order from top-left
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _get_element_type(item: Any) -> str:
    """
    Map docling item label to our schema element type.

    Args:
        item: Docling DocItem

    Returns:
        Element type string for our schema

    """
    if not DOCLING_AVAILABLE:
        return "text"

    label = item.label if hasattr(item, "label") else None

    # Map docling label to our type
    if label in DOCLING_LABEL_MAP:
        return DOCLING_LABEL_MAP[label]

    # Default fallback
    return "text"


def _get_heading_level(item: Any) -> int | None:
    """
    Get heading level from docling item.

    Args:
        item: Docling DocItem

    Returns:
        Heading level (1-6) or None

    """
    if not DOCLING_AVAILABLE:
        return None

    if hasattr(item, "label"):
        if item.label == DocItemLabel.TITLE:
            return 1
        elif item.label == DocItemLabel.DOCUMENT_INDEX:
            return 1
        elif item.label == DocItemLabel.SECTION_HEADER:
            # Try to get hierarchy level
            if hasattr(item, "props") and hasattr(item.props, "level"):
                return item.props.level
            return 2

    return None


def parse(pdf_path: Path) -> dict[str, Any]:
    """
    Parse a PDF file using Docling and return schema-conformant parser_output.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Dictionary conforming to parser_output.schema.json.

    Raises:
        ImportError: If docling is not installed.
        RuntimeError: If parsing fails.

    """
    if not DOCLING_AVAILABLE:
        raise ImportError("docling is not installed. Install with: uv add docling")

    doc_id = _get_doc_id(pdf_path)

    try:
        # Get cached converter (created once, reused for all documents)
        converter = _get_converter()
        result = converter.convert(pdf_path)
        doc = result.document

        # Get file hash
        sha256 = _get_sha256(pdf_path)

        # Build pages array (doc.pages is dict in newer docling versions)
        pages = []
        doc_pages_dict = doc.pages if isinstance(doc.pages, dict) else {}
        page_count = len(doc_pages_dict)

        for page_no in range(page_count):
            page = doc_pages_dict.get(page_no)
            if page is None:
                # Default page size if not found
                pages.append(
                    {
                        "page_index": page_no,
                        "width": 612,
                        "height": 792,
                        "rotation": 0,
                    }
                )
                continue

            # Handle both Page object and simple size dict
            if hasattr(page, "size"):
                if hasattr(page.size, "width"):
                    width = page.size.width
                    height = page.size.height
                else:
                    # size is a dict
                    width = page.size.get("width", 612)
                    height = page.size.get("height", 792)
            else:
                # page might be a dict
                width = page.get("width", 612) if isinstance(page, dict) else 612
                height = page.get("height", 792) if isinstance(page, dict) else 792

            pages.append(
                {
                    "page_index": page_no,
                    "width": width,
                    "height": height,
                    "rotation": 0,
                }
            )

        # Build elements array
        elements = []
        char_offset = 0

        for item in doc.texts:
            # prov is a list of ProvenanceItem in newer docling versions
            prov_list = item.prov if hasattr(item, "prov") else []
            if not prov_list:
                continue

            # Get page info from first provenance item
            prov = prov_list[0]
            page_idx = prov.page_no - 1  # Docling uses 1-based

            # Get text content
            item_text = item.text if hasattr(item, "text") else ""

            if not item_text:
                continue

            # Calculate character span
            char_start = char_offset
            char_end = char_offset + len(item_text)
            char_offset = char_end

            # Get element type
            element_type = _get_element_type(item)

            # Build element
            element: dict[str, Any] = {
                "element_id": f"{doc_id}_{element_type}_{len(elements):03d}",
                "type": element_type,
                "page_index": page_idx,
                "char_span": [char_start, char_end],
                "text": item_text,
                "content": {"kind": "text"},
            }

            # Add bbox if available
            if prov and hasattr(prov, "bbox"):
                page_height = (
                    pages[page_idx]["height"] if page_idx < len(pages) else 792
                )
                element["bbox"] = {
                    "x0": float(prov.bbox.l),
                    "y0": float(prov.bbox.t),
                    "x1": float(prov.bbox.r),
                    "y1": float(prov.bbox.b),
                }

            # Add heading level if applicable
            if element_type == "heading":
                level = _get_heading_level(item)
                if level is not None:
                    element["level"] = level

            elements.append(element)

        # Build output
        output = {
            "schema_version": "1.0.0",
            "parser_version": f"docling-{result._version if hasattr(result, '_version') else 'unknown'}",
            "parsed_at": result._time_started
            if hasattr(result, "_time_started")
            else "2025-01-01T00:00:00Z",
            "source": {
                "doc_id": doc_id,
                "filename": pdf_path.name,
                "mime_type": "application/pdf",
                "sha256": sha256,
                "page_count": len(pages),
                "language": "en",
            },
            "pages": pages,
            "elements": elements,
            "warnings": [],
        }

        return output

    except Exception as e:
        raise RuntimeError(f"Failed to parse {pdf_path}: {e}") from e
