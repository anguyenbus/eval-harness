"""
Convert parser_output JSON schema to Markdown for evaluation.

This module converts the structured parser_output format (elements array)
into Markdown strings so we can use opendataloader-bench metrics (NID, TEDS, MHS).
"""

from typing import Any


def elements_to_markdown(elements: list[dict[str, Any]]) -> str:
    """
    Convert parser_output elements array to Markdown string.

    Args:
        elements: List of element dicts from parser_output schema.

    Returns:
        Markdown string representation.

    """
    lines = []

    for elem in elements:
        elem_type = elem.get("type")
        text = elem.get("text", "")

        if not text:
            continue

        if elem_type == "heading":
            level = elem.get("level", 1)
            lines.append(f"{'#' * level} {text}")
            lines.append("")  # Blank line after heading

        elif elem_type == "paragraph":
            lines.append(text)
            lines.append("")  # Blank line after paragraph

        elif elem_type == "list":
            _ = elem.get("content", {}).get("ordered", False)
            # List container itself doesn't add text
            # list_item elements with parent_id will be rendered separately

        elif elem_type == "list_item":
            # Simple rendering - just the text
            # Real implementation would handle nesting via parent_id
            lines.append(f"- {text}")
            lines.append("")

        elif elem_type == "table":
            content = elem.get("content", {})
            if content.get("kind") == "table":
                lines.append(_table_to_markdown(content))
            else:
                lines.append(text)
            lines.append("")

        elif elem_type == "figure":
            alt_text = elem.get("content", {}).get("alt_text", "")
            if alt_text:
                lines.append(f"![{alt_text}]")
            else:
                lines.append(text)
            lines.append("")

        elif elem_type == "code_block":
            lines.append("```")
            lines.append(text)
            lines.append("```")
            lines.append("")

        elif elem_type in (
            "caption",
            "footnote",
            "header",
            "footer",
            "page_number",
            "equation",
        ):
            lines.append(text)
            lines.append("")

        else:
            # Default: just output text
            lines.append(text)
            lines.append("")

    return "\n".join(lines).strip()


def _table_to_markdown(table_content: dict[str, Any]) -> str:
    """
    Convert table content to Markdown table.

    Args:
        table_content: TableContent dict with rows, cols, cells.

    Returns:
        Markdown table string.

    """
    rows = table_content.get("rows", 0)
    cols = table_content.get("cols", 0)
    cells = table_content.get("cells", [])
    _ = table_content.get("header_rows", 1)

    if rows == 0 or cols == 0:
        return ""

    # Build cell grid
    grid: dict[tuple[int, int], str] = {}
    for cell in cells:
        row = cell["row"]
        col = cell["col"]
        text = cell.get("text", "")
        grid[(row, col)] = text

    # Generate markdown table
    lines = []

    for row_idx in range(rows):
        row_cells = []
        for col_idx in range(cols):
            cell_text = grid.get((row_idx, col_idx), "")
            row_cells.append(cell_text)

        if row_idx == 0:
            # Header row + separator
            lines.append("| " + " | ".join(row_cells) + " |")
            lines.append("| " + " | ".join(["---"] * cols) + " |")
        else:
            lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


def parser_output_to_markdown(parser_output: dict[str, Any]) -> str:
    """
    Convert full parser_output dict to Markdown.

    Args:
        parser_output: Full parser_output schema dict.

    Returns:
        Markdown string representation.

    """
    elements = parser_output.get("elements", [])
    return elements_to_markdown(elements)
