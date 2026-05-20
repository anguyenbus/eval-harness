"""
TEDS (Tree Edit Distance Similarity) metric for table evaluation.

Derived from opendataloader-bench and the PubTabNet paper.
TEDS measures table structure similarity using tree edit distance.

TEDS: Structure + content similarity.
TEDS-S: Structure-only similarity.
"""

import re
from html import unescape

from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree
from rapidfuzz.distance import Levenshtein


def _normalize(text: str) -> str:
    """Normalize HTML text content."""
    result = unescape(text)
    result = re.sub(r"<br\s*/?>", "\n", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _convert_headers_to_cells(node: etree.Element) -> None:
    """Convert th elements to td for uniformity."""
    for header in node.xpath(".//th"):
        header.tag = "td"


class TableTree(Tree):
    """Light wrapper around Tree for table metadata."""

    def __init__(
        self,
        tag: str,
        colspan: int | None = None,
        rowspan: int | None = None,
        content: list[str] | None = None,
        *children: "TableTree",
    ) -> None:
        """Initialize table tree node with tag, span info, content, and children."""
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self) -> str:
        """Show tree using brackets notation."""
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return f"{{{result}}}"


class CustomConfig(Config):
    """Custom Configuration for APTED."""

    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value."""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1."""
        return Levenshtein.distance(*sequences) / float(self.maximum(*sequences))

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        """Compare attributes of trees."""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td" and (node1.content or node2.content):
            content1 = "".join(node1.content or [])
            content2 = "".join(node2.content or [])
            normalized_content1 = _normalize(content1)
            normalized_content2 = _normalize(content2)
            if not normalized_content1 and not normalized_content2:
                return 0.0
            return self.normalized_distance(normalized_content1, normalized_content2)
        return 0.0


class TEDSEvaluator:
    """Tree Edit Distance based Similarity evaluator."""

    def __init__(
        self, structure_only: bool = False, n_jobs: int = 1, ignore_nodes=None
    ):
        """Initialize TEDS evaluator with structure-only mode and job parallelism."""
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenize table cells."""
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Convert HTML tree to format required by apted."""
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
            )
        else:
            new_node = TableTree(node.tag, None, None, None)
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred: str, true: str) -> float:
        """Compute TEDS score between prediction and ground truth."""
        if (not pred) or (not true):
            return 0.0

        from lxml import html

        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)

        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]
            _convert_headers_to_cells(pred)
            _convert_headers_to_cells(true)
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(
                tree_pred, tree_true, CustomConfig()
            ).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        return 0.0


def _markdown_table_to_html(table_markdown: str) -> str:
    """Convert markdown table to HTML table."""
    if not table_markdown.strip():
        return ""

    lines = [
        line.strip() for line in table_markdown.strip().splitlines() if line.strip()
    ]
    if len(lines) < 2:
        return ""

    # Check if separator line exists
    if not lines[1].startswith("|"):
        return ""

    html_lines = ["<table>"]

    for i, line in enumerate(lines):
        if line.startswith("|---"):
            continue  # Skip separator

        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        tag = "th" if i == 0 else "td"

        html_lines.append("<tr>")
        for cell in cells:
            html_lines.append(f"<{tag}>{cell}</{tag}>")
        html_lines.append("</tr>")

    html_lines.append("</table>")
    return "\n".join(html_lines)


def _extract_tables_from_markdown(markdown: str) -> list[str]:
    """Extract table sections from markdown."""
    tables = []
    lines = markdown.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and i + 1 < len(lines):
            # Check if next line is separator
            next_line = lines[i + 1].strip()
            if next_line.startswith("|---"):
                # Found a table
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                tables.append("\n".join(table_lines))
                continue
        i += 1

    return tables


def _wrap_tables_in_html(tables: list[str]) -> str:
    """Wrap multiple HTML tables in body tags."""
    body_content = "\n".join(tables)
    return f"<html><body>\n{body_content}\n</body></html>"


def evaluate_table(
    gt: str,
    pred: str,
) -> tuple[float | None, float | None]:
    """
    Evaluate predicted table markup against ground truth using TEDS.

    Args:
        gt: Ground truth markdown string.
        pred: Predicted markdown string.

    Returns:
        Tuple of (teds_score, teds_s_score) in [0.0, 1.0].
        Returns (None, None) when ground truth has no tables.

    """
    gt_tables = _extract_tables_from_markdown(gt)
    pred_tables = _extract_tables_from_markdown(pred)

    if not gt_tables:
        return None, None
    if not pred_tables:
        return 0.0, 0.0

    # Convert markdown tables to HTML
    gt_html_tables = [_markdown_table_to_html(t) for t in gt_tables]
    pred_html_tables = [_markdown_table_to_html(t) for t in pred_tables]

    gt_data = _wrap_tables_in_html(gt_html_tables)
    pred_data = _wrap_tables_in_html(pred_html_tables)

    structure_evaluator = TEDSEvaluator(structure_only=True)
    teds_s_score = structure_evaluator.evaluate(pred_data, gt_data)

    content_evaluator = TEDSEvaluator(structure_only=False)
    teds_score = content_evaluator.evaluate(pred_data, gt_data)

    return teds_score, teds_s_score


def teds_score(gt: str, pred: str) -> float:
    """
    Calculate TEDS score (structure + content).

    Args:
        gt: Ground truth markdown string.
        pred: Predicted markdown string.

    Returns:
        TEDS score in [0.0, 1.0]. Returns 0.0 if ground truth empty.

    """
    teds, _ = evaluate_table(gt, pred)
    return teds if teds is not None else 0.0


def teds_s_score(gt: str, pred: str) -> float:
    """
    Calculate TEDS-S score (structure-only).

    Args:
        gt: Ground truth markdown string.
        pred: Predicted markdown string.

    Returns:
        TEDS-S score in [0.0, 1.0]. Returns 0.0 if ground truth empty.

    """
    _, teds_s = evaluate_table(gt, pred)
    return teds_s if teds_s is not None else 0.0


# Legacy alias for backward compatibility
def table_teds(predicted_table: dict, gold_table: dict) -> float:
    """
    Legacy: Calculate simplified table similarity.

    DEPRECATED: Use teds_score() with markdown strings instead.
    This is kept for backward compatibility with existing tests.
    """
    pred_cells = predicted_table.get("cells", [])
    gold_cells = gold_table.get("cells", [])

    if not pred_cells and not gold_cells:
        return 1.0
    if not pred_cells or not gold_cells:
        return 0.0

    pred_cells_dict = {}
    for cell in pred_cells:
        key = (cell.get("row"), cell.get("col"))
        pred_cells_dict[key] = cell.get("text", "")

    gold_cells_dict = {}
    for cell in gold_cells:
        key = (cell.get("row"), cell.get("col"))
        gold_cells_dict[key] = cell.get("text", "")

    all_positions = set(pred_cells_dict.keys()) | set(gold_cells_dict.keys())

    if not all_positions:
        return 1.0

    matches = 0
    for pos in all_positions:
        pred_text = pred_cells_dict.get(pos, "")
        gold_text = gold_cells_dict.get(pos, "")
        if pred_text == gold_text:
            matches += 1

    return matches / len(all_positions)
