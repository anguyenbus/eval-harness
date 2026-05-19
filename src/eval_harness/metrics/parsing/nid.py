"""NID (Normalized Indel Distance) metric for reading order evaluation.

Derived from opendataloader-bench. Measures reading order similarity
using fuzzy string matching on document text.

NID: Full document text similarity (includes tables).
NID-S: Structure-only (tables stripped out).
"""

import re

from rapidfuzz import fuzz


def _normalize(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


_HTML_TABLE_PATTERN = re.compile(r"<table[^>]*?>.*?</table>", re.IGNORECASE | re.DOTALL)


def _strip_tables(text: str) -> str:
    """Remove HTML table tags from text."""
    without_html = _HTML_TABLE_PATTERN.sub(" ", text)
    return without_html


def evaluate_reading_order(
    gt: str,
    pred: str,
) -> tuple[float | None, float | None]:
    """Evaluate reading order using NID metric.

    Args:
        gt: Ground truth markdown string.
        pred: Predicted markdown string.

    Returns:
        Tuple of (nid_score, nid_s_score) in range [0.0, 1.0].
        Returns (None, None) if ground truth is empty.

    """
    gt_normalized = _normalize(gt)
    gt_stripped = _strip_tables(gt or "")
    gt_stripped_normalized = _normalize(gt_stripped)

    if not gt_normalized:
        return None, None

    pred_normalized = _normalize(pred)
    pred_stripped = _strip_tables(pred or "")
    pred_stripped_normalized = _normalize(pred_stripped)

    nid_score = fuzz.ratio(gt_normalized, pred_normalized) / 100.0
    nid_s_score = fuzz.ratio(gt_stripped_normalized, pred_stripped_normalized) / 100.0

    return nid_score, nid_s_score


def nid_score(gt: str, pred: str) -> float:
    """Calculate NID score (full text similarity).

    Args:
        gt: Ground truth markdown string.
        pred: Predicted markdown string.

    Returns:
        NID score in [0.0, 1.0]. Returns 0.0 if ground truth empty.

    """
    nid, _ = evaluate_reading_order(gt, pred)
    return nid if nid is not None else 0.0


def nid_s_score(gt: str, pred: str) -> float:
    """Calculate NID-S score (structure-only, tables stripped).

    Args:
        gt: Ground truth markdown string.
        pred: Predicted markdown string.

    Returns:
        NID-S score in [0.0, 1.0]. Returns 0.0 if ground truth empty.

    """
    _, nid_s = evaluate_reading_order(gt, pred)
    return nid_s if nid_s is not None else 0.0


# Legacy aliases for compatibility
def normalized_indel_distance(predicted: list, gold: list) -> float:
    """Legacy: Calculate edit distance between sequences.

    DEPRECATED: Use nid_score() with markdown strings instead.
    This is kept for backward compatibility.
    """
    if not predicted and not gold:
        return 0.0
    if not predicted or not gold:
        return 1.0

    from rapidfuzz.distance import Levenshtein

    distance = Levenshtein.distance(predicted, gold)
    max_len = max(len(predicted), len(gold))
    return distance / max_len if max_len > 0 else 0.0
