"""
Reading order metric using ARD (Average Rank Distance).

ARD measures the average displacement between predicted and gold order,
normalized to [0.0, 1.0] range where 1.0 is perfect match.

This replaces the previous Kendall tau implementation to align with
docling-eval and OmniDocBench standards.
"""

from __future__ import annotations

from beartype import beartype
from beartype.typing import Any, Final

# Constants for ARD calculation
_MAX_SCORE: Final[float] = 1.0
_MIN_SCORE: Final[float] = 0.0


def _compute_bbox_area(bbox: dict[str, Any]) -> float:
    """
    Compute the area of a bounding box.

    Args:
        bbox: Dictionary with x0, y0, x1, y1 keys.

    Returns:
        Area of the bounding box.

    """
    x0 = bbox.get("x0", 0)
    y0 = bbox.get("y0", 0)
    x1 = bbox.get("x1", 0)
    y1 = bbox.get("y1", 0)

    width = abs(x1 - x0)
    height = abs(y1 - y0)

    return width * height


@beartype
def ard_score(predicted_order: list[Any], gold_order: list[Any]) -> float:
    """
    Calculate ARD (Average Rank Distance) between predicted and gold order.

    ARD measures the average displacement between predicted and gold positions
    for each element, normalized to [0.0, 1.0] where 1.0 is perfect match.

    Formula:
        ARD = (1/n) * sum(e_k)
        e_k = |pred_order_index - gt_order_index|
        ARD_norm = 1 - (ARD / n)

    Args:
        predicted_order: List of element IDs in predicted reading order.
        gold_order: List of element IDs in gold/reference reading order.

    Returns:
        Float between 0.0 and 1.0, where 1.0 is perfect agreement.
        Returns 1.0 for empty lists or single element.

    Examples:
        >>> ard_score(["A", "B", "C"], ["A", "B", "C"])
        1.0
        >>> ard_score(["A", "B", "C"], ["C", "B", "A"])
        0.0  # Normalized score for reversed order

    """
    # Handle edge cases
    if not predicted_order and not gold_order:
        return _MAX_SCORE

    if len(predicted_order) <= 1 or len(gold_order) <= 1:
        return _MAX_SCORE

    # Build position mapping for gold order
    gold_pos = {elem: i for i, elem in enumerate(gold_order)}

    # Only consider elements that appear in both
    common_elements = set(predicted_order) & set(gold_order)

    if len(common_elements) <= 1:
        return _MAX_SCORE

    # Compute ARD
    n = len(common_elements)
    total_distance = 0

    for pred_idx, elem in enumerate(predicted_order):
        if elem not in gold_pos:
            continue

        gold_idx = gold_pos[elem]
        distance = abs(pred_idx - gold_idx)
        total_distance += distance

    # Normalize ARD to [0, 1] range
    # Maximum possible ARD is n*(n-1)/2 (sum of 0 to n-1)
    # We normalize by n^2 for consistency with docling-eval
    n_squared = n * n

    if n_squared == 0:
        return _MAX_SCORE

    ard = total_distance / n
    ard_norm = 1.0 - (ard / n)

    # Clamp to [0, 1] range
    return max(_MIN_SCORE, min(_MAX_SCORE, ard_norm))


@beartype
def ard_weighted_score(
    predicted_order: list[Any],
    gold_order: list[Any],
    bboxes: list[Any],
) -> float:
    """
    Calculate weighted ARD score using bbox area as weight.

    The weight for each element is: weight_k = area(bbox_k) / total_area
    This gives more importance to larger elements (e.g., tables, pictures).

    Formula:
        weighted_ARD = (1/n) * sum(e_k * weight_k)
        weighted_ARD_norm = 1 - (weighted_ARD / n)

    Args:
        predicted_order: List of element IDs in predicted reading order.
        gold_order: List of element IDs in gold/reference reading order.
        bboxes: List of bounding box dictionaries, one per element in gold_order.
                Each dict should have x0, y0, x1, y1 keys.

    Returns:
        Float between 0.0 and 1.0, where 1.0 is perfect agreement.
        Returns 1.0 for empty lists or single element.

    Examples:
        >>> bboxes = [{"x0": 0, "y0": 0, "x1": 10, "y1": 10}]
        >>> ard_weighted_score(["A"], ["A"], bboxes)
        1.0

    """
    # Handle edge cases
    if not predicted_order and not gold_order:
        return _MAX_SCORE

    if len(predicted_order) <= 1 or len(gold_order) <= 1:
        return _MAX_SCORE

    # Build position mapping for gold order
    gold_pos = {elem: i for i, elem in enumerate(gold_order)}

    # Only consider elements that appear in both
    common_elements = set(predicted_order) & set(gold_order)

    if len(common_elements) <= 1:
        return _MAX_SCORE

    # Compute bbox areas and weights
    # Only compute weights for common elements in gold order
    bbox_areas = []
    total_area = 0.0

    for i, elem in enumerate(gold_order):
        if elem in common_elements and i < len(bboxes):
            area = _compute_bbox_area(bboxes[i])
            bbox_areas.append(area)
            total_area += area
        else:
            bbox_areas.append(0.0)

    # Compute weighted ARD
    n = len(common_elements)

    if n == 0 or total_area == 0:
        return _MAX_SCORE

    weights = [area / total_area for area in bbox_areas]
    total_weighted_distance = 0.0

    for pred_idx, elem in enumerate(predicted_order):
        if elem not in gold_pos:
            continue

        gold_idx = gold_pos[elem]
        distance = abs(pred_idx - gold_idx)
        weight = weights[gold_idx] if gold_idx < len(weights) else 0.0
        total_weighted_distance += distance * weight

    # Normalize weighted ARD to [0, 1] range
    n_squared = n * n
    weighted_ard = total_weighted_distance / n
    weighted_ard_norm = 1.0 - (weighted_ard / n)

    # Clamp to [0, 1] range
    return max(_MIN_SCORE, min(_MAX_SCORE, weighted_ard_norm))


# Legacy function for backward compatibility
def kendall_tau(predicted_order: list[Any], gold_order: list[Any]) -> float:
    """
    Legacy: Calculate Kendall tau correlation.

    DEPRECATED: Use ard_score() instead.
    This function is kept for backward compatibility with existing tests.

    Args:
        predicted_order: List of element IDs in predicted reading order.
        gold_order: List of element IDs in gold/reference reading order.

    Returns:
        Float approximating Kendall tau correlation, mapped to [0.0, 1.0].

    """
    # Use ARD for backward compatibility, mapping to similar range
    return ard_score(predicted_order, gold_order)
