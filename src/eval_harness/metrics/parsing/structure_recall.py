"""Structure recall metric for element detection coverage.

This metric measures how many of the gold element types were detected
by the parser, regardless of count or accuracy within each type.
"""

from typing import Any


def structure_recall(
    predicted_elements: list[dict[str, Any]],
    gold_elements: list[dict[str, Any]],
    element_types: list[str],
) -> float:
    """Calculate structure recall: fraction of gold element types detected.

    For each element type present in gold, check if at least one element of
    that type appears in predicted. The score is the fraction of gold types
    that were detected.

    Args:
        predicted_elements: List of predicted elements, each with 'type' field.
        gold_elements: List of gold/reference elements, each with 'type' field.
        element_types: List of all possible element types (24 for OmniDocBench).

    Returns:
        Float between 0.0 and 1.0, where 1.0 means all gold types detected.
        Returns 1.0 if gold_elements is empty.

    Examples:
        >>> gold = [{"type": "text_block"}, {"type": "table"}]
        >>> pred = [{"type": "text_block"}]
        >>> structure_recall(pred, gold, element_types=["text_block", "table"])
        0.5

    """
    if not gold_elements:
        return 1.0

    # Get set of unique element types present in gold
    gold_types = {elem.get("type") for elem in gold_elements if "type" in elem}
    gold_types.discard(None)  # Remove None if present

    if not gold_types:
        return 1.0

    # Get set of unique element types present in predicted
    pred_types = {elem.get("type") for elem in predicted_elements if "type" in elem}
    pred_types.discard(None)

    # Count how many gold types are present in predicted
    detected_types = gold_types & pred_types

    # Calculate recall
    recall = len(detected_types) / len(gold_types)
    return recall
