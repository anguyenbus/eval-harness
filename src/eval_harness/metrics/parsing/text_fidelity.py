"""Character-level F1 score for text fidelity comparison.

This metric measures the character-level overlap between predicted and gold text,
useful for evaluating OCR and text extraction quality.
"""


def text_f1_score(predicted: str, gold: str) -> float:
    """Calculate character-level F1 score between predicted and gold text.

    F1 score is the harmonic mean of precision and recall at the character level.
    Uses character set intersection for counting.

    Args:
        predicted: The predicted/extracted text.
        gold: The ground truth reference text.

    Returns:
        Float between 0.0 and 1.0, where 1.0 is perfect match.
        Returns 1.0 for empty string comparison (both empty).

    Examples:
        >>> text_f1_score("hello", "hello")
        1.0
        >>> text_f1_score("hello", "hello world")
        0.625
        >>> text_f1_score("abc", "xyz")
        0.0

    """
    # Handle empty string case
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0

    # Count character occurrences
    pred_chars = {}
    gold_chars = {}

    for c in predicted:
        pred_chars[c] = pred_chars.get(c, 0) + 1

    for c in gold:
        gold_chars[c] = gold_chars.get(c, 0) + 1

    # Calculate intersection (minimum count for each character)
    intersection = 0
    for char in gold_chars:
        pred_count = pred_chars.get(char, 0)
        gold_count = gold_chars[char]
        intersection += min(pred_count, gold_count)

    # Calculate precision and recall
    pred_total = len(predicted)
    gold_total = len(gold)

    if pred_total == 0 or gold_total == 0:
        return 0.0

    precision = intersection / pred_total
    recall = intersection / gold_total

    # Calculate F1
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
