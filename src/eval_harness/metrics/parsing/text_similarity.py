"""
Text similarity metrics for OCR and text extraction evaluation.

This module implements BLEU, METEOR, character-level edit distance,
and word-level edit distance metrics following docling-eval patterns.
All scores are normalized to [0.0, 1.0] range where 1.0 is perfect match.
"""

from __future__ import annotations

import nltk
from beartype import beartype
from beartype.typing import Final
from rapidfuzz.distance import Levenshtein

# Constants for text similarity
_MAX_SCORE: Final[float] = 1.0
_MIN_SCORE: Final[float] = 0.0
_BLEU_WEIGHTS: Final[tuple[float, ...]] = (0.25, 0.25, 0.25, 0.25)

# Cached model instances (loaded once, reused)
_wordnet_checked = False
_punkt_checked = False


def _ensure_nltk_data() -> None:
    """
    Ensure required NLTK data is downloaded.

    NOTE: This function is idempotent and safe to call multiple times.
    It downloads the minimum required data for METEOR and tokenization.
    """
    global _wordnet_checked, _punkt_checked

    if not _punkt_checked:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        _punkt_checked = True

    if not _wordnet_checked:
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        _wordnet_checked = True


@beartype
def bleu_score(gt: str, pred: str) -> float:
    """
    Calculate BLEU score between ground truth and predicted text.

    Uses sacrebleu for fast computation (100x faster than HuggingFace evaluate wrapper).

    Args:
        gt: Ground truth reference text.
        pred: Predicted/extracted text.

    Returns:
        Float in [0.0, 1.0] where 1.0 is perfect match.
        Returns 1.0 for empty string comparison (both empty).

    Examples:
        >>> bleu_score("hello world", "hello world")
        1.0  # Or very close to 1.0 for short texts
        >>> bleu_score("hello world", "hello")
        0.0  # BLEU penalizes missing words heavily

    """
    # Handle empty strings
    if not gt and not pred:
        return _MAX_SCORE

    if not gt or not pred:
        return _MIN_SCORE

    try:
        import sacrebleu

        # sentence_bleu for single sentence/document comparison
        # corpus_bleu is designed for corpus-level stats, returns 0 for single sentences
        bleu = sacrebleu.sentence_bleu(pred, [gt])
        return bleu.score / 100.0
    except Exception:
        # Fallback to 0.0 if evaluation fails
        return _MIN_SCORE


@beartype
def meteor_score(gt: str, pred: str) -> float:
    """
    Calculate METEOR score between ground truth and predicted text.

    METEOR is based on harmonic mean of unigram precision and recall,
    with penalty for fragmentation.

    Args:
        gt: Ground truth reference text.
        pred: Predicted/extracted text.

    Returns:
        Float in [0.0, 1.0] where 1.0 is perfect match.
        Returns 0.0 if either string is empty.

    Examples:
        >>> meteor_score("hello world", "hello world")
        0.9375  # METEOR typically gives close to 1.0 for perfect matches

    """
    # Ensure NLTK data is available
    _ensure_nltk_data()

    # Handle empty strings
    if not gt and not pred:
        return _MAX_SCORE

    if not gt or not pred:
        return _MIN_SCORE

    try:
        from nltk.tokenize import word_tokenize

        # Import with alias to avoid name collision
        from nltk.translate import meteor_score as nltk_meteor

        gt_tokens = word_tokenize(gt)
        pred_tokens = word_tokenize(pred)

        # METEOR requires a list of reference sentences
        score = nltk_meteor.meteor_score([gt_tokens], pred_tokens)
        return float(score)
    except Exception:
        # Fallback to 0.0 if evaluation fails
        return _MIN_SCORE


@beartype
def char_edit_distance(gt: str, pred: str) -> float:
    """
    Calculate normalized character-level edit distance.

    Uses rapidfuzz Levenshtein distance normalized by maximum length.

    Args:
        gt: Ground truth reference text.
        pred: Predicted/extracted text.

    Returns:
        Float in [0.0, 1.0] where 1.0 is perfect match (0 distance).
        Returns 1.0 for empty string comparison (both empty).

    Examples:
        >>> char_edit_distance("hello", "hello")
        1.0
        >>> char_edit_distance("hello", "hallo")  # 1 char diff
        0.8

    """
    # Handle empty strings
    if not gt and not pred:
        return _MAX_SCORE

    if not gt or not pred:
        return _MIN_SCORE

    # Calculate Levenshtein distance
    distance = Levenshtein.distance(gt, pred)
    max_len = max(len(gt), len(pred))

    if max_len == 0:
        return _MAX_SCORE

    # Normalize to [0, 1] where 1.0 is perfect match
    normalized_distance = 1.0 - (distance / max_len)
    return max(_MIN_SCORE, min(_MAX_SCORE, normalized_distance))


@beartype
def word_edit_distance(gt: str, pred: str) -> float:
    """
    Calculate normalized word-level edit distance.

    Tokenizes text using NLTK word_tokenize, then computes
    Levenshtein distance normalized by maximum token count.

    Args:
        gt: Ground truth reference text.
        pred: Predicted/extracted text.

    Returns:
        Float in [0.0, 1.0] where 1.0 is perfect match (0 distance).
        Returns 1.0 for empty string comparison (both empty).

    Examples:
        >>> word_edit_distance("hello world", "hello world")
        1.0
        >>> word_edit_distance("hello world", "hello earth")
        0.5  # 1 word diff out of 2

    """
    # Ensure NLTK data is available
    _ensure_nltk_data()

    # Handle empty strings
    if not gt and not pred:
        return _MAX_SCORE

    if not gt or not pred:
        return _MIN_SCORE

    try:
        from nltk.tokenize import word_tokenize

        gt_tokens = word_tokenize(gt)
        pred_tokens = word_tokenize(pred)

        # Calculate word-level Levenshtein distance
        distance = Levenshtein.distance(gt_tokens, pred_tokens)
        max_len = max(len(gt_tokens), len(pred_tokens))

        if max_len == 0:
            return _MAX_SCORE

        # Normalize to [0, 1] where 1.0 is perfect match
        normalized_distance = 1.0 - (distance / max_len)
        return max(_MIN_SCORE, min(_MAX_SCORE, normalized_distance))
    except Exception:
        # Fallback to character-level distance if tokenization fails
        return char_edit_distance(gt, pred)
