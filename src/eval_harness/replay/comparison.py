"""
Comparison logic for paired statistical analysis.

This module provides statistical comparison functions for candidate
versus baseline adapter evaluation using Wilcoxon signed-rank test
and effect size calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from beartype import beartype

# Constants
DEFAULT_ALPHA: Final[float] = 0.05

# Cliff's Delta effect size thresholds:
# - negligible: < 0.15
# - small: 0.15 - 0.33
# - medium: 0.33 - 0.47
# - large: >= 0.47
#
# DEFAULT_EFFECT_SIZE_THRESHOLD set to 0.15 (small) for practical significance.
# Adjust based on your use case:
# - 0.15: Detect small but meaningful improvements
# - 0.33: Only care about medium+ effects
# - 0.47: Only care about large effects
DEFAULT_EFFECT_SIZE_THRESHOLD: Final[float] = 0.15


@beartype
@dataclass(frozen=True)
class ComparisonResult:
    """
    Result of paired statistical comparison.

    Attributes:
        statistic: Test statistic value.
        p_value: P-value from statistical test.
        effect_size: Effect size (Cliff's Delta or Cohen's D).
        pass_fail: Whether the comparison passes the threshold.
        winner: Which adapter won ("candidate", "baseline", or "tie").

    """

    statistic: float
    p_value: float
    effect_size: float
    pass_fail: bool
    winner: str


@beartype
def paired_comparison(
    candidate_scores: list[float],
    baseline_scores: list[float],
    alpha: float = DEFAULT_ALPHA,
    effect_size_threshold: float = DEFAULT_EFFECT_SIZE_THRESHOLD,
) -> ComparisonResult:
    """
    Perform paired statistical comparison.

    Uses Wilcoxon signed-rank test for paired comparison and
    computes Cliff's Delta effect size.

    Args:
        candidate_scores: Scores from candidate adapter.
        baseline_scores: Scores from baseline adapter.
        alpha: Significance threshold for p-value (default: 0.05).
        effect_size_threshold: Threshold for meaningful effect size.
            Default: 0.15 (small effect per Cliff's Delta).
            Options: 0.15 (small), 0.33 (medium), 0.47 (large).

    Returns:
        ComparisonResult with test statistics and determination.
        - pass_fail: True if p_value < alpha AND |effect_size| >= threshold
        - winner: "candidate" if effect_size > 0, "baseline" if < 0, "tie" if near 0

    Raises:
        ValueError: If score lists have different lengths.

    """
    if len(candidate_scores) != len(baseline_scores):
        raise ValueError("Score lists must have the same length")

    if len(candidate_scores) == 0:
        return ComparisonResult(
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            pass_fail=False,
            winner="tie",
        )

    # Compute Wilcoxon signed-rank test
    statistic, p_value = _wilcoxon_test(candidate_scores, baseline_scores)

    # Compute effect size (Cliff's Delta)
    effect_size = _cliffs_delta(candidate_scores, baseline_scores)

    # Determine winner
    if abs(effect_size) < effect_size_threshold:
        winner = "tie"
    elif effect_size > 0:
        winner = "candidate"
    else:
        winner = "baseline"

    # Determine pass/fail
    pass_fail = p_value < alpha and abs(effect_size) >= effect_size_threshold

    return ComparisonResult(
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        pass_fail=pass_fail,
        winner=winner,
    )


@beartype
def _wilcoxon_test(
    candidate_scores: list[float],
    baseline_scores: list[float],
) -> tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test.

    Args:
        candidate_scores: Scores from candidate adapter.
        baseline_scores: Scores from baseline adapter.

    Returns:
        Tuple of (test_statistic, p_value).

    """
    try:
        from scipy.stats import wilcoxon

        statistic, p_value = wilcoxon(candidate_scores, baseline_scores)
        return float(statistic), float(p_value)
    except ImportError:
        # Fallback: simple sign test if scipy not available
        n = len(candidate_scores)
        n_positive = sum(
            1 for c, b in zip(candidate_scores, baseline_scores, strict=True) if c > b
        )
        n_negative = sum(
            1 for c, b in zip(candidate_scores, baseline_scores, strict=True) if c < b
        )

        # Simple approximation
        statistic = min(n_positive, n_negative)
        p_value = 2.0 ** (-abs(n_positive - n_negative) - 1) if n > 0 else 1.0

        return float(statistic), float(p_value)


@beartype
def _cliffs_delta(
    candidate_scores: list[float],
    baseline_scores: list[float],
) -> float:
    """
    Compute Cliff's Delta effect size.

    Cliff's Delta measures the degree of overlap between two distributions.
    Values range from -1 to 1, where:
    - 1 indicates all candidate scores > all baseline scores
    - -1 indicates all candidate scores < all baseline scores
    - 0 indicates complete overlap

    Args:
        candidate_scores: Scores from candidate adapter.
        baseline_scores: Scores from baseline adapter.

    Returns:
        Cliff's Delta effect size.

    """
    n1 = len(candidate_scores)
    n2 = len(baseline_scores)

    if n1 == 0 or n2 == 0:
        return 0.0

    # Count comparisons
    greater = 0
    less = 0

    for c in candidate_scores:
        for b in baseline_scores:
            if c > b:
                greater += 1
            elif c < b:
                less += 1

    # Compute Cliff's Delta
    total = n1 * n2
    delta = (greater - less) / total if total > 0 else 0.0

    return delta
