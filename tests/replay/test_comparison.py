"""
Tests for replay comparison logic.
"""

import pytest

from eval_harness.replay.comparison import (
    ComparisonResult,
    _cliffs_delta,
    _wilcoxon_test,
    paired_comparison,
)


class TestComparisonLogic:
    """Tests for statistical comparison logic."""

    def test_paired_comparison_basic(self) -> None:
        """Test basic paired comparison."""
        candidate = [0.8, 0.7, 0.9, 0.6, 0.85]
        baseline = [0.7, 0.6, 0.8, 0.5, 0.75]

        result = paired_comparison(candidate, baseline)

        assert isinstance(result, ComparisonResult)
        assert result.p_value >= 0.0
        assert result.p_value <= 1.0
        assert isinstance(result.effect_size, float)

    def test_paired_comparison_different_lengths(self) -> None:
        """Test that different length lists raise ValueError."""
        candidate = [0.8, 0.7, 0.9]
        baseline = [0.7, 0.6]

        with pytest.raises(ValueError):
            paired_comparison(candidate, baseline)

    def test_paired_comparison_empty_lists(self) -> None:
        """Test comparison with empty lists."""
        result = paired_comparison([], [])

        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert result.effect_size == 0.0
        assert result.pass_fail is False
        assert result.winner == "tie"

    def test_paired_comparison_identical_scores(self) -> None:
        """Test comparison with identical scores (should be tie)."""
        scores = [0.8, 0.7, 0.9, 0.6, 0.85]

        result = paired_comparison(scores, scores)

        # Effect size should be 0 for identical scores
        assert abs(result.effect_size) < 0.01
        assert result.winner == "tie"

    def test_paired_comparison_candidate_wins(self) -> None:
        """Test where candidate clearly wins."""
        candidate = [0.9, 0.85, 0.95, 0.88, 0.92]
        baseline = [0.6, 0.55, 0.65, 0.58, 0.62]

        result = paired_comparison(candidate, baseline)

        # Candidate should have higher effect size
        assert result.effect_size > 0
        assert result.winner == "candidate"

    def test_paired_comparison_baseline_wins(self) -> None:
        """Test where baseline clearly wins."""
        candidate = [0.6, 0.55, 0.65, 0.58, 0.62]
        baseline = [0.9, 0.85, 0.95, 0.88, 0.92]

        result = paired_comparison(candidate, baseline)

        # Baseline should have negative effect size
        assert result.effect_size < 0
        assert result.winner == "baseline"

    def test_paired_comparison_custom_thresholds(self) -> None:
        """Test comparison with custom thresholds."""
        candidate = [0.8, 0.75, 0.85]
        baseline = [0.7, 0.65, 0.75]

        result = paired_comparison(
            candidate, baseline, alpha=0.01, effect_size_threshold=0.8
        )

        assert isinstance(result, ComparisonResult)

    def test_cliffs_delta_calculation(self) -> None:
        """Test Cliff's Delta effect size calculation."""
        # No overlap - all candidate > all baseline
        candidate = [0.9, 0.85, 0.95]
        baseline = [0.5, 0.45, 0.55]

        delta = _cliffs_delta(candidate, baseline)

        # Should be close to 1
        assert delta > 0.5

    def test_cliffs_delta_negative(self) -> None:
        """Test Cliff's Delta for baseline > candidate."""
        candidate = [0.5, 0.45, 0.55]
        baseline = [0.9, 0.85, 0.95]

        delta = _cliffs_delta(candidate, baseline)

        # Should be negative
        assert delta < 0

    def test_wilcoxon_test_returns_tuple(self) -> None:
        """Test that Wilcoxon test returns (statistic, p_value)."""
        candidate = [0.8, 0.7, 0.9]
        baseline = [0.7, 0.6, 0.8]

        statistic, p_value = _wilcoxon_test(candidate, baseline)

        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_comparison_result_fields(self) -> None:
        """Test that ComparisonResult has all required fields."""
        result = ComparisonResult(
            statistic=100.0,
            p_value=0.01,
            effect_size=0.5,
            pass_fail=True,
            winner="candidate",
        )

        assert result.statistic == 100.0
        assert result.p_value == 0.01
        assert result.effect_size == 0.5
        assert result.pass_fail is True
        assert result.winner == "candidate"

    def test_comparison_result_is_frozen(self) -> None:
        """Test that ComparisonResult is frozen (immutable)."""
        result = ComparisonResult(
            statistic=100.0,
            p_value=0.01,
            effect_size=0.5,
            pass_fail=True,
            winner="candidate",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.p_value = 0.05
