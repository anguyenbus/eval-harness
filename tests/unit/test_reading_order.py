"""Tests for reading order metric (Kendall tau)."""

from eval_harness.metrics.parsing.reading_order import kendall_tau


class TestReadingOrder:
    """Test suite for Kendall tau correlation for reading order."""

    def test_identical_order(self):
        """Test Kendall tau of 1.0 for identical order."""
        predicted = ["A", "B", "C", "D"]
        gold = ["A", "B", "C", "D"]

        score = kendall_tau(predicted, gold)
        assert score == 1.0

    def test_reversed_order(self):
        """Test Kendall tau of -1.0 for completely reversed order."""
        predicted = ["A", "B", "C", "D"]
        gold = ["D", "C", "B", "A"]

        score = kendall_tau(predicted, gold)
        assert score == -1.0

    def test_partial_disagreement(self):
        """Test Kendall tau for partial order disagreement."""
        predicted = ["A", "B", "C", "D"]
        gold = ["A", "C", "B", "D"]  # B and C swapped

        score = kendall_tau(predicted, gold)
        # Should be between -1 and 1
        assert -1 < score < 1

    def test_single_element(self):
        """Test Kendall tau with single element lists."""
        score = kendall_tau(["A"], ["A"])
        assert score == 1.0

    def test_empty_lists(self):
        """Test Kendall tau with empty lists."""
        score = kendall_tau([], [])
        assert score == 1.0

    def test_different_elements(self):
        """Test Kendall tau with completely different elements."""
        predicted = ["A", "B", "C"]
        gold = ["X", "Y", "Z"]  # No overlap

        score = kendall_tau(predicted, gold)
        # Since no overlap, the calculation should handle it gracefully
        assert -1 <= score <= 1

    def test_deterministic_behavior(self):
        """Test that same inputs produce same output."""
        predicted = ["A", "C", "B", "D"]
        gold = ["A", "B", "C", "D"]

        score1 = kendall_tau(predicted, gold)
        score2 = kendall_tau(predicted, gold)
        assert score1 == score2

    def test_longer_lists(self):
        """Test Kendall tau with longer element sequences."""
        predicted = list(range(10))
        gold = list(range(10))

        score = kendall_tau(predicted, gold)
        assert score == 1.0

    def test_one_swap(self):
        """Test Kendall tau with exactly one pair swapped."""
        predicted = ["A", "B", "C", "D", "E"]
        gold = ["A", "C", "B", "D", "E"]  # B and C swapped

        score = kendall_tau(predicted, gold)
        # For 5 items with 1 swap: tau = 1 - 4*disagree / (n*(n-1))
        # Disagreements: (B,C), (C,B), plus comparisons involving C now
        # Expected: around 0.6
        assert 0 < score < 1.0
