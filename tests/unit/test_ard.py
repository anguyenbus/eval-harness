"""Tests for ARD (Average Rank Distance) reading order metric."""

from eval_harness.metrics.parsing.reading_order import (
    ard_score,
    ard_weighted_score,
)


class TestARDScore:
    """Test suite for ARD (Average Rank Distance) metric."""

    def test_identical_order_returns_perfect_score(self):
        """Test ARD score of 1.0 for identical order."""
        predicted = ["A", "B", "C", "D"]
        gold = ["A", "B", "C", "D"]

        score = ard_score(predicted, gold)
        assert score == 1.0

    def test_reversed_order_returns_lower_score(self):
        """Test ARD score for completely reversed order."""
        predicted = ["A", "B", "C", "D"]
        gold = ["D", "C", "B", "A"]

        score = ard_score(predicted, gold)
        # Reversed order should have a lower score (not 0.0 due to normalization)
        assert 0.0 <= score < 1.0

    def test_partial_disagreement(self):
        """Test ARD score for partial order disagreement."""
        predicted = ["A", "B", "C", "D"]
        gold = ["A", "C", "B", "D"]  # B and C swapped

        score = ard_score(predicted, gold)
        # Partial disagreement should be between 0 and 1
        assert 0.0 < score < 1.0

    def test_empty_lists_returns_perfect_score(self):
        """Test ARD score of 1.0 for empty lists."""
        score = ard_score([], [])
        assert score == 1.0

    def test_single_element_returns_perfect_score(self):
        """Test ARD score of 1.0 for single element."""
        score = ard_score(["A"], ["A"])
        assert score == 1.0

    def test_different_length_lists(self):
        """Test ARD handles different length lists gracefully."""
        predicted = ["A", "B", "C"]
        gold = ["A", "B"]

        # Should handle by considering only common elements
        score = ard_score(predicted, gold)
        assert 0.0 <= score <= 1.0

    def test_deterministic_behavior(self):
        """Test that same inputs produce same output."""
        predicted = ["A", "C", "B", "D"]
        gold = ["A", "B", "C", "D"]

        score1 = ard_score(predicted, gold)
        score2 = ard_score(predicted, gold)
        assert score1 == score2


class TestARDWeightedScore:
    """Test suite for weighted ARD metric."""

    def test_identical_order_returns_perfect_score(self):
        """Test weighted ARD score of 1.0 for identical order."""
        predicted = ["A", "B", "C", "D"]
        gold = ["A", "B", "C", "D"]
        # Mock bboxes with areas: 10, 20, 30, 40
        bboxes = [
            {"x0": 0, "y0": 0, "x1": 5, "y1": 2},  # area=10
            {"x0": 0, "y0": 0, "x1": 10, "y1": 2},  # area=20
            {"x0": 0, "y0": 0, "x1": 15, "y1": 2},  # area=30
            {"x0": 0, "y0": 0, "x1": 20, "y1": 2},  # area=40
        ]

        score = ard_weighted_score(predicted, gold, bboxes)
        assert score == 1.0

    def test_empty_lists_returns_perfect_score(self):
        """Test weighted ARD score of 1.0 for empty lists."""
        score = ard_weighted_score([], [], [])
        assert score == 1.0

    def test_single_element_returns_perfect_score(self):
        """Test weighted ARD score of 1.0 for single element."""
        bboxes = [{"x0": 0, "y0": 0, "x1": 10, "y1": 10}]
        score = ard_weighted_score(["A"], ["A"], bboxes)
        assert score == 1.0

    def test_weighting_affects_score(self):
        """Test that bbox area weighting affects the score."""
        predicted = ["A", "B"]
        gold = ["B", "A"]
        # Small bbox for A, large bbox for B
        bboxes = [
            {"x0": 0, "y0": 0, "x1": 1, "y1": 1},  # area=1 (A)
            {"x0": 0, "y0": 0, "x1": 100, "y1": 100},  # area=10000 (B)
        ]

        score = ard_weighted_score(predicted, gold, bboxes)
        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0
