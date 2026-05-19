"""Tests for NID (Normalized Indel Distance) metric."""

from eval_harness.metrics.parsing.nid import normalized_indel_distance


class TestNID:
    """Test suite for Normalized Indel Distance."""

    def test_identical_sequences(self):
        """Test NID of 0.0 for identical sequences."""
        score = normalized_indel_distance(["A", "B", "C"], ["A", "B", "C"])
        assert score == 0.0

    def test_no_similarity(self):
        """Test NID of 1.0 for completely different sequences."""
        score = normalized_indel_distance(["A"], ["B"])
        assert score == 1.0

    def test_partial_match(self):
        """Test NID for sequences with partial overlap."""
        # "ABC" vs "ABD" - substitution of C for D
        score = normalized_indel_distance(["A", "B", "C"], ["A", "B", "D"])
        # Distance = 1 (substitute C for D), max_len = 3
        # Normalized = 1/3
        assert abs(score - 1 / 3) < 0.01

    def test_insertions_only(self):
        """Test NID with only insertions needed."""
        score = normalized_indel_distance(["A"], ["A", "B", "C"])
        # Distance = 2 (insert B, C), max_len = 3
        assert abs(score - 2 / 3) < 0.01

    def test_deletions_only(self):
        """Test NID with only deletions needed."""
        score = normalized_indel_distance(["A", "B", "C"], ["A"])
        # Distance = 2 (delete B, C), max_len = 3
        assert abs(score - 2 / 3) < 0.01

    def test_empty_sequences(self):
        """Test NID with empty sequences."""
        score = normalized_indel_distance([], [])
        # Both empty - no edit distance
        assert score == 0.0

    def test_one_empty_sequence(self):
        """Test NID with one empty sequence."""
        score = normalized_indel_distance([], ["A", "B"])
        # Need to insert both
        assert score == 1.0

    def test_deterministic_behavior(self):
        """Test that same inputs produce same output."""
        seq1 = ["A", "B", "C", "D"]
        seq2 = ["A", "C", "B", "D"]

        score1 = normalized_indel_distance(seq1, seq2)
        score2 = normalized_indel_distance(seq1, seq2)
        assert score1 == score2

    def test_longer_sequences(self):
        """Test NID with longer sequences."""
        seq1 = list(range(10))
        seq2 = list(range(10))

        score = normalized_indel_distance(seq1, seq2)
        assert score == 0.0

    def test_string_elements(self):
        """Test NID with string elements (words)."""
        pred = ["the", "quick", "brown", "fox"]
        gold = ["the", "brown", "fox"]  # "quick" needs deletion

        score = normalized_indel_distance(pred, gold)
        # Distance = 1 (delete "quick"), max_len = 4
        assert abs(score - 1 / 4) < 0.01
