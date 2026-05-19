"""Tests for text fidelity metric (character-level F1)."""

from eval_harness.metrics.parsing.text_fidelity import text_f1_score


class TestTextFidelity:
    """Test suite for character-level F1 score."""

    def test_perfect_match(self):
        """Test F1 score of 1.0 for perfect match."""
        score = text_f1_score("hello world", "hello world")
        assert score == 1.0

    def test_no_match(self):
        """Test F1 score of 0.0 for completely different strings."""
        score = text_f1_score("abc", "xyz")
        assert score == 0.0

    def test_partial_match(self):
        """Test F1 score for partial overlap."""
        # "hello" vs "hello world" - 5/11 precision, 5/5 recall
        score = text_f1_score("hello", "hello world")
        # F1 = 2 * (5/11 * 5/5) / (5/11 + 5/5) = 2 * (5/11) / (16/11) = 10/16 = 0.625
        assert abs(score - 0.625) < 0.01

    def test_empty_strings(self):
        """Test F1 score handles empty strings."""
        score = text_f1_score("", "")
        # Empty strings should be treated as perfect match
        assert score == 1.0

    def test_one_empty_string(self):
        """Test F1 score when one string is empty."""
        score = text_f1_score("text", "")
        assert score == 0.0

    def test_deterministic_behavior(self):
        """Test that same inputs always produce same output."""
        score1 = text_f1_score("the quick brown fox", "the quick brown fox jumps")
        score2 = text_f1_score("the quick brown fox", "the quick brown fox jumps")
        assert score1 == score2

    def test_whitespace_handling(self):
        """Test that whitespace is treated like other characters."""
        score = text_f1_score("hello world", "hello  world")  # Double space
        # Should be a partial match due to extra space
        assert 0 < score < 1.0

    def test_unicode_characters(self):
        """Test F1 score with unicode characters."""
        score = text_f1_score("hello 世界", "hello 世界")
        assert score == 1.0
